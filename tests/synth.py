"""Deterministic synthetic pose data.

The exactness tests need inputs that are identical across backends, reproducible on any
machine, and shaped like real tracking (smooth trajectories, rigid-ish body geometry,
occasional dropped keypoints). Real recordings are not in the repo, so we synthesize.

Two mice do a correlated random walk with smoothly varying heading; each mouse's seven
usable keypoints are placed on a body frame that rotates with the heading. Missing
keypoints are emitted as NaN, which is exactly the sentinel the model's lag-difference
projections already handle.

    python -m tests.synth /tmp/track --videos 2 --frames 900
"""
import argparse
import os

import numpy as np
import pandas as pd

# The seven bodyparts the model actually consumes, in schema order (BODYPARTS[:7]).
USABLE_BODYPARTS = [
    "tail_base", "ear_right", "ear_left", "nose", "neck", "body_center", "tail_tip",
]

# Offsets along the body frame in cm: +x is the heading direction, +y is the mouse's left.
# Roughly a 9 cm mouse: nose ahead of neck ahead of body_center ahead of tail_base.
_BODY_FRAME_CM = {
    "nose":        (4.5, 0.0),
    "ear_left":    (3.2, 0.9),
    "ear_right":   (3.2, -0.9),
    "neck":        (2.6, 0.0),
    "body_center": (0.0, 0.0),
    "tail_base":   (-3.0, 0.0),
    "tail_tip":    (-7.5, 0.0),
}


def _smooth_walk(rng, n, step, n_smooth=15):
    """A random walk whose increments are box-smoothed, so the path is C1-ish like a real
    centroid track rather than white noise."""
    inc = rng.normal(0.0, step, size=(n + n_smooth, 2))
    kern = np.ones(n_smooth) / n_smooth
    inc = np.stack([np.convolve(inc[:, i], kern, mode="valid")[:n] for i in range(2)], axis=1)
    return np.cumsum(inc, axis=0)


def _interaction_episodes(rng, n_frames, n_episodes=3):
    """Frame windows during which mouse 2 is pulled towards mouse 1 and held there.

    Without this the mice wander independently and every social head sits at its floor
    (~1e-4), which makes an end-to-end probability comparison nearly insensitive. Scripted
    approach-and-hold episodes push the social heads into a range where a real numerical
    discrepancy would actually show up in the probabilities.
    """
    episodes = []
    if n_frames < 200:
        return episodes
    bounds = np.linspace(0, n_frames, n_episodes + 1).astype(int)
    for i in range(n_episodes):
        lo, hi = bounds[i], bounds[i + 1]
        if hi - lo < 150:
            continue
        start = int(rng.integers(lo + 20, hi - 120))
        episodes.append((start, start + int(rng.integers(60, 110))))
    return episodes


def synth_video(n_frames=900, pix_per_cm=16.0, fps=30.0, arena_cm=50.0, n_mice=2,
                missing_rate=0.02, dropout_bodyparts=(), interact=True, seed=0):
    """One video's worth of pose, as a long-format DataFrame.

    dropout_bodyparts  names to omit entirely (simulates a rig that tracks fewer points)
    missing_rate       fraction of (frame, mouse, bodypart) cells emitted as NaN
    interact           script approach-and-hold episodes between mouse 1 and mouse 2
    """
    rng = np.random.default_rng(seed)
    parts = [b for b in USABLE_BODYPARTS if b not in set(dropout_bodyparts)]
    centre = arena_cm / 2.0
    margin = 6.0
    episodes = _interaction_episodes(rng, n_frames) if (interact and n_mice >= 2) else []

    paths = []
    rows = []
    for m in range(n_mice):
        # Centroid path, reflected to stay inside the arena.
        path = _smooth_walk(rng, n_frames, step=0.55) + centre + rng.uniform(-8, 8, size=2)
        lo, hi = margin, arena_cm - margin
        span = 2 * (hi - lo)
        path = np.abs((path - lo) % span)
        path = lo + np.where(path > (hi - lo), span - path, path)

        # Mouse 2 is drawn onto mouse 1's flank during each episode, ramped in and out so
        # the velocity stays continuous (the model reads lag differences, not positions).
        if m == 1 and episodes:
            for start, stop in episodes:
                stop = min(stop, n_frames)
                if stop - start < 10:
                    continue
                ramp = np.minimum(np.arange(stop - start) / 12.0, 1.0)
                ramp = np.minimum(ramp, (stop - start - np.arange(stop - start)) / 12.0)
                ramp = np.clip(ramp, 0.0, 1.0)[:, None]
                contact = paths[0][start:stop] + np.array([3.4, 1.1])
                path[start:stop] = path[start:stop] * (1 - ramp) + contact * ramp
        paths.append(path)

        # Heading follows the direction of travel, smoothed, so the body frame is coherent.
        vel = np.gradient(path, axis=0)
        heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
        kern = np.ones(9) / 9
        heading = np.convolve(heading, kern, mode="same")
        cos_h, sin_h = np.cos(heading), np.sin(heading)

        for part in parts:
            ox, oy = _BODY_FRAME_CM[part]
            # Per-frame jitter on the offset: real keypoints are not perfectly rigid.
            jx = ox + rng.normal(0.0, 0.12, size=n_frames)
            jy = oy + rng.normal(0.0, 0.12, size=n_frames)
            x = path[:, 0] + jx * cos_h - jy * sin_h
            y = path[:, 1] + jx * sin_h + jy * cos_h
            x, y = x * pix_per_cm, y * pix_per_cm

            if missing_rate > 0:
                drop = rng.random(n_frames) < missing_rate
                x = np.where(drop, np.nan, x)
                y = np.where(drop, np.nan, y)

            rows.append(pd.DataFrame({
                "video_frame": np.arange(n_frames, dtype=np.int64),
                "mouse_id": f"mouse{m + 1}",
                "bodypart": part,
                "x": x.astype(np.float64),
                "y": y.astype(np.float64),
            }))

    df = pd.concat(rows, ignore_index=True)
    return df.sort_values(["video_frame", "mouse_id", "bodypart"], ignore_index=True)


def write_dataset(out_dir, n_videos=1, n_frames=900, pix_per_cm=16.0, fps=30.0, seed=0, **kw):
    """Write `n_videos` parquets plus the metadata.csv that HiDRA requires."""
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for i in range(n_videos):
        df = synth_video(n_frames=n_frames, pix_per_cm=pix_per_cm, fps=fps, seed=seed + i, **kw)
        name = f"synth{i:02d}.parquet"
        df.to_parquet(os.path.join(out_dir, name), index=False)
        names.append(name)
    pd.DataFrame([{"file": "*", "pix_per_cm": pix_per_cm, "fps": fps}]).to_csv(
        os.path.join(out_dir, "metadata.csv"), index=False)
    return [os.path.join(out_dir, n) for n in names]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("--videos", type=int, default=1)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--pix-per-cm", type=float, default=16.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    paths = write_dataset(args.out_dir, args.videos, args.frames, args.pix_per_cm, args.fps, args.seed)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
