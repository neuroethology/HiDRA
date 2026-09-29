"""The data pipeline: tracking/label access, windowing, augmentation, and prediction
accumulation.

Split out of `solution.py`, which imports JAX at module scope. Everything here is numpy
only, so the PyTorch backend can read data, batch it and accumulate predictions without the
JAX runtime present. `solution.py` re-exports every name, so `solution.Dataset is
data.Dataset` and existing references keep working.

The JAX original reached for `jax.tree.*` in the stream helpers (`batch`, `unbatch`,
`get_batch_size`, `to_host`). Those calls only ever saw a *flat* dict of arrays -- one
element or one batch -- so each is replaced by the equivalent per-key operation. The one
genuinely JAX-specific step, `Pipeline.to_device`, imports JAX lazily and is only reached
from the JAX backend.

Three module globals name where data lives. Entry points reassign them (each dataset the
inference driver runs has its own tree), so they stay plain strings rather than becoming
lookups. `solution` forwards reads *and writes* of these three onto this module, so
`solution.dataset_dir = ...` still does what it always did.
"""
import ctypes
import functools
import json
import mmap
import multiprocessing
import multiprocessing.sharedctypes
import os
import pickle
import secrets
from collections import defaultdict

import numpy as np
import pandas as pd

from . import paths
from .schema import ACTIONS, BODYPARTS, LABS, MOUSE_IDS, parse_mouse_id

# Where the data and weights live. See the module docstring; `paths.py` resolves the
# defaults for a checkout and for an installed wheel.
project_dir = str(paths.repo_root() or paths.PKG_DIR)
dataset_dir = str(paths.dataset_dir())
working_dir = str(paths.work_root() / "tmp")
persist_dir = str(paths.models_dir())


def _read_at(path, size, offset):
    """Read `size` bytes at byte `offset` from `path`, portably across platforms.

    Two POSIX-isms broke this on Windows: `os.pread` does not exist there, and `os.open`
    without `O_BINARY` opens in text mode, so CRLF and Ctrl-Z (0x1A) translation silently
    corrupts a binary read (the memory-mapped tracking/label cache) and the downstream
    `np.frombuffer(...).reshape(...)` fails with a size mismatch. This keeps the fast atomic
    `pread` path where it exists and falls back to `lseek` + a read loop where it does not;
    `O_BINARY` is a no-op flag (0) off Windows.
    """
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        if hasattr(os, "pread"):
            return os.pread(fd, size, offset)
        os.lseek(fd, offset, os.SEEK_SET)
        parts, remaining = [], size
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            parts.append(chunk)
            remaining -= len(chunk)
        return b"".join(parts)
    finally:
        os.close(fd)


class TrackingData:
    def __init__(self, path, dtype, num_frames, mouse_index, bodypart_index):
        self.path = path
        self.dtype = dtype
        self.num_frames = num_frames
        self.mouse_index = mouse_index
        self.bodypart_index = bodypart_index

        self.num_mice = len(self.mouse_index)
        self.num_bodyparts = len(self.bodypart_index)
        self.num_coords = 2

        self.itemsize = np.dtype(dtype).itemsize
        self.frame_stride = self.num_mice * self.num_bodyparts * self.num_coords * self.itemsize
        self.mouse_to_idx = {self.mouse_index[i]: i for i in range(self.num_mice)}
        self.bodypart_to_idx = {self.bodypart_index[i]: i for i in range(self.num_bodyparts)}

    def __getitem__(self, slices):
        assert isinstance(slices, tuple)
        assert 1 <= len(slices) <= 3

        squeeze_dims = []
        frame_slice = slices[0]
        int_index = isinstance(frame_slice, int)
        if int_index:
            read_offset = frame_slice * self.frame_stride
            read_size = self.frame_stride
            squeeze_dims.append(0)
        else:
            assert isinstance(frame_slice, slice)
            read_offset = frame_slice.start * self.frame_stride
            read_size = (frame_slice.stop - frame_slice.start) * self.frame_stride

        buffer = _read_at(self.path, read_size, read_offset)

        frames = np.frombuffer(buffer, dtype=self.dtype)
        frames = frames.reshape((-1, self.num_mice, self.num_bodyparts, self.num_coords))

        # mouse dim
        if len(slices) > 1 and slices[1] != slice(None, None, None):
            mouse_slice = slices[1]
            if not isinstance(mouse_slice, (list, tuple)):
                mouse_slice = [mouse_slice]
                squeeze_dims.append(1)
            mouse_indices = [self.mouse_to_idx[i] for i in mouse_slice]
            frames = frames[:, mouse_indices]

        # bodypart dim
        bodypart_indices = slice(None, None, None)
        if len(slices) > 2 and slices[2] != slice(None, None, None):
            bodypart_slice = slices[2]
            if not isinstance(bodypart_slice, (list, tuple)):
                bodypart_slice = [bodypart_slice]
                squeeze_dims.append(2)
            bodypart_indices = [self.bodypart_to_idx[i] for i in bodypart_slice]
            frames = frames[:, :, bodypart_indices]

        frames = frames.squeeze(tuple(squeeze_dims)) if squeeze_dims else frames
        return frames


def create_tracking_data(row):
    lab_name = str(row["lab_id"])
    video_id = int(row["video_id"])
    mode = row["mode"]

    tracking_path = f"{dataset_dir}/{mode}_tracking/{lab_name}/{video_id}.parquet"
    tracking_df = pd.read_parquet(tracking_path)

    num_frames = int(tracking_df["video_frame"].max() + 1)
    index_names = ["video_frame", "mouse_id", "bodypart"]
    tracking_df = tracking_df.pivot(index=index_names, columns=[])

    frame_index = np.arange(num_frames)
    mouse_index = tracking_df.index.levels[1]
    bodypart_index = tracking_df.index.levels[2]
    levels = (frame_index, mouse_index, bodypart_index)

    dense_idx = pd.MultiIndex.from_product(levels, names=index_names)
    tracking_df = tracking_df.reindex(dense_idx)
    data_mm = 10 * tracking_df.values.copy() / float(row["pix_per_cm_approx"])
    data_mm[np.all(data_mm == 0.0, axis=-1)] = np.nan

    data_shape = dense_idx.levshape + (2,)
    data_mm = data_mm.reshape(data_shape)

    if lab_name == "MABe22_movies":
        # MABe22_movies keypoints are scrambled
        mouse_ids_mabe = [
            [0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 1, 2],
            [0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 1, 2],
            [0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 1, 2],
        ]
        bodypart_ids_mabe = [
            [4, 8, 8, 3, 3, 4, 4, 3, 8, 7, 7, 7],
            [0, 1, 1, 2, 2, 0, 0, 2, 1, 5, 5, 5],
            [10, 6, 6, 9, 9, 10, 10, 9, 6, 11, 11, 11],
        ]
        mouse_ids_mabe = np.array(mouse_ids_mabe).ravel()
        bodypart_ids_mabe = np.array(bodypart_ids_mabe).ravel()
        data_mm[:, mouse_ids_mabe, bodypart_ids_mabe] = data_mm.reshape(data_mm.shape[0], -1, 2)

    processed_tracking_dir = f"{working_dir}/{mode}/tracking"
    os.makedirs(processed_tracking_dir, exist_ok=True)
    tracking_path = f"{processed_tracking_dir}/{video_id}.bin"
    data_mm.astype("float16").tofile(tracking_path)

    mouse_index = [MOUSE_IDS.encode(parse_mouse_id(m)) for m in mouse_index]
    bodypart_index = list(map(BODYPARTS.encode, bodypart_index))
    return TrackingData(
        path=tracking_path,
        dtype="float16",
        num_frames=num_frames,
        mouse_index=mouse_index,
        bodypart_index=bodypart_index,
    )


class Labels:
    def __init__(self, path, dtype, labeled_behaviors, label_index, label_masks):
        self.path = path
        self.dtype = dtype
        self.labeled_behaviors = labeled_behaviors
        self.label_index = label_index

        self.num_labels = len(self.label_index)
        self.itemsize = np.dtype(dtype).itemsize
        self.frame_stride = self.num_labels * self.itemsize
        self.label_to_idx = {self.label_index[i]: i for i in range(self.num_labels)}
        self.label_masks = label_masks

    def __getitem__(self, slices):
        assert isinstance(slices, tuple)
        assert 1 <= len(slices) <= 2

        squeeze_dims = []
        frame_slice = slices[0]
        int_index = isinstance(frame_slice, int)
        if int_index:
            read_offset = frame_slice * self.frame_stride
            read_size = self.frame_stride
            squeeze_dims.append(0)
        else:
            assert isinstance(frame_slice, slice)
            read_offset = frame_slice.start * self.frame_stride
            read_size = (frame_slice.stop - frame_slice.start) * self.frame_stride

        buffer = _read_at(self.path, read_size, read_offset)

        frames = np.frombuffer(buffer, dtype=self.dtype)
        frames = frames.reshape((-1, self.num_labels))

        # label dim
        if len(slices) > 1 and slices[1] != slice(None, None, None):
            label_slice = slices[1]
            if isinstance(label_slice, tuple) and len(label_slice) == 3:
                assert all(isinstance(elt, int) for elt in label_slice)
                label_slice = [label_slice]
                squeeze_dims.append(1)
            label_indices = [self.label_to_idx[i] for i in label_slice]
            frames = frames[:, label_indices]

        frames = frames.squeeze(tuple(squeeze_dims)) if squeeze_dims else frames
        return frames


def create_labels(row):
    lab_name = str(row["lab_id"])
    video_id = int(row["video_id"])
    mode = row["mode"]

    if pd.isna(row["behaviors_labeled"]):
        behaviors_labeled = []
    else:
        behaviors_labeled = row["behaviors_labeled"].replace("'", "")
        behaviors_labeled = list(set(json.loads(behaviors_labeled)))

    behaviors_labeled = [b for b in behaviors_labeled if "ejaculate" not in b]

    # filter out behaviors where agent/target have labels but don't appear
    # in tracking data.  this only occurs in a few videos from AdaptableSnail
    mouse_pairs = set()
    mouse_pair_label_set = defaultdict(lambda: set())
    filtered_behaviors = set()
    for behavior in behaviors_labeled:
        agent, target, action = behavior.split(",")

        agent = int(agent[-1])
        if target == "self":
            target = agent
        else:
            target = int(target[-1])

        agent_id = MOUSE_IDS.encode(agent)
        target_id = MOUSE_IDS.encode(target)
        action_id = ACTIONS.encode(action)

        if agent_id in row["tracked_mice"] and target_id in row["tracked_mice"]:
            label = (agent_id, target_id, action_id)
            filtered_behaviors.add(label)
            mouse_pairs.add((agent_id, target_id))
            mouse_pair_label_set[(agent_id, target_id)].add(action_id)

    mouse_pairs = list(mouse_pairs)
    mouse_pair_encodings = {mouse_pair: i for i, mouse_pair in enumerate(mouse_pairs)}
    labels = np.zeros([len(mouse_pairs), row["num_frames"]], dtype="int16")  # int16: >127 action ids (cluster vocab)

    label_masks = []
    for mouse_pair in mouse_pairs:
        label_masks.append(list(mouse_pair_label_set[mouse_pair]))

    annotation_path = f"{dataset_dir}/{mode}_annotation/{lab_name}/{video_id}.parquet"
    if os.path.isfile(annotation_path):
        annotation_df = pd.read_parquet(annotation_path)
        behavior_gb = annotation_df.groupby(["agent_id", "target_id", "action"])

        for key, behavior_df in behavior_gb:
            if key[2] == "ejaculate":
                continue
            agent_raw = parse_mouse_id(key[0])
            target_raw = key[1]
            if target_raw == "self":
                target_raw = agent_raw
            else:
                target_raw = parse_mouse_id(target_raw)
            agent_id = MOUSE_IDS.encode(agent_raw)
            target_id = MOUSE_IDS.encode(target_raw)
            action_id = ACTIONS.encode(key[2])
            behavior = (agent_id, target_id, action_id)
            assert behavior in filtered_behaviors

            mouse_pair_encoding = mouse_pair_encodings[(agent_id, target_id)]
            for i in range(behavior_df.shape[0]):
                start_idx = behavior_df.iloc[i]["start_frame"]
                end_idx = behavior_df.iloc[i]["stop_frame"]
                labels[mouse_pair_encoding, start_idx:end_idx] = action_id

    processed_labels_dir = f"{working_dir}/{mode}/labels"
    os.makedirs(processed_labels_dir, exist_ok=True)
    labels_path = f"{processed_labels_dir}/{video_id}.bin"
    labels.T.tofile(labels_path)

    labeled_behaviors = list(filtered_behaviors)
    return Labels(
        path=labels_path,
        dtype=labels.dtype,
        labeled_behaviors=labeled_behaviors,
        label_index=mouse_pairs,
        label_masks=label_masks,
    )


class Video:
    def __init__(self, video_id, lab_name, tracking_data, labels, fps):
        self.lab_name = lab_name
        self.video_id = video_id
        self.tracking_data = tracking_data
        self.labels = labels
        self.num_frames = tracking_data.num_frames
        self.fps = fps
        self.duration = self.num_frames / self.fps


def create_video(row_idx, row):
    tracking_data = create_tracking_data(row)
    row["tracked_mice"] = tracking_data.mouse_index
    row["num_frames"] = tracking_data.num_frames
    labels = create_labels(row)
    return Video(
        video_id=int(row["video_id"]),
        lab_name=str(row["lab_id"]),
        tracking_data=tracking_data,
        labels=labels,
        fps=float(row["frames_per_second"]),
    )


def load_videos(mode, use_cached=True):
    videos_path = f"{working_dir}/{mode}/videos.pkl"
    if use_cached and os.path.isfile(videos_path):
        return pickle.load(open(videos_path, "rb"))

    df = pd.read_csv(f"{dataset_dir}/{mode}.csv")
    df["mode"] = mode
    videos = [create_video(idx, row) for idx, row in df.iterrows()]

    with open(videos_path, "wb") as f:
        f.write(pickle.dumps(videos))

    return videos


def split_videos(videos, validation_frac, random_seed):
    rng = np.random.default_rng(random_seed)

    videos_by_lab = defaultdict(list)
    for video in videos:
        videos_by_lab[video.lab_name].append(video)

    train_videos, val_videos = [], []
    for lab_name, lab_videos in videos_by_lab.items():
        val_indices = set()
        if len(lab_videos) != 1:
            indices = np.arange(len(lab_videos))
            durations = np.array([video.duration for video in lab_videos])
            p = durations / durations.sum()
            permutation = rng.choice(indices, size=[len(lab_videos) - 1], p=p, replace=False)
            val_duration_frac = 0
            for idx in permutation:
                val_indices.add(idx)
                val_duration_frac += p[idx]
                if val_duration_frac > validation_frac:
                    break

        for i, video in enumerate(lab_videos):
            split = val_videos if i in val_indices else train_videos
            split.append(video)

    return train_videos, val_videos


def flat_seed(*parts):
    """Flatten a seed spec into the 1-D list np.random.SeedSequence requires.

    The original code wrote `default_rng([epoch_idx, dataset.seed])` where `dataset.seed`
    is itself a list, relying on numpy coercing the nested sequence. numpy 2.5 removed that
    coercion ("SeedSequence does not accept nested sequences"), so the nesting is flattened
    explicitly here. Flattening reproduces the *same* entropy numpy derived from the nested
    form, so the random stream -- and therefore every prediction -- is unchanged; that
    equivalence is pinned by tests/test_seed_flattening.py.
    """
    out = []
    for part in parts:
        if isinstance(part, (list, tuple)):
            out.extend(flat_seed(*part))
        else:
            out.append(int(part))
    return out


def shared_array(shape, dtype):
    num_elts = int(np.prod(shape))
    elt_size = np.dtype(dtype).itemsize
    buffer_size = num_elts * elt_size
    buffer = mmap.mmap(-1, buffer_size)
    return np.ndarray(shape=shape, dtype=dtype, buffer=buffer)


class RingBuffer:
    def __init__(self, dtype, max_items):
        self.dtype = dtype
        self.max_items = max_items
        self.array = shared_array([max_items], dtype)

        ptr_dtype = ctypes.c_int32 if self.max_items < 2**31 else ctypes.c_int64
        self.head = multiprocessing.sharedctypes.RawValue(ptr_dtype)
        self.tail = multiprocessing.sharedctypes.RawValue(ptr_dtype)

        self.lock = multiprocessing.get_context("fork").Lock()
        self.filled_slots = multiprocessing.get_context("fork").Semaphore(0)
        self.vacant_slots = multiprocessing.get_context("fork").Semaphore(self.max_items)

    def put(self, item):
        self.vacant_slots.acquire()
        with self.lock:
            tail = int(self.tail.value)
            self.array[tail] = item
            self.tail.value = (tail + 1) % self.max_items
        self.filled_slots.release()

    def get(self):
        self.filled_slots.acquire()
        with self.lock:
            head = int(self.head.value)
            item = self.array[head]
            self.head.value = (head + 1) % self.max_items
        self.vacant_slots.release()
        return item


class Epoch:
    def __init__(self, dataset, rows, epoch_idx, shuffle=True):
        self.dataset = dataset
        self.rows = rows
        self.epoch_idx = epoch_idx
        self.shuffle = shuffle
        self.n = len(rows)
        self.seed = self.dataset.seed
        self.num_buffered_elements = self.dataset.num_buffered_elements
        self.num_workers = self.dataset.num_workers

        self.row_ptr_dtype = "int32" if self.n < 2**31 else "int64"
        self.ptr_dtype = "int32" if self.num_buffered_elements < 2**31 else "int64"
        self.input_dtype = np.dtype([("element_idx", self.row_ptr_dtype), ("row_idx", self.row_ptr_dtype)])
        self.input_queue = RingBuffer(dtype=self.input_dtype, max_items=self.num_buffered_elements)
        self.output_queue = RingBuffer(dtype=self.ptr_dtype, max_items=self.num_buffered_elements)
        self.vacant_slots = multiprocessing.get_context("fork").Semaphore(self.num_buffered_elements)

        self.initialize_element_buffers()
        self.enqueue_worker = multiprocessing.get_context("fork").Process(target=self.enqueue_worker, daemon=True)
        self.enqueue_worker.start()
        self.next_idx = 0

        self.workers = []
        for i in range(self.num_workers):
            worker = multiprocessing.get_context("fork").Process(target=self.worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)

        self.dequeued_elements = {}

    def initialize_element_buffers(self):
        ctx = {"epoch_idx": 0, "element_idx": 0, "row_idx": 0}
        element = self.dataset.get_element(self.rows[0], ctx=ctx)
        n = self.num_buffered_elements

        self.element_buffers = {}
        for k, v in element.items():
            self.element_buffers[k] = shared_array((n,) + v.shape, v.dtype)

    def enqueue_worker(self):
        if self.shuffle:
            rng = np.random.default_rng(flat_seed(self.epoch_idx, self.seed))
            rng.bit_generator.state = rng.bit_generator.jumped().state
            permutation = rng.permutation(self.n)
        else:
            permutation = np.arange(self.n)
        for element_idx, row_idx in enumerate(permutation):
            self.vacant_slots.acquire()
            item = np.array((element_idx, row_idx), self.input_dtype)
            self.input_queue.put(item)

        for i in range(self.num_workers):
            self.vacant_slots.acquire()
            item = np.array((self.n, self.n), self.input_dtype)
            self.input_queue.put(item)

    def write_element(self, element, buffer_idx):
        for k, v in element.items():
            self.element_buffers[k][buffer_idx] = v

    def read_element(self, buffer_idx):
        element = {}
        for k, v in self.element_buffers.items():
            element[k] = v[buffer_idx].copy()
        return element

    def worker_loop(self):
        while True:
            item = self.input_queue.get()
            element_idx, row_idx = item["element_idx"], item["row_idx"]
            if element_idx == self.n:
                return

            ctx = {"epoch_idx": self.epoch_idx, "element_idx": element_idx, "row_idx": row_idx}
            row = self.rows[row_idx]
            element = self.dataset.get_element(row, ctx)
            buffer_idx = element_idx % self.num_buffered_elements
            self.write_element(element, buffer_idx)
            self.output_queue.put(element_idx)

    def join(self):
        self.enqueue_worker.join()
        for worker in self.workers:
            worker.join()

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_idx == self.n:
            self.join()
            raise StopIteration

        while self.next_idx not in self.dequeued_elements:
            element_idx = int(self.output_queue.get())
            element = self.read_element(element_idx % self.num_buffered_elements)
            self.dequeued_elements[element_idx] = element

        element = self.dequeued_elements.pop(self.next_idx)
        self.next_idx += 1
        self.vacant_slots.release()
        return element


class SimpleEpoch:
    """Single-process epoch iterator that avoids fork deadlocks with JAX."""

    def __init__(self, dataset, rows, epoch_idx, shuffle=True):
        self.dataset = dataset
        self.rows = rows
        self.epoch_idx = epoch_idx
        self.shuffle = shuffle
        self.n = len(rows)

        if shuffle:
            rng = np.random.default_rng(flat_seed(epoch_idx, dataset.seed))
            rng.bit_generator.state = rng.bit_generator.jumped().state
            self.permutation = rng.permutation(self.n)
        else:
            self.permutation = np.arange(self.n)

        self.next_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_idx >= self.n:
            raise StopIteration

        row_idx = self.permutation[self.next_idx]
        ctx = {"epoch_idx": self.epoch_idx, "element_idx": self.next_idx, "row_idx": row_idx}
        element = self.dataset.get_element(self.rows[row_idx], ctx)
        self.next_idx += 1
        return element


class BaseDataset:
    def __init__(self, num_workers=1, num_buffered_elements=128, seed=None):
        self.num_workers = num_workers
        self.num_buffered_elements = num_buffered_elements
        self.seed = secrets.randbits(128) if seed is None else seed

    def get_element(self, row):
        raise NotImplementedError

    def epoch_generator(self):
        raise NotImplementedError

    def epochs(self):
        for epoch_idx, epoch in enumerate(self.epoch_generator()):
            yield SimpleEpoch(self, epoch, epoch_idx)

    def element_iterator(self):
        for epoch in self.epochs():
            for element in epoch:
                yield element


class Dataset(BaseDataset):
    def __init__(
        self,
        videos,
        seq_len,
        sample_rate,
        padding,
        num_bodyparts,
        num_epochs,
        unsupervised,
        max_scale=1,
        max_time_dilation=1,
        rotate=False,
        flip=False,
        noise_scale=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.videos = videos
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.padding = padding
        self.input_seq_len = seq_len + 2 * padding
        self.output_seq_len = seq_len
        self.num_epochs = num_epochs
        self.num_bodyparts = num_bodyparts
        self.unsupervised = unsupervised
        self.max_scale = max_scale
        self.max_time_dilation = max_time_dilation
        self.noise_scale = noise_scale
        self.rotate = rotate
        self.flip = flip

        self.epoch_seed = flat_seed(0, self.seed)
        self.worker_seed = flat_seed(1, self.seed)

    def epochs_supervised(self):
        rng = np.random.default_rng(self.epoch_seed)
        for epoch in range(self.num_epochs):
            epoch_keys = []
            for video_idx, video in enumerate(self.videos):
                # (AdaptableSnail@25fps videos are NOW included in supervised training:
                # their annotations were rebuilt from raw + ID-permutation-corrected 2026-07-09.
                # The former `continue` skip has been removed.)

                agent_to_targets = defaultdict(list)
                mice = set()
                agents = set()
                self_pairs = set()
                cross_pairs = defaultdict(list)
                for agent, target in video.labels.label_index:
                    agent_to_targets[agent].append(target)
                    mice.add(agent)
                    mice.add(target)
                    agents.add(agent)
                    if agent == target:
                        self_pairs.add(agent)
                    else:
                        cross_pairs[agent].append(target)

                mouse_pairs = []
                for agent in agents:
                    if len(cross_pairs[agent]) == 0:
                        # An agent with only a self-label needs a partner to fill the pair's target
                        # slot, but its cross prediction is a dummy (cross_label = -1, masked out of
                        # the loss). A single-mouse video has no other animal, so `rng.choice([])`
                        # raised "a cannot be empty" and killed supervised training on any
                        # single-animal clip. Fall back to the agent itself when it is the only mouse:
                        # the target poses then duplicate the agent's, but nothing supervises them, so
                        # the self-label still trains correctly.
                        others = [i for i in mice if i != agent]
                        target = rng.choice(others) if others else agent

                        mouse_pair = (agent, target)
                        self_label = video.labels.label_to_idx[(agent, agent)]
                        cross_label = -1
                        mouse_pairs.append((mouse_pair, self_label, cross_label))

                    else:
                        self_label = -1
                        if agent in self_pairs:
                            self_label = video.labels.label_to_idx[(agent, agent)]

                        for target in cross_pairs[agent]:
                            mouse_pair = (agent, target)
                            cross_label = video.labels.label_to_idx[(agent, target)]
                            mouse_pairs.append((mouse_pair, self_label, cross_label))

                video_duration = video.num_frames / video.fps
                output_seq_duration = self.output_seq_len / self.sample_rate
                min_chunk_duration = output_seq_duration / self.max_time_dilation
                max_chunk_duration = output_seq_duration * self.max_time_dilation
                for mouse_pair in mouse_pairs:
                    data_pair, self_label, cross_label = mouse_pair

                    h = min(output_seq_duration, video_duration - max_chunk_duration)
                    h = max(h, 0)
                    t0 = rng.uniform(low=0, high=h)

                    max_chunks = int((video_duration - t0) // min_chunk_duration)
                    u = rng.uniform(low=-1, high=1.0, size=max_chunks).astype("float32")
                    chunk_durations = output_seq_duration * np.power(self.max_time_dilation, u)
                    chunk_boundaries = t0 + np.insert(chunk_durations, 0, 0).cumsum()
                    max_idx = np.searchsorted(chunk_boundaries, video_duration)

                    chunk_starts = chunk_boundaries[: max_idx - 1]
                    chunk_ends = chunk_boundaries[1:max_idx]

                    if chunk_starts.shape[0] == 0:
                        continue

                    assert np.abs(chunk_starts[0] - t0) < 1e-3
                    assert chunk_ends[-1] <= video_duration
                    min_obs_duration = (chunk_ends - chunk_starts).min()
                    assert min_obs_duration >= (min_chunk_duration - 1e-2)
                    max_obs_duration = (chunk_ends - chunk_starts).max()
                    assert max_obs_duration <= (max_chunk_duration + 1e-2)

                    video_keys = np.empty([7, chunk_starts.shape[0]], dtype="float32")
                    video_keys[0] = video_idx
                    video_keys[1] = data_pair[0]
                    video_keys[2] = data_pair[1]
                    video_keys[3] = self_label
                    video_keys[4] = cross_label
                    video_keys[5] = chunk_starts
                    video_keys[6] = chunk_ends
                    epoch_keys.append(video_keys.T)

            epoch_keys = np.concatenate(epoch_keys, axis=0)
            yield epoch_keys

    def epochs_unsupervised(self):
        rng = np.random.default_rng(self.epoch_seed)
        for epoch in range(self.num_epochs):
            epoch_keys = []
            for video_idx, video in enumerate(self.videos):
                mouse_index = video.tracking_data.mouse_index
                rng.shuffle(mouse_index)
                mouse_pairs = []
                mouse_pairs.append((mouse_index[0], mouse_index[1]))
                if len(mouse_index) == 4:
                    mouse_pairs = [(mouse_index[2], mouse_index[3])]

                video_duration = video.num_frames / video.fps
                output_seq_duration = self.output_seq_len / self.sample_rate
                min_chunk_duration = output_seq_duration / self.max_time_dilation
                max_chunk_duration = output_seq_duration * self.max_time_dilation
                for agent_id, target_id in mouse_pairs:
                    h = min(output_seq_duration, video_duration - max_chunk_duration)
                    h = max(h, 0)
                    t0 = rng.uniform(low=0, high=h)

                    max_chunks = int((video_duration - t0) // min_chunk_duration)
                    u = rng.uniform(low=-1, high=1.0, size=max_chunks).astype("float32")
                    chunk_durations = output_seq_duration * np.power(self.max_time_dilation, u)
                    chunk_boundaries = t0 + np.insert(chunk_durations, 0, 0).cumsum()
                    max_idx = np.searchsorted(chunk_boundaries, video_duration)
                    chunk_starts = chunk_boundaries[: max_idx - 1]
                    chunk_ends = chunk_boundaries[1:max_idx]

                    if chunk_starts.shape[0] == 0:
                        continue

                    assert np.abs(chunk_starts[0] - t0) < 1e-3
                    assert chunk_ends[-1] <= video_duration
                    min_obs_duration = (chunk_ends - chunk_starts).min()
                    assert min_obs_duration >= (min_chunk_duration - 1e-2)
                    max_obs_duration = (chunk_ends - chunk_starts).max()
                    assert max_obs_duration <= (max_chunk_duration + 1e-2)

                    video_keys = np.empty([7, chunk_starts.shape[0]], dtype="float32")
                    video_keys[0] = video_idx
                    video_keys[1] = agent_id
                    video_keys[2] = target_id
                    video_keys[3] = 0
                    video_keys[4] = 0
                    video_keys[5] = chunk_starts
                    video_keys[6] = chunk_ends
                    epoch_keys.append(video_keys.T)

            epoch_keys = np.concatenate(epoch_keys, axis=0)
            yield epoch_keys

    def epoch_generator(self):
        epoch_gen = self.epochs_unsupervised if self.unsupervised else self.epochs_supervised
        yield from epoch_gen()

    def augment(self, element, ctx):
        def rotate(coords, radians):
            x, y = coords[..., 0], coords[..., 1]
            xx = x * np.cos(radians) + y * np.sin(radians)
            yy = -x * np.sin(radians) + y * np.cos(radians)
            return np.stack([xx, yy], axis=-1)

        rng = np.random.default_rng(flat_seed(ctx["element_idx"], ctx["epoch_idx"], self.worker_seed))

        agent, target = element["agent"], element["target"]
        if self.noise_scale is not None and self.noise_scale != 0.0:
            half = self.noise_scale / 2
            noise = rng.uniform(low=-half, high=half, size=agent.shape)
            agent = agent + noise.astype("float32")

            noise = rng.uniform(low=-half, high=half, size=target.shape)
            target = target + noise.astype("float32")

        scale = 1.0
        if self.max_scale is not None and self.max_scale != 1.0:
            u = rng.uniform(low=-1.0, high=1.0)
            scale = np.power(self.max_scale, u, dtype="float32")
            agent = agent * scale
            target = target * scale

        rotate_x, rotate_y = 1, 0
        if self.rotate:
            theta = rng.uniform(low=0, high=2 * np.pi)
            agent = rotate(agent, theta)
            target = rotate(target, theta)
            rotate_x = np.cos(theta)
            rotate_y = np.sin(theta)

        flip_x, flip_y = 0, 0
        if self.flip:
            flip = rng.uniform(size=[2]) > 0.5
            agent = np.where(flip[None, None], -agent, agent)
            target = np.where(flip[None, None], -target, target)
            flip_x, flip_y = flip[0], flip[1]

        augmentation_params = [element["time_dilation"], scale, rotate_x, rotate_y, flip_x, flip_y]
        augmentation_params = np.array(augmentation_params, dtype="float32")

        element["agent"] = agent
        element["target"] = target
        element["augmentation_params"] = augmentation_params
        return element

    def get_element(self, key, ctx):
        video_idx = int(key[0])
        agent_id = int(key[1])
        target_id = int(key[2])
        self_label = int(key[3])
        cross_label = int(key[4])
        t_start = float(key[5])
        t_end = float(key[6])

        video = self.videos[video_idx]

        in_out_ratio = self.input_seq_len / self.output_seq_len
        out_duration = t_end - t_start
        in_duration = out_duration * in_out_ratio

        t_mid = (t_start + t_end) / 2
        input_t_start = t_mid - (in_duration / 2)
        input_t_end = t_mid + (in_duration / 2)
        input_ts = np.linspace(input_t_start, input_t_end, self.input_seq_len, dtype="float32")

        input_start_frame = int(max(np.floor(input_t_start * video.fps), 0))
        input_end_frame = int(min(np.ceil(input_t_end * video.fps) + 1, video.num_frames))
        input_num_frames = input_end_frame - input_start_frame   # label-window length (below)
        input_sparse = video.tracking_data[input_start_frame:input_end_frame, [agent_id, target_id]]
        # The slice can return FEWER rows than requested (or zero) when the
        # augmented window falls past the available tracking -- e.g. video.num_frames
        # (metadata) exceeds the actual tracking length, or time-dilation pushes the
        # window past the end. Key off the ACTUAL rows returned (n_avail) instead of
        # assuming the full window; emit all-nan (the model's missing-keypoint
        # sentinel) when nothing is available. Identical to the original in the
        # normal case where the slice returns the full window.
        n_avail = input_sparse.shape[0]
        agent_target = np.full([self.input_seq_len, 2, len(BODYPARTS), 2], np.nan, dtype="float32")
        if n_avail > 0:
            tck_ts = (input_start_frame + np.arange(n_avail, dtype="float32")) / video.fps
            input_crop_flat = input_sparse.reshape(n_avail, -1).T
            agent_target_flat = np.empty([input_crop_flat.shape[0], self.input_seq_len], dtype="float32")
            for i in range(input_crop_flat.shape[0]):
                agent_target_flat[i] = np.interp(input_ts, tck_ts, input_crop_flat[i], left=np.nan, right=np.nan)
            agent_target_sparse = agent_target_flat.T.reshape((self.input_seq_len,) + input_sparse.shape[1:])
            agent_target[:, :, video.tracking_data.bodypart_index] = agent_target_sparse
        agent_target = agent_target[:, :, : self.num_bodyparts]
        agent, target = agent_target[:, 0], agent_target[:, 1]

        self_labels = np.zeros([self.output_seq_len], dtype="int16")
        cross_labels = np.zeros([self.output_seq_len], dtype="int16")
        self_label_mask = np.zeros([len(ACTIONS)], dtype="int8")
        cross_label_mask = np.zeros([len(ACTIONS)], dtype="int8")
        if not self.unsupervised:
            labels = video.labels[input_start_frame:input_end_frame, :]

            # video.labels can be shorter than video.num_frames (tracking) for
            # some videos, so an augmented window may extend past the labels'
            # end -> the `labels` slice is shorter than input_num_frames. Keep
            # both label tracks at input_num_frames (tail past labels = 0/none)
            # so np.stack below never sees a shape mismatch.
            n_lab = labels.shape[0]
            self_labels = np.zeros([input_num_frames], dtype="int16")
            cross_labels = np.zeros([input_num_frames], dtype="int16")
            if self_label != -1:
                self_labels[:n_lab] = labels[:n_lab, self_label]
                for label_id in video.labels.label_masks[self_label]:
                    self_label_mask[label_id] = 1
            if cross_label != -1:
                cross_labels[:n_lab] = labels[:n_lab, cross_label]
                for label_id in video.labels.label_masks[cross_label]:
                    cross_label_mask[label_id] = 1

            labels = np.stack([self_labels, cross_labels], axis=1)
            labels = np.eye(len(ACTIONS), dtype="float32")[labels]

            # label timestamps span the full requested window (length
            # input_num_frames), independent of how many tracking rows were
            # available; defined here so it exists even when n_avail == 0 above.
            input_crop_ts = (input_start_frame + np.arange(input_num_frames, dtype="float32")) / video.fps
            labels_flat = labels.reshape(input_num_frames, -1).T
            labels_flat_resampled = np.zeros([labels_flat.shape[0], self.input_seq_len], dtype="float32")
            for i in range(labels_flat.shape[0]):
                input_i = labels_flat[i]
                if np.any(input_i != 0):
                    labels_flat_resampled[i] = np.interp(input_ts, input_crop_ts, input_i)
            labels_flat_resampled = labels_flat_resampled.T.reshape((self.input_seq_len,) + labels.shape[1:])
            labels_flat_resampled = np.argmax(labels_flat_resampled, axis=-1).astype("int16")
            pad = self.input_seq_len - self.output_seq_len
            lpad = pad // 2
            rpad = pad - lpad
            labels = labels_flat_resampled[lpad : labels_flat_resampled.shape[0] - rpad]

            self_labels, cross_labels = labels[:, 0], labels[:, 1]

        output_seq_duration = self.output_seq_len / self.sample_rate
        time_dilation = (t_end - t_start) / output_seq_duration
        element = {
            "agent": agent,
            "target": target,
            "self_labels": self_labels,
            "cross_labels": cross_labels,
            "self_label_mask": self_label_mask,
            "cross_label_mask": cross_label_mask,
            "video_id": np.int32(video.video_id),
            "t_start": np.float32(t_start),
            "t_end": np.float32(t_end),
            "lab_id": np.int32(LABS.encode(video.lab_name)),
            "agent_id": np.int32(agent_id),
            "target_id": np.int32(target_id),
            "video_fps": np.float32(video.fps),
            "sample_rate": np.float32(self.sample_rate),
            "time_dilation": np.float32(time_dilation),
        }
        element = self.augment(element, ctx)
        return element


# ---------------------------------------------------------------------------------------
# Minimal pytree helpers, standing in for the `jax.tree.*` calls the stream helpers used.
#
# These are NOT only over flat dicts. `run_test_probs_perlab.predict_into` maps the stream
# through a function returning `(batch, probs)`, so everything downstream of it --
# `to_host`, `unbatch`, `get_batch_size` -- sees a *tuple* whose first element is the batch
# dict and whose second is a probability array. jax.tree handled that structure for free;
# these reproduce it for dicts, lists and tuples, which is every container the pipeline
# actually carries.


def tree_map(fn, tree):
    """Apply `fn` to every leaf, preserving the container structure."""
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(tree_map(fn, v) for v in tree)
    if isinstance(tree, list):
        return [tree_map(fn, v) for v in tree]
    return fn(tree)


def tree_leaves(tree):
    """Every leaf, in container order."""
    if isinstance(tree, dict):
        for value in tree.values():
            yield from tree_leaves(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from tree_leaves(value)
    else:
        yield tree


def tree_stack(trees):
    """Stack a list of identically-structured trees into one tree of stacked leaves.

    The equivalent of the original's `jax.tree.transpose` followed by
    `jax.tree.map(lambda x, y: np.stack(y), ...)`.
    """
    first = trees[0]
    if isinstance(first, dict):
        return {k: tree_stack([t[k] for t in trees]) for k in first}
    if isinstance(first, tuple):
        return tuple(tree_stack([t[i] for t in trees]) for i in range(len(first)))
    if isinstance(first, list):
        return [tree_stack([t[i] for t in trees]) for i in range(len(first))]
    return np.stack(trees)


def take(stream, n):
    for i in range(n):
        yield next(stream)


def batch(stream, n):
    while True:
        batch_elements = []
        for i in range(n):
            try:
                element = next(stream)
                element["batch_mask"] = np.int8(1)
                batch_elements.append(element)
            except StopIteration:
                break

        if len(batch_elements) == 0:
            return

        if len(batch_elements) < n:
            # np.empty_like, i.e. UNINITIALIZED memory, exactly as the original: the padding
            # rows are flagged batch_mask=0 and dropped by Predictions.update, so their
            # contents never reach an output. They are not harmless everywhere, though --
            # `lab_id` then holds garbage that would index out of bounds; see
            # hidra/torch/layers.py:jax_gather_index.
            pad_elt = tree_map(np.empty_like, batch_elements[-1])
            pad_elt["batch_mask"] = np.int8(0)
            num_pad_elts = n - len(batch_elements)
            for i in range(num_pad_elts):
                batch_elements.append(pad_elt)

        yield tree_stack(batch_elements)


def to_device(stream, sharding):
    """Move batches onto the JAX device. Only the JAX backend reaches this.

    JAX is imported here rather than at module scope so that this module -- and with it the
    whole numpy data pipeline -- stays importable without the JAX runtime.
    """
    import jax
    import jax.numpy as jnp

    for item in stream:
        if sharding is not None:
            yield jax.make_array_from_process_local_data(sharding, item)
        else:
            yield tree_map(jnp.array, item)


def to_host(stream):
    def delay(stream):
        prev_item = None
        prev_item_loaded = False
        for item in stream:
            if prev_item_loaded:
                yield prev_item
            prev_item = item
            prev_item_loaded = True

        if prev_item_loaded:
            yield prev_item

    for item in delay(stream):
        yield tree_map(np.array, item)


def get_batch_size(item):
    batch_size = None
    for leaf in tree_leaves(item):
        if batch_size is None:
            batch_size = leaf.shape[0]
        assert batch_size == leaf.shape[0]
    return batch_size


def unbatch(stream):
    for batch in stream:
        batch_size = get_batch_size(batch)
        for i in range(batch_size):
            element = tree_map(lambda x: x[i].copy(), batch)
            yield element


def average_metrics(metrics_stream):
    def accumulate_metrics(metrics_state, metrics):
        if metrics_state is None:
            metrics_state = {k: [np.float64(0.0), np.float64(0.0)] for k in metrics}
        for k, v in metrics.items():
            metrics_state[k][0] += v[0] * v[1]
            metrics_state[k][1] += v[1]
        return metrics_state

    metrics_sum = functools.reduce(accumulate_metrics, metrics_stream, None)
    metrics = {k: float(v[0] / v[1]) for k, v in metrics_sum.items()}
    return metrics


class Pipeline:
    def __init__(self, stream):
        self.stream = stream

    @staticmethod
    def from_stream(stream):
        return Pipeline(stream)

    def map(self, fn):
        return Pipeline(map(fn, self.stream))

    def batch(self, batch_size):
        return Pipeline(batch(self.stream, batch_size))

    def take(self, n):
        return Pipeline(take(self.stream, n))

    def to_device(self, sharding):
        return Pipeline(to_device(self.stream, sharding))

    def to_host(self):
        return Pipeline(to_host(self.stream))

    def unbatch(self):
        return Pipeline(unbatch(self.stream))

    def last(self):
        last = None
        for item in self:
            last = item
        return last

    def average_metrics(self):
        return average_metrics(self.stream)

    def __iter__(self):
        yield from self.stream


class F1:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.nums = {threshold: 0 for threshold in thresholds}
        self.dens = {threshold: 0 for threshold in thresholds}

    def update(self, probs, labels):
        for threshold in self.thresholds:
            action_preds = probs > (threshold / 100)
            tp = np.sum(action_preds & labels)
            fp_fn = np.sum(np.bitwise_xor(action_preds, labels))
            self.nums[threshold] += 2 * tp
            self.dens[threshold] += 2 * tp + fp_fn

    def evaluate(self, key="argmax"):
        f1s = {}
        for t in self.thresholds:
            f1s[t] = 0 if self.dens[t] == 0 else self.nums[t] / self.dens[t]

        if key == "argmax":
            best_f1, best_thresh = -1, None
            for threshold in self.thresholds:
                if f1s[threshold] > best_f1:
                    best_f1 = f1s[threshold]
                    best_thresh = threshold

            return best_f1, best_thresh
        else:
            return f1s[key], key


class Predictions:
    def __init__(self, videos):
        self.videos = videos
        self.sums = {}
        self.weights = {}
        for video in videos:
            for behavior in video.labels.labeled_behaviors:
                agent, target, action = behavior
                key = (video.video_id, agent, target, action)
                self.sums[key] = np.zeros([video.num_frames], dtype="float32")
                self.weights[key] = np.zeros([video.num_frames], dtype="float32")

    def update(self, element, probs):
        if element["batch_mask"] == 0:
            return

        probs = probs.T
        n_actions, seq_len = probs.shape

        video_id = int(element["video_id"])
        agent = int(element["agent_id"])
        target = int(element["target_id"])
        t_start = float(element["t_start"])
        t_end = float(element["t_end"])
        self_mask = element["self_label_mask"]
        cross_mask = element["cross_label_mask"]
        video_fps = float(element["video_fps"])

        input_seq_len = element["agent"].shape[0]
        output_seq_len = seq_len
        in_out_ratio = input_seq_len / output_seq_len
        out_duration = t_end - t_start
        in_duration = out_duration * in_out_ratio

        t_mid = (t_start + t_end) / 2
        input_t_start = t_mid - (in_duration / 2)
        input_t_end = t_mid + (in_duration / 2)
        input_ts = np.linspace(input_t_start, input_t_end, input_seq_len, dtype="float32")
        pad = input_seq_len - output_seq_len
        lpad = pad // 2
        rpad = pad - lpad
        probs_ts = input_ts[lpad : input_seq_len - rpad]

        output_start_frame = int(np.floor(probs_ts[0] * video_fps))
        output_end_frame = int(np.ceil(probs_ts[-1] * video_fps))

        for action in range(n_actions):
            if self_mask[action] == 1 or cross_mask[action] == 1:
                target2 = agent if self_mask[action] == 1 else target
                key = (video_id, agent, target2, action)

                max_frames = self.sums[key].shape[0]
                output_end_frame = min(output_end_frame, max_frames - 1)
                output_start_frame = max(output_start_frame, 0)
                output_idx = np.arange(output_start_frame, output_end_frame + 1)
                output_ts = output_idx / video_fps

                probs_i = np.interp(output_ts, probs_ts, probs[action])
                self.sums[key][output_start_frame : output_end_frame + 1] += probs_i
                self.weights[key][output_start_frame : output_end_frame + 1] += 1

    def average_probs(self):
        average_probs = {}
        for key in self.sums:
            if np.all(self.weights[key] == 0):
                average_probs[key] = np.full_like(self.sums[key], 0.05)
            else:
                mask = self.weights[key] != 0
                probs = np.divide(self.sums[key], self.weights[key], where=mask, out=None)
                probs[~mask] = np.interp(np.flatnonzero(~mask), np.flatnonzero(mask), probs[mask])
                average_probs[key] = probs
        return average_probs

    def score(self):
        probs_dict = self.average_probs()
        nll_sum, nll_weight = np.float64(0.0), np.float64(0.0)

        thresholds = list(range(0, 100))
        f1s = defaultdict(lambda: defaultdict(lambda: F1(thresholds)))
        for video in self.videos:
            for behavior in video.labels.labeled_behaviors:
                agent, target, action = behavior
                label_idx = video.labels.label_to_idx[(agent, target)]
                labels = video.labels[0 : video.num_frames, :][:, label_idx]
                action_labels = labels == action

                probs = probs_dict[(video.video_id, agent, target, action)]
                probs = probs.astype("float64")
                nll = -np.where(action_labels, np.log(probs), np.log(1 - probs))
                nll_sum += nll.sum()
                nll_weight += nll.shape[0]

                f1s[video.lab_name][action].update(probs, action_labels)

        lab_f1s = []
        lab_action_thresholds = defaultdict(dict)
        for lab_name in f1s:
            action_f1s = []
            for action in f1s[lab_name]:
                action_f1, action_thresh = f1s[lab_name][action].evaluate()
                action_f1s.append(action_f1)
                lab_action_thresholds[lab_name][action] = action_thresh
            lab_f1s.append(np.mean(action_f1s))
        f1 = np.mean(lab_f1s)

        nll = nll_sum / nll_weight
        metrics = {"nll": nll, "f1": f1}
        return metrics, lab_action_thresholds

    def to_submission_df(self):
        thresholds = pickle.load(open(f"{persist_dir}/thresholds.pkl", "rb"))
        video_id_to_video = {video.video_id: video for video in self.videos}

        submission_df = {
            "lab": [],
            "video_id": [],
            "agent_id": [],
            "target_id": [],
            "action": [],
            "start_frame": [],
            "stop_frame": [],
        }
        best_scores = {}
        probs_dict = self.average_probs()
        for key, probs in probs_dict.items():
            video_id, agent, target, action = key

            video = video_id_to_video[video_id]
            lab_name = video.lab_name
            action_name = ACTIONS.decode(action)
            threshold = thresholds.get((lab_name, action_name), 30) / 100

            pred_frames = np.where(probs > threshold)[0]
            for pred_frame in pred_frames:
                key = (video_id, agent, target, pred_frame)
                if key in best_scores:
                    if probs[pred_frame] > best_scores[key][0]:
                        best_scores[key] = (probs[pred_frame], action)
                else:
                    best_scores[key] = (probs[pred_frame], action)

        for key, probs in probs_dict.items():
            video_id, agent, target, action = key

            video = video_id_to_video[video_id]
            lab_name = video.lab_name
            action_name = ACTIONS.decode(action)
            threshold = thresholds.get((lab_name, action_name), 30) / 100

            pred_frames = np.where(probs > threshold)[0]
            for pred_frame in pred_frames:
                key = (video_id, agent, target, pred_frame)
                if key in best_scores and best_scores[key][1] == action:
                    agent_id = f"mouse{MOUSE_IDS.decode(agent)}"
                    target_id = f"mouse{MOUSE_IDS.decode(target)}"
                    if target_id == agent_id:
                        target_id = "self"

                    submission_df["lab"].append(video.lab_name)
                    submission_df["video_id"].append(video_id)
                    submission_df["agent_id"].append(agent_id)
                    submission_df["target_id"].append(target_id)
                    submission_df["action"].append(ACTIONS.decode(action))
                    submission_df["start_frame"].append(pred_frame)
                    submission_df["stop_frame"].append(pred_frame + 1)

        submission_df = pd.DataFrame(submission_df)
        submission_df["row_id"] = np.arange(len(submission_df))

        submission_cols = "row_id,video_id,agent_id,target_id,action,start_frame,stop_frame"
        submission_cols = submission_cols.split(",")
        submission_df = submission_df[submission_cols]
        return submission_df
