import ctypes
import functools
import json
import mmap
import multiprocessing
import multiprocessing.sharedctypes
import os
import pickle
import secrets
import time
import types
from collections import defaultdict
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from . import paths


# The label vocabularies live in schema.py, which is jax-free so the PyTorch backend can
# import them too. Re-exported here (same objects, not copies) so every existing
# `solution.ACTIONS` / `solution.LABS` reference keeps working and there is one vocabulary
# per process -- important because SNIFFALL mutates ACTIONS at import time.
from .schema import (  # noqa: F401
    ACTIONS,
    BODYPARTS,
    Enum,
    LABS,
    MOUSE_IDS,
    SELF_DIRECTED,
    SNIFF_FAMILY,
    SNIFFALL_LABS,
    TRAIN_ONLY_LABS,
    get_configs,
    parse_mouse_id,
)


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

        fd = os.open(self.path, os.O_RDONLY)
        buffer = os.pread(fd, read_size, read_offset)
        os.close(fd)

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

        fd = os.open(self.path, os.O_RDONLY)
        buffer = os.pread(fd, read_size, read_offset)
        os.close(fd)

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
                        target = rng.choice([i for i in mice if i != agent])

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


@dataclass
class Variable:
    value: None
    trainable: bool

    def set(self, x):
        self.value = x
        return self


def is_variable(x):
    return isinstance(x, Variable)


jax.tree_util.register_dataclass(Variable, data_fields=["value"], meta_fields=["trainable"])


class Layer:
    def create_variables(self, key):
        raise NotImplementedError

    def get_weights(self, variables):
        raise NotImplementedError

    def set_context(self, context):
        self.context = context

    def set_variables(self, variables):
        return ParameterizedLayer(self, variables)


class Linear(Layer):
    def __init__(
        self,
        input_dim,
        output_dim,
        dtype,
        use_bias=True,
        batch_dims=[],
        init_bias=0.0,
        normalize_input=True,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_dims = batch_dims
        self.dtype = dtype
        self.use_bias = use_bias
        self.init_bias = init_bias
        self.normalize_input = normalize_input

    def create_variables(self, key):
        dim = max(self.input_dim, self.output_dim)
        default_lrmul = np.sqrt(dim) / np.sqrt(64)

        variables = {}
        w_key, b_key = jax.random.split(key)
        w_shape = self.batch_dims + [self.input_dim, self.output_dim]
        w = jax.random.normal(w_key, shape=w_shape, dtype="float32")
        w = w / default_lrmul
        variables["w"] = Variable(w, trainable=True)

        s_shape = self.batch_dims + [self.output_dim]
        s = jnp.ones(shape=s_shape, dtype="float32")
        variables["s"] = Variable(s, trainable=True)

        if self.use_bias:
            b_shape = self.batch_dims + [self.output_dim]
            init_bias = jnp.array(self.init_bias, dtype="float32")
            init_bias = jnp.broadcast_to(init_bias, b_shape)
            variables["b"] = Variable(init_bias, trainable=True)

        if self.normalize_input:
            input_shape = self.batch_dims + [self.input_dim]
            variables["m1"] = jnp.zeros(input_shape, dtype="float32")
            variables["m2"] = jnp.zeros(input_shape, dtype="float32")
            variables["n"] = jnp.zeros([], dtype="float32")
            variables["mean"] = jnp.zeros(input_shape, dtype="float32")
            variables["std"] = jnp.ones(input_shape, dtype="float32")

            variables["m1"] = Variable(variables["m1"], trainable=False)
            variables["m2"] = Variable(variables["m2"], trainable=False)
            variables["n"] = Variable(variables["n"], trainable=False)
            variables["mean"] = Variable(variables["mean"], trainable=False)
            variables["std"] = Variable(variables["std"], trainable=False)

        return variables

    def get_weights(self, variables):
        weights = {}
        norm = jnp.linalg.norm(variables["w"], axis=-2, keepdims=True)
        w = variables["w"] / norm
        w = w * variables["s"][..., None, :]

        weights["w"] = w.astype(self.dtype)
        if self.use_bias:
            weights["b"] = variables["b"].astype(self.dtype)

        if self.normalize_input:
            weights["m1"] = variables["m1"]
            weights["m2"] = variables["m2"]
            weights["n"] = variables["n"]
            weights["mean"] = variables["mean"]
            weights["std"] = variables["std"]

        return weights

    def apply(self, weights, x, mask=None):
        updates = {}
        if self.normalize_input:
            if self.context["stage"] == "init":
                decay = self.context["init_decay"]
                leading_dims = list(range(len(x.shape) - len(self.batch_dims) - 1))

                m1 = jnp.mean(x, leading_dims)
                m2 = jnp.mean(x**2, leading_dims)
                weights["m1"] = weights["m1"] + (m1 - weights["m1"]) * (1 - decay)
                weights["m2"] = weights["m2"] + (m2 - weights["m2"]) * (1 - decay)
                weights["n"] = weights["n"] + 1

                norm = 1.0 / (1 - decay ** weights["n"])
                m1 = weights["m1"] * norm
                m2 = weights["m2"] * norm
                input_std = jnp.sqrt(m2 - m1**2)

                weights["std"] = input_std + 1e-4
                weights["mean"] = m1

                updates["m1"] = weights["m1"]
                updates["m2"] = weights["m2"]
                updates["n"] = weights["n"]
                updates["std"] = weights["std"]
                updates["mean"] = weights["mean"]

            x = (x - weights["mean"]) / weights["std"]
            x = x.astype(self.dtype)

        y = jnp.einsum("...i,...io->...o", x, weights["w"])
        if self.use_bias:
            y += weights["b"]

        return y, updates


class Constant(Layer):
    def __init__(self, shape, dtype, batch_dims=[]):
        self.shape = shape
        self.dtype = dtype
        self.batch_dims = batch_dims

    def create_variables(self, key):
        shape = self.batch_dims + list(self.shape)
        c = jax.random.normal(key, shape=shape, dtype="float32")
        return {"c": Variable(c, trainable=True)}

    def get_weights(self, variables):
        return {"c": variables["c"].astype(self.dtype)}

    def apply(self, weights):
        updates = {}
        return weights["c"], updates


class Embedding(Layer):
    def __init__(self, input_dim, cardinality, dtype):
        self.input_dim = input_dim
        self.cardinality = cardinality
        self.dtype = dtype

    def create_variables(self, key):
        w = jax.random.normal(key, shape=[self.cardinality, self.input_dim], dtype="float32")
        return {"w": Variable(w, trainable=True)}

    def get_weights(self, variables):
        return {"w": variables["w"].astype(self.dtype)}

    def apply(self, weights, indices):
        updates = {}
        return weights["w"][indices], updates


class ParameterizedLayer:
    def __init__(self, layer, variables):
        self.layer = layer
        self.variables = variables
        self.weights = layer.get_weights(variables)

    def get_variables(self):
        return self.variables

    def set_variables(self, variables):
        self.variables = variables

    def apply(self, *args, **kwargs):
        outputs, updates = self.layer.apply(self.weights, *args, **kwargs)
        updates_key_vals = dict(jax.tree.flatten_with_path(updates)[0])
        self.variables = jax.tree.map_with_path(lambda path, x: updates_key_vals.get(path, x), self.variables)
        return outputs


class Module:
    def create_variables(self, key):
        variables = {}
        for layer_name, layer in self.layers.items():
            key, subkey = jax.random.split(key)
            variables[layer_name] = layer.create_variables(subkey)
        return variables

    def get_variables(self):
        state = {}
        for layer_name, layer in self.layers.items():
            state[layer_name] = layer.get_variables()
        return state

    def set_variables(self, variables):
        return ParameterizedModule(self, variables)

    def set_context(self, context):
        self.context = context
        for layer_name, layer in self.layers.items():
            layer.set_context(context)


class ParameterizedModule:
    def __init__(self, module, variables):
        self.module = module
        self.variables = variables

        self.layers = {}
        for layer_name, layer in module.layers.items():
            self.layers[layer_name] = layer.set_variables(variables[layer_name])

    def get_variables(self):
        variables = {}
        for layer_name, layer in self.layers.items():
            variables[layer_name] = layer.get_variables()
        return variables

    def set_variables(self, variables):
        for layer_name, layer in self.layers.items():
            layer.set_variables(variables[layer_name])

    def parameterized_function(self, name):
        module_fn = getattr(self.module, name)

        def function(*args, **kwargs):
            outputs = module_fn(self.layers, *args, **kwargs)
            return outputs

        return function

    def __getattr__(self, name):
        if hasattr(self.module, name):
            if isinstance(getattr(self.module, name), types.MethodType):
                return self.parameterized_function(name)
            else:
                t = type(getattr(self.module, name))
                assert False, f"only function attributes passed through, received {t}"
        else:
            assert False


class LSTM(Module):
    def __init__(self, input_dim, hidden_dim, dtype, forget_bias=0.0, batch_dims=[]):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_dims = batch_dims
        self.dtype = dtype
        self.forget_bias = forget_bias

        forget_bias = self.forget_bias * np.repeat([0.0, 1.0, 0.0, 0.0], hidden_dim)

        self.layers = {}
        self.layers["is_linear"] = Linear(
            input_dim=self.input_dim,
            output_dim=4 * self.hidden_dim,
            batch_dims=self.batch_dims,
            dtype=self.dtype,
            use_bias=True,
            init_bias=forget_bias,
        )
        self.layers["ss_linear"] = Linear(
            input_dim=self.hidden_dim,
            output_dim=4 * self.hidden_dim,
            batch_dims=self.batch_dims,
            dtype=self.dtype,
            use_bias=False,
            normalize_input=False,
        )
        self.layers["h0"] = Constant(shape=[self.hidden_dim], batch_dims=self.batch_dims, dtype=self.dtype)
        self.layers["c0"] = Constant(shape=[self.hidden_dim], batch_dims=self.batch_dims, dtype=self.dtype)

    def apply(self, layers, x):
        tsteps, bs = x.shape[0], x.shape[1]

        h0 = jnp.tanh(layers["h0"].apply())
        c0 = layers["c0"].apply()

        tiling = [bs] + [1] * (len(self.batch_dims) + 1)
        h0 = jnp.tile(h0[None], tiling)
        c0 = jnp.tile(c0[None], tiling)

        xs = layers["is_linear"].apply(x)
        hs_shape = [tsteps, bs] + self.batch_dims + [self.hidden_dim]
        hs = jnp.empty(hs_shape, dtype=self.dtype)

        def body(t, state):
            x_t = xs[t]
            x_hh = layers["ss_linear"].apply(state["h"])
            gates = x_t + x_hh

            i, f, c, o = jnp.split(gates, 4, axis=-1)
            c = jax.nn.sigmoid(f) * state["c"] + jax.nn.sigmoid(i) * jnp.tanh(c)
            h = jax.nn.sigmoid(o) * jnp.tanh(c)
            hs = state["hs"].at[t].set(h)
            return {"h": h, "c": c, "hs": hs}

        state = {"h": h0, "c": c0, "hs": hs}
        outputs = jax.lax.fori_loop(0, tsteps, body, state)
        return outputs["hs"]


class BidirectionalLSTM(Module):
    def __init__(self, input_dim, hidden_dim, dtype, forget_bias=0.0, batch_dims=[]):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_dims = batch_dims
        self.dtype = dtype
        self.forget_bias = forget_bias

        self.layers = {}
        self.layers["lstm-fw"] = LSTM(input_dim, hidden_dim, dtype, forget_bias, batch_dims)
        self.layers["lstm-bw"] = LSTM(input_dim, hidden_dim, dtype, forget_bias, batch_dims)

    def apply(self, layers, x):
        h_fw = layers["lstm-fw"].apply(x)
        x_bw = jnp.flip(x, axis=0)
        h_bw = layers["lstm-bw"].apply(x_bw)
        h_bw = jnp.flip(h_bw, axis=0)
        h = jnp.concat([h_fw, h_bw], axis=-1)
        return h


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
            pad_elt = jax.tree.map(np.empty_like, batch_elements[-1])
            pad_elt["batch_mask"] = np.int8(0)
            num_pad_elts = n - len(batch_elements)
            for i in range(num_pad_elts):
                batch_elements.append(pad_elt)

        outer_structure = jax.tree.structure(["*"] * len(batch_elements))
        y = jax.tree.transpose(outer_structure, None, batch_elements)
        batch = jax.tree.map(lambda x, y: np.stack(y), batch_elements[-1], y)
        yield batch


def to_device(stream, sharding):
    for item in stream:
        if sharding is not None:
            yield jax.make_array_from_process_local_data(sharding, item)
        else:
            yield jax.tree.map(jnp.array, item)


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
        yield jax.tree.map(lambda x: np.array(x), item)


def get_batch_size(item):
    def reduce_fn(batch_size, leaf):
        if batch_size is None:
            return leaf.shape[0]
        assert batch_size == leaf.shape[0]
        return batch_size

    return jax.tree.reduce(reduce_fn, item, initializer=None)


def unbatch(stream):
    for batch in stream:
        batch_size = get_batch_size(batch)
        for i in range(batch_size):
            element = jax.tree.map(lambda x: x[i].copy(), batch)
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


class Adam:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def create_variables(self, params):
        def init_moments(param):
            return {
                "m1": jnp.zeros(param.shape, dtype=param.dtype),
                "m2": jnp.zeros(param.shape, dtype=param.dtype),
            }

        moments = jax.tree.map(init_moments, params, is_leaf=is_variable)
        return {"t": jnp.ones([]), "moments": moments}

    def update(self, params, opt_state, grads):
        b1_correction = 1.0 / (1 - self.beta1 ** opt_state["t"])
        b2_correction = 1.0 / (1 - self.beta2 ** opt_state["t"])

        def update_moments(grad, moments):
            moments["m1"] += (grad - moments["m1"]) * (1 - self.beta1)
            moments["m2"] += (grad**2 - moments["m2"]) * (1 - self.beta2)
            return moments

        def update_param(param, moments):
            m1 = moments["m1"] * b1_correction
            m2 = moments["m2"] * b2_correction
            param -= self.learning_rate * (m1 / (jnp.sqrt(m2) + self.eps))
            return param

        moments = jax.tree.map(update_moments, grads, opt_state["moments"])
        params = jax.tree.map(update_param, params, moments)
        opt_state = {"t": opt_state["t"] + 1, "moments": moments}
        return params, opt_state


class EMA:
    def __init__(self, decay):
        self.decay = decay

    def create_variables(self, params):
        return {
            "t": jnp.ones([], dtype="float32"),
            "sums": jax.tree.map(lambda x: x * (1.0 - self.decay), params),
        }

    def update(self, ema_state, variables):
        def update_sums(variable, sum):
            return sum + (variable.value - sum) * (1 - self.decay)

        return {
            "t": ema_state["t"] + 1,
            "sums": jax.tree.map(update_sums, variables, ema_state["sums"], is_leaf=is_variable),
        }

    def values(self, ema_state):
        norm = 1 - self.decay ** ema_state["t"]
        return jax.tree.map(lambda x: x / norm, ema_state["sums"])


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir,
        max_checkpoints,
        metric_name,
        lower_is_better=True,
        patience=None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.lower_is_better = lower_is_better
        self.patience = patience
        self.checkpoints = [(None, float("-inf"))] * self.max_checkpoints

    def write_file(self, params, path):
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        cpu = jax.devices("cpu")[0]
        params = jax.tree.map(lambda x: jax.device_put(x, cpu), params)
        with open(path, "wb") as f:
            pickle.dump(params, f)

    def path(self, step):
        return f"{self.checkpoint_dir}/{step}.pkl"

    def update(self, params, step, metrics):
        score = metrics[self.metric_name]
        if np.isnan(score):
            # todo: nan breaks things
            assert False
        if self.lower_is_better:
            score = -score

        if score > self.checkpoints[0][1]:
            del_path = self.path(self.checkpoints[0][0])
            if os.path.isfile(del_path):
                os.remove(del_path)

            self.checkpoints[0] = (step, score)
            self.checkpoints = sorted(self.checkpoints, key=lambda x: x[1])
            self.write_file(params, self.path(step))

            msg = f"writing {self.path(step)}"
            if step == self.checkpoints[-1][0]:
                if os.path.islink(self.path("best")):
                    os.unlink(self.path("best"))
                os.symlink(os.path.abspath(self.path(step)), self.path("best"))
                msg += " *"
            print(msg)

        terminate_training = False
        if self.patience is not None:
            if step - self.checkpoints[-1][0] > self.patience:
                terminate_training = True

        return terminate_training


class Trainer:
    def __init__(
        self,
        experiment_name,
        model,
        optimizer,
        train_dataset,
        train_batch_size,
        eval_dataset=None,
        eval_batch_size=None,
        custom_eval_loop=None,
        eval_interval=2000,
        skip_eval=False,
        seed=None,
        train_log_interval=500,
        max_training_steps=10**10,
        ema_decay=0.9993,
        early_stopping_config={},
    ):
        self.experiment_name = experiment_name
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size or 4 * train_batch_size
        self.custom_eval_loop = custom_eval_loop
        self.eval_interval = eval_interval
        self.skip_eval = skip_eval
        self.seed = seed
        self.train_log_interval = train_log_interval
        self.max_training_steps = max_training_steps
        self.ema_decay = ema_decay

        self.experiment_dir = f"experiments/{self.experiment_name}"
        self.checkpoint_dir = f"{self.experiment_dir}/checkpoints"
        self.checkpoint_manager = CheckpointManager(
            self.checkpoint_dir,
            max_checkpoints=3,
            metric_name=early_stopping_config.get("metric_name", "obj"),
            lower_is_better=early_stopping_config.get("lower_is_better", True),
            patience=early_stopping_config.get("patience", None),
        )

        self.ema = EMA(self.ema_decay)

        rng = np.random.default_rng(self.seed)
        seeds = rng.integers(low=0, high=2**63, size=[3], dtype="int64")
        self.model_seed, self.train_seed, self.eval_seed = seeds

        n_devices = jax.local_device_count()
        mesh = jax.make_mesh((n_devices,), ("batch",))
        jax.set_mesh(mesh)
        self.sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))

    def get_ddi_loop(self):
        @functools.partial(jax.jit, donate_argnums=[0])
        def ddi_step(state, batch):
            context = {"stage": "init", "init_decay": 0.99}
            self.model.set_context(context)
            key, subkey = jax.random.split(state["key"])

            variable_values = jax.tree.map(
                lambda x: x.value,
                state["variables"],
                is_leaf=is_variable,
            )
            pmodel = self.model.set_variables(variable_values)
            pmodel.compute_loss(batch, subkey)
            updated_variable_values = pmodel.get_variables()
            variables = jax.tree.map(lambda x, y: y.set(x), updated_variable_values, state["variables"])
            next_state = {"variables": variables, "key": key}
            return next_state

        def ddi_loop(ddi_state, n):
            def batch_update(batch):
                nonlocal ddi_state
                ddi_state = ddi_step(ddi_state, batch)
                return ddi_state

            elements = Pipeline(self.train_dataset.element_iterator())
            batches = elements.batch(self.train_batch_size).to_device(self.sharding)
            ddi_state = batches.map(batch_update).take(n).last()
            return ddi_state

        return ddi_loop

    def initial_state(self):
        key = jax.random.key(self.model_seed)

        init_key, ddi_key = jax.random.split(key, 2)
        variables = self.model.create_variables(init_key)

        ddi_loop = self.get_ddi_loop()
        ddi_state = {"variables": variables, "key": ddi_key}
        ddi_state = ddi_loop(ddi_state, 256)
        variables = ddi_state["variables"]

        variables_flat, variables_treedef = jax.tree.flatten(variables, is_leaf=is_variable)

        trainable_indices = []
        static_indices = []
        trainable_values = []
        static_values = []
        for i, variable in enumerate(variables_flat):
            if variable.trainable:
                trainable_values.append(variable.value)
                trainable_indices.append(i)
            else:
                static_values.append(variable.value)
                static_indices.append(i)

        optimizer_state = self.optimizer.create_variables(trainable_values)

        variable_values = jax.tree.map(lambda x: x.value, variables, is_leaf=is_variable)
        ema_state = self.ema.create_variables(variable_values)

        train_state = {
            "variables": variables,
            "optimizer": optimizer_state,
            "ema": ema_state,
            "key": jax.random.key(self.train_seed),
        }
        return train_state

    def get_eval_loop(self):
        @jax.jit
        def eval_step(variables, batch, key):
            context = {"stage": "eval"}
            self.model.set_context(context)
            key, subkey = jax.random.split(key)
            pmodel = self.model.set_variables(variables)
            loss, metrics = pmodel.compute_loss(batch, subkey)
            return metrics, key

        def eval_loop(train_state):
            def get_batch_metrics(batch):
                nonlocal key
                metrics, key = eval_step(train_state, batch, key)
                return metrics

            key = jax.random.key(self.eval_seed)
            elements = Pipeline(self.eval_dataset.element_iterator())
            batches = elements.batch(self.eval_batch_size).to_device(self.sharding)
            metrics = batches.map(get_batch_metrics).to_host().average_metrics()
            return metrics

        return eval_loop

    def train_step(self, state, batch):
        context = {"stage": "train"}
        self.model.set_context(context)
        key, subkey = jax.random.split(state["key"])

        variables = state["variables"]
        variables_flat, variables_treedef = jax.tree.flatten(variables, is_leaf=is_variable)

        trainable_indices = []
        static_indices = []
        trainable_values = []
        static_values = []
        for i, variable in enumerate(variables_flat):
            if variable.trainable:
                trainable_values.append(variable.value)
                trainable_indices.append(i)
            else:
                static_values.append(variable.value)
                static_indices.append(i)

        def compute_loss(trainable_values, static_values, batch, key):
            merged_vars_flat = [None] * (len(variables_flat))
            for i, var in zip(trainable_indices, trainable_values):
                merged_vars_flat[i] = var
            for i, var in zip(static_indices, static_values):
                merged_vars_flat[i] = var
            merged_vars = jax.tree.unflatten(variables_treedef, merged_vars_flat)
            pmodel = self.model.set_variables(merged_vars)
            loss, metrics = pmodel.compute_loss(batch, key)
            updated_variables = pmodel.get_variables()
            return loss, (metrics, updated_variables)

        key, subkey = jax.random.split(key)
        grad_fn = jax.value_and_grad(compute_loss, argnums=0, has_aux=True)
        (loss, aux), grads = grad_fn(trainable_values, static_values, batch, key)
        metrics, updated_variables = aux
        updated_variables_flat, _ = jax.tree.flatten(updated_variables)

        grad_updated_values, next_opt_state = self.optimizer.update(trainable_values, state["optimizer"], grads)
        for i, value in zip(trainable_indices, grad_updated_values):
            updated_variables_flat[i] = value

        updated_variables_values = jax.tree.unflatten(variables_treedef, updated_variables_flat)
        updated_variables = jax.tree.map(lambda x, y: y.set(x), updated_variables_values, variables)
        next_ema_state = self.ema.update(state["ema"], variables)

        next_state = {"variables": updated_variables, "optimizer": next_opt_state, "ema": next_ema_state, "key": key}
        return next_state, metrics

    def get_train_loop(self):
        train_step_fn = jax.jit(self.train_step, donate_argnames=["state"])
        elements = Pipeline(self.train_dataset.element_iterator())
        batches = elements.batch(self.train_batch_size).to_device(self.sharding)

        def train_loop(train_state, n):
            def get_batch_metrics(batch):
                nonlocal train_state
                train_state, metrics = train_step_fn(train_state, batch)
                return metrics

            metrics = batches.take(n).map(get_batch_metrics).to_host()
            metrics = metrics.average_metrics()
            return train_state, metrics

        return train_loop

    def log_metrics(self, metrics, dt=None, tag=None):
        strings = []
        if tag is not None:
            strings.append(tag)
        for k, v in metrics.items():
            strings.append(f"{k}: {v:>8.3f}")
        if dt is not None:
            strings.append(f"({dt: >3.1f}s)")
        print("     ".join(strings))

    def train(self, path=None):
        state = self.initial_state()

        eval_loop = None
        if not self.skip_eval:
            if self.custom_eval_loop is not None:
                eval_loop = self.custom_eval_loop
            elif self.eval_dataset is not None:
                eval_loop = self.get_eval_loop()

        train_loop = self.get_train_loop()
        get_ema_values = jax.jit(self.ema.values)

        t_prev = time.time()
        step = 0
        while step < self.max_training_steps:
            if eval_loop is not None and step % self.eval_interval == 0:
                if eval_loop is not None:
                    eval_t0 = time.time()
                    ema_values = get_ema_values(state["ema"])
                    metrics = eval_loop(ema_values)
                    eval_dt = time.time() - eval_t0
                    self.log_metrics(metrics, dt=eval_dt, tag="[EVAL]")
                    terminate_training = self.checkpoint_manager.update(ema_values, step, metrics)
                    t_prev = time.time()
                    if terminate_training:
                        print("training finished")
                        return

            state, train_metrics = train_loop(state, self.train_log_interval)
            step += self.train_log_interval

            dt = time.time() - t_prev
            self.log_metrics(train_metrics, dt=dt, tag=f"[step {step:>8}]")
            t_prev = time.time()
        print("training finished")


class UnsupervisedModel(Module):
    def __init__(
        self,
        d_res,
        d_lstm,
        d_ff,
        d_edge,
        n_layers,
        n_bp,
        sample_rate,
        aggregation_radius,
        dtype,
    ):
        self.d_res = d_res
        self.d_lstm = d_lstm
        self.d_ff = d_ff
        self.d_edge = d_edge

        self.n_layers = n_layers
        self.dtype = dtype
        self.n_bp = n_bp
        self.output_bins = 16
        self.sample_rate = sample_rate

        self.max_y_norm_30 = 8
        self.norm_rescale = np.sqrt(30 / self.sample_rate)
        self.max_y_norm = self.norm_rescale * self.max_y_norm_30
        self.aggregation_radius = aggregation_radius

        self.lags = [1, 2, 3, 4]
        max_norms_30 = {1: 19, 2: 34, 3: 46, 4: 57}
        self.lag_max_norms = {k: self.norm_rescale * v for k, v in max_norms_30.items()}

        self.layers = {}
        self.layers["x-emb"] = Constant([n_bp, d_res], dtype)
        for lag in self.lags:
            self.layers[f"nan-emb-{lag}"] = Constant([d_res], dtype)
            self.layers[f"dt-proj-{lag}"] = Linear(2, d_res, dtype, batch_dims=[n_bp])

        self.layers["aug-proj"] = Linear(6, d_res, dtype)
        for l in range(self.n_layers):
            kwargs = dict(dtype=dtype, batch_dims=[n_bp, n_bp])
            self.layers[f"self-in-{l}"] = Linear(2 * d_res, d_edge, **kwargs)
            self.layers[f"self-dx-{l}"] = Linear(2, d_edge, **kwargs)
            self.layers[f"self-out-{l}"] = Linear(d_edge, d_res, **kwargs)

            self.layers[f"cross-in-{l}"] = Linear(2 * d_res, d_edge, **kwargs)
            self.layers[f"cross-dx-{l}"] = Linear(2, d_edge, **kwargs)
            self.layers[f"cross-out-{l}"] = Linear(d_edge, d_res, **kwargs)

            self.layers[f"merge-{l}"] = Linear(2 * d_res, d_res, dtype, batch_dims=[n_bp])

            self.layers[f"ff-in-{l}"] = Linear(d_res, d_ff, dtype, batch_dims=[n_bp])
            self.layers[f"ff-out-{l}"] = Linear(d_ff, d_res, dtype, batch_dims=[n_bp])

            self.layers[f"lstm-{l}"] = LSTM(d_res, d_lstm, dtype, batch_dims=[n_bp])
            self.layers[f"lstm-res-{l}"] = Linear(d_lstm, d_res, dtype, batch_dims=[n_bp])

        self.layers["out-proj"] = Linear(d_res, self.output_bins**2, dtype, batch_dims=[n_bp])

    def get_labels(self, x, batch):
        x_tp1 = jnp.pad(x[1:], [(0, 1), (0, 0), (0, 0), (0, 0)], constant_values=np.nan)
        y = x_tp1 - x

        y_mask = jnp.all(~jnp.isnan(y), axis=-1)
        y = jnp.where(y_mask[..., None], y, 1.0)

        y_r = jnp.linalg.norm(y, axis=-1)
        y_r = jnp.minimum(y_r, self.max_y_norm)
        y_r = (y_r / self.max_y_norm) * (self.output_bins - 1)
        y_r = jnp.round(y_r).astype("int32")

        y_theta = jnp.arctan2(y[..., 1], y[..., 0])
        y_theta = (y_theta + np.pi) / (2 * np.pi)
        y_theta = y_theta * (self.output_bins - 1)
        y_theta = jnp.round(y_theta).astype("int32")

        y = y_r * self.output_bins + y_theta

        batch_mask = jnp.concat([batch["batch_mask"]] * 2, axis=0)
        y_mask = y_mask & ((batch_mask == 1)[None, :, None])
        return y, y_mask

    def forward(self, layers, x, batch, key, extract_features=False):
        tsteps, bs, num_bodyparts, channels = x.shape
        x_t = x

        x_emb = layers["x-emb"].apply()
        x = x_emb[None, None]
        for lag in self.lags:
            x_lag = jnp.pad(x_t[:-lag], [(lag, 0), (0, 0), (0, 0), (0, 0)], constant_values=np.nan)
            dt = x_t - x_lag
            dt_mask = ~jnp.any(jnp.isnan(dt), axis=-1, keepdims=True)
            max_norm = self.lag_max_norms[lag]
            norms = jnp.linalg.norm(dt, axis=-1, keepdims=True)
            dt = jnp.where(norms > max_norm, max_norm * (dt / norms), dt)
            dt = jnp.where(dt_mask, dt, 0.0)
            dt = dt.astype(self.dtype)

            dt_proj = layers[f"dt-proj-{lag}"].apply(dt)
            nan_emb = layers[f"nan-emb-{lag}"].apply()
            dt_proj = jnp.where(dt_mask, dt_proj, nan_emb[None, None, None])

            x += dt_proj

        x = x / ((len(self.lags) + 1) ** 0.5)

        ap = jnp.concat([batch["augmentation_params"], batch["augmentation_params"]], axis=0)
        ap = ap.astype(self.dtype)
        b = layers["aug-proj"].apply(ap)[None, :, None]
        x = (x + b) * (0.5**0.5)

        x = jax.nn.silu(x)

        self_dx = x_t[:, :, :, None] - x_t[:, :, None]
        self_norms = jnp.linalg.norm(self_dx, axis=-1, keepdims=True)
        self_adj = self_norms < self.aggregation_radius
        self_dx = (self_dx / self_norms) * jnp.sqrt(self_norms)
        self_dx = jnp.where(jnp.isnan(self_dx), 0.0, self_dx)
        self_dx, self_adj = self_dx.astype(self.dtype), self_adj.astype(self.dtype)

        x_cross = jnp.concat(jnp.split(x_t, 2, axis=1)[::-1], axis=1)
        cross_dx = x_cross[:, :, :, None] - x_t[:, :, None]
        cross_norms = jnp.linalg.norm(cross_dx, axis=-1, keepdims=True)
        cross_adj = cross_norms < self.aggregation_radius
        cross_dx = (cross_dx / cross_norms) * jnp.sqrt(cross_norms)
        cross_dx = jnp.where(jnp.isnan(cross_dx), 0.0, cross_dx)
        cross_dx, cross_adj = cross_dx.astype(self.dtype), cross_adj.astype(self.dtype)

        feats = []
        for l in range(self.n_layers):
            # self
            src_feats = jnp.tile(x[:, :, :, None], (1, 1, 1, self.n_bp, 1))
            dst_feats = jnp.tile(x[:, :, None], (1, 1, self.n_bp, 1, 1))
            self_feats = jnp.concat([src_feats, dst_feats], axis=-1)
            self_feats = layers[f"self-in-{l}"].apply(self_feats)
            self_feats = self_feats + layers[f"self-dx-{l}"].apply(self_dx)
            self_feats = jax.nn.silu(self_feats)
            self_feats = layers[f"self-out-{l}"].apply(self_feats)
            self_feats = (self_feats * self_adj).sum(axis=-2) / 2.0

            # cross
            x_cross = jnp.concat(jnp.split(x, 2, axis=1)[::-1], axis=1)
            cross_src = jnp.tile(x[:, :, :, None], (1, 1, 1, self.n_bp, 1))
            cross_dst = jnp.tile(x_cross[:, :, None], (1, 1, self.n_bp, 1, 1))
            cross_feats = jnp.concat([cross_src, cross_dst], axis=-1)
            cross_feats = layers[f"cross-in-{l}"].apply(cross_feats)
            cross_feats = cross_feats + layers[f"cross-dx-{l}"].apply(cross_dx)
            cross_feats = jax.nn.silu(cross_feats)
            cross_feats = layers[f"cross-out-{l}"].apply(cross_feats)
            cross_feats = (cross_feats * cross_adj).sum(axis=-2) / 2.0

            # merge
            y = jnp.concat([self_feats, cross_feats], axis=-1)
            y = layers[f"merge-{l}"].apply(y)
            x += y

            # ff
            y = layers[f"ff-in-{l}"].apply(x)
            y = jax.nn.silu(y)
            y = layers[f"ff-out-{l}"].apply(y)
            x += y

            # lstm
            y = layers[f"lstm-{l}"].apply(x)
            y = layers[f"lstm-res-{l}"].apply(y)
            x += y

            feats.append(x)

        if extract_features:
            return feats

        logits = layers["out-proj"].apply(x)
        return logits

    def compute_loss(self, layers, batch, key):
        x = jnp.concat([batch["agent"], batch["target"]], axis=0)
        x = x.transpose(1, 0, 2, 3)

        logits = self.forward(layers, x, batch, key).astype("float32")
        y, y_mask = self.get_labels(x, batch)

        labels = jax.nn.one_hot(y, self.output_bins**2)
        logprobs = (labels * jax.nn.log_softmax(logits)).sum(axis=-1)
        y_mask = y_mask.astype(logprobs.dtype)
        nll_weight = y_mask.sum()
        nll = -(logprobs * y_mask).sum() / nll_weight

        metrics = {"nll": (nll, nll_weight)}
        return (nll, metrics)

    def extract_features(self, layers, batch, key):
        x = jnp.concat([batch["agent"], batch["target"]], axis=0)
        x = x.transpose(1, 0, 2, 3)
        feats = self.forward(layers, x, batch, key, extract_features=True)
        return feats


def pretrain(config):
    videos = load_videos(mode="train", use_cached=True)
    train_videos, val_videos = split_videos(videos, validation_frac=0.15, random_seed=config["split_seed"])

    train_dataset = Dataset(
        videos=train_videos,
        seq_len=64,
        sample_rate=config["sample_rate"],
        padding=32,
        num_bodyparts=config["num_bodyparts"],
        num_epochs=100000,
        unsupervised=True,
        max_scale=config["max_scale"],
        max_time_dilation=config["max_time_dilation"],
        rotate=True,
        flip=True,
        noise_scale=config["noise_scale"],
        num_workers=4,
        seed=[0] + config["pretrain_seed"],
    )
    val_dataset = Dataset(
        videos=val_videos,
        seq_len=64,
        sample_rate=config["sample_rate"],
        padding=32,
        num_bodyparts=config["num_bodyparts"],
        num_epochs=1,
        unsupervised=True,
        max_scale=1,
        max_time_dilation=1,
        rotate=False,
        flip=False,
        noise_scale=config["noise_scale"],
        num_workers=8,
        seed=[1] + config["pretrain_seed"],
    )
    model = UnsupervisedModel(
        d_res=192,
        d_lstm=192,
        d_ff=192 * 2,
        d_edge=96,
        n_layers=4,
        n_bp=config["num_bodyparts"],
        sample_rate=config["sample_rate"],
        aggregation_radius=config["aggregation_radius"],
        dtype="bfloat16",
    )
    trainer = Trainer(
        experiment_name=f"{config['name']}/pretrain",
        model=model,
        optimizer=Adam(0.02),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        train_batch_size=128,
        seed=[2] + config["pretrain_seed"],
        early_stopping_config={
            "metric_name": "nll",
            "lower_is_better": True,
            "patience": 15000,
        },
        max_training_steps=125000,
    )
    trainer.train()

    os.makedirs(persist_dir, exist_ok=True)
    with open(f"{trainer.checkpoint_dir}/best.pkl", "rb") as f1:
        with open(f"{persist_dir}/{config['name']}_unsupervised.pkl", "wb") as f2:
            f2.write(f1.read())


class SupervisedModel(Module):
    def __init__(self, d_res, d_ff, d_lstm, n_layers, n_bp, padding, dtype, unsupervised_model):
        self.d_res = d_res
        self.d_ff = d_ff
        self.d_lstm = d_lstm
        self.n_layers = n_layers
        self.dtype = dtype
        self.n_bp = n_bp
        self.padding = padding

        unsupervised_model, unsupervised_path = unsupervised_model
        unsupervised_params = pickle.load(open(unsupervised_path, "rb"))
        feat_dim = unsupervised_model.n_layers * unsupervised_model.d_res
        node_dim = unsupervised_model.d_res
        unsupervised_model.set_context({"stage": "eval"})
        self.unsupervised_model = unsupervised_model.set_variables(unsupervised_params)

        self.layers = {}
        self.layers["ff-merge-in"] = Linear(2 * feat_dim, node_dim, dtype, batch_dims=[n_bp])
        self.layers["ff-merge-out"] = Linear(node_dim, node_dim, dtype, batch_dims=[n_bp])
        self.layers["feat-flat-proj"] = Linear(node_dim, d_res, dtype)

        self.layers["lab-embedding"] = Embedding(d_res, len(LABS), dtype)
        for l in range(self.n_layers):
            self.layers[f"lstm-{l}"] = BidirectionalLSTM(d_res, d_lstm, dtype)
            self.layers[f"out-proj-{l}"] = Linear(2 * d_lstm, d_res, dtype)

            self.layers[f"ff-{l}-in"] = Linear(d_res, d_ff, dtype)
            self.layers[f"ff-{l}-out"] = Linear(d_ff, d_res, dtype)

        self.layers["out-proj"] = Linear(d_res, len(ACTIONS), dtype)

    def compute_logits(self, layers, batch, key):
        xs = self.unsupervised_model.extract_features(batch, key)
        x = jnp.concat(xs, axis=-1)
        x_agent, x_target = jnp.split(x, 2, axis=1)
        x = jnp.concat([x_agent, x_target], axis=-1)

        x = layers["ff-merge-in"].apply(x)
        x = jax.nn.silu(x)
        x = layers["ff-merge-out"].apply(x)
        x = jnp.sum(x, axis=2)
        x = layers["feat-flat-proj"].apply(x)

        x += 0.1 * layers["lab-embedding"].apply(batch["lab_id"])
        for l in range(self.n_layers):
            y = layers[f"lstm-{l}"].apply(x)
            y = layers[f"out-proj-{l}"].apply(y)
            x += y

            y = layers[f"ff-{l}-in"].apply(x)
            y = jax.nn.silu(y)
            y = layers[f"ff-{l}-out"].apply(y)
            x += y

        logits = layers["out-proj"].apply(x)
        logits = jnp.transpose(logits, (1, 0, 2))

        logits = logits[:, self.padding : logits.shape[1] - self.padding]
        logits = logits.astype("float32")
        return logits

    def get_labels(self, batch):
        self_labels = jax.nn.one_hot(batch["self_labels"], len(ACTIONS))
        cross_labels = jax.nn.one_hot(batch["cross_labels"], len(ACTIONS))
        labels = self_labels + cross_labels
        label_mask = batch["self_label_mask"] + batch["cross_label_mask"]
        mask = (label_mask == 1) & ((batch["batch_mask"] == 1)[:, None])
        mask = mask.astype("float32")
        return labels, mask

    def compute_loss(self, layers, batch, key):
        logits = self.compute_logits(layers, batch, key)
        labels, mask = self.get_labels(batch)

        log_prob = jnp.where(labels == 1, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits))
        log_prob = (log_prob * mask[:, None]).sum(axis=-1)
        nll_weight = mask.sum()
        nll = -log_prob.mean(axis=1).sum() / nll_weight
        metrics = {"nll": (nll, nll_weight)}
        return (nll, metrics)

    def predict(self, layers, batch, key):
        logits = self.compute_logits(layers, batch, key)
        probs = jax.nn.sigmoid(logits)
        return probs


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


def get_custom_eval_loop(trainer):
    @jax.jit
    def eval_step(variables, batch, key):
        key, subkey = jax.random.split(key)
        pmodel = trainer.model.set_variables(variables)
        probs = pmodel.predict(batch, subkey)
        return probs, key

    def custom_eval_loop(variables):
        def get_batch_outputs(batch):
            nonlocal key
            probs, key = eval_step(variables, batch, key)
            return batch, probs

        key = jax.random.key(trainer.eval_seed)
        predictions = Predictions(trainer.eval_dataset.videos)

        elements = Pipeline(trainer.eval_dataset.element_iterator())
        batches = elements.batch(trainer.eval_batch_size).to_device(trainer.sharding)
        batch_outputs = batches.map(get_batch_outputs).to_host()
        batch_outputs.unbatch().map(lambda x: predictions.update(*x)).last()
        metrics, _ = predictions.score()
        return metrics

    return custom_eval_loop


def train(config):
    videos = load_videos(mode="train", use_cached=True)
    train_videos, val_videos = split_videos(videos, validation_frac=0.15, random_seed=config["split_seed"])
    val_videos = [v for v in val_videos if v.lab_name not in TRAIN_ONLY_LABS]

    train_dataset = Dataset(
        videos=train_videos,
        seq_len=64,
        sample_rate=config["sample_rate"],
        padding=32,
        num_bodyparts=config["num_bodyparts"],
        num_epochs=100000,
        unsupervised=False,
        max_scale=config["max_scale"],
        max_time_dilation=config["max_time_dilation"],
        rotate=True,
        flip=True,
        noise_scale=config["noise_scale"],
        num_workers=4,
        seed=[0] + config["train_seed"],
    )
    val_dataset = Dataset(
        videos=val_videos,
        seq_len=64,
        sample_rate=config["sample_rate"],
        padding=32,
        num_bodyparts=config["num_bodyparts"],
        num_epochs=1,
        unsupervised=False,
        max_scale=1,
        max_time_dilation=1,
        rotate=False,
        flip=False,
        noise_scale=config["noise_scale"],
        num_workers=8,
        seed=[1] + config["train_seed"],
    )
    unsupervised_model = UnsupervisedModel(
        d_res=192,
        d_lstm=192,
        d_ff=192 * 2,
        d_edge=96,
        n_layers=4,
        n_bp=config["num_bodyparts"],
        sample_rate=config["sample_rate"],
        aggregation_radius=config["aggregation_radius"],
        dtype="bfloat16",
    )
    unsupervised_path = f"{persist_dir}/{config['name']}_unsupervised.pkl"
    supervised_model = SupervisedModel(
        d_res=256,
        d_ff=768,
        d_lstm=256,
        n_layers=3,
        n_bp=config["num_bodyparts"],
        padding=32,
        dtype="bfloat16",
        unsupervised_model=(unsupervised_model, unsupervised_path),
    )
    trainer = Trainer(
        experiment_name=f"{config['name']}/train",
        model=supervised_model,
        optimizer=Adam(0.004),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        train_batch_size=128,
        seed=[2] + config["train_seed"],
        early_stopping_config={
            "metric_name": "f1",
            "lower_is_better": False,
            "patience": 10000,
        },
        max_training_steps=50000,
    )
    trainer.custom_eval_loop = get_custom_eval_loop(trainer)
    trainer.train()

    os.makedirs(persist_dir, exist_ok=True)
    with open(f"{trainer.checkpoint_dir}/best.pkl", "rb") as f1:
        with open(f"{persist_dir}/{config['name']}_supervised.pkl", "wb") as f2:
            f2.write(f1.read())


def compute_ensemble_thresholds(configs):
    def get_action_counts(videos):
        counts = defaultdict(int)
        videos_filtered = [v for v in videos if v.lab_name not in TRAIN_ONLY_LABS]
        for video in videos_filtered:
            annotation_path = f"{dataset_dir}/train_annotation/{video.lab_name}/{video.video_id}.parquet"
            if os.path.isfile(annotation_path):
                df = pd.read_parquet(annotation_path)
                label_counts = df["action"].value_counts()
                for action, v in dict(label_counts).items():
                    if action != "ejaculate":
                        action_id = ACTIONS.encode(action)
                        counts[(video.lab_name, action_id)] += v
        return counts

    rows = []
    for config_name, config in configs.items():
        thresholds = pickle.load(open(f"{persist_dir}/{config_name}_thresholds.pkl", "rb"))

        videos = load_videos(mode="train", use_cached=True)
        train_videos, val_videos = split_videos(videos, validation_frac=0.15, random_seed=config["split_seed"])
        train_counts = get_action_counts(train_videos)
        val_counts = get_action_counts(val_videos)

        all_keys = set(train_counts.keys()) | set(val_counts.keys())
        for lab, action in all_keys:
            weight = min(train_counts[(lab, action)], val_counts[(lab, action)])
            threshold = thresholds[lab].get(action, np.nan)
            row = {"config": config_name, "lab": lab, "action": action, "weight": weight, "threshold": threshold}
            rows.append(row)

    ensemble_thresholds = {}
    df = pd.DataFrame.from_records(rows)
    for (lab_name, action_id), action_df in df.groupby(["lab", "action"]):
        b = action_df[~pd.isnull(action_df["threshold"])]
        if len(b) > 0:
            thresholds = []
            for weight, threshold in zip(b["weight"], b["threshold"]):
                thresholds.extend([threshold] * int(round(weight)))

            prior_mean, prior_var = 30, 15**2
            obs_mean, obs_var = np.mean(thresholds), np.var(thresholds)
            pos_mean = (obs_var / (prior_var + obs_var)) * prior_mean + (prior_var / (prior_var + obs_var)) * obs_mean
            ensemble_thresholds[(lab_name, ACTIONS.decode(action_id))] = int(round(pos_mean))

    os.makedirs(persist_dir, exist_ok=True)
    pickle.dump(ensemble_thresholds, open(f"{persist_dir}/thresholds.pkl", "wb"))


def predict(config, predictions=None, is_val=False):
    if predictions is None:
        is_val = True
        videos = load_videos(mode="train", use_cached=True)
        train_videos, val_videos = split_videos(videos, validation_frac=0.15, random_seed=config["split_seed"])
        val_videos = [v for v in val_videos if v.lab_name not in TRAIN_ONLY_LABS]
        predictions = Predictions(val_videos)

    sharding = None

    dataset = Dataset(
        videos=predictions.videos,
        seq_len=64,
        sample_rate=config["sample_rate"],
        padding=32,
        num_bodyparts=config["num_bodyparts"],
        num_epochs=5,
        unsupervised=False,
        max_scale=0.95 * config["max_scale"],
        max_time_dilation=0.95 * config["max_time_dilation"],
        rotate=True,
        flip=True,
        noise_scale=config["noise_scale"],
        num_workers=8,
        seed=[1] + config["eval_seed"],
    )
    unsupervised_model = UnsupervisedModel(
        d_res=192,
        d_lstm=192,
        d_ff=192 * 2,
        d_edge=96,
        n_layers=4,
        n_bp=config["num_bodyparts"],
        sample_rate=config["sample_rate"],
        aggregation_radius=config["aggregation_radius"],
        dtype="bfloat16",
    )
    unsupervised_path = f"{persist_dir}/{config['name']}_unsupervised.pkl"
    supervised_model = SupervisedModel(
        d_res=256,
        d_ff=768,
        d_lstm=256,
        n_layers=3,
        n_bp=config["num_bodyparts"],
        padding=32,
        dtype="bfloat16",
        unsupervised_model=(unsupervised_model, unsupervised_path),
    )

    supervised_path = f"{persist_dir}/{config['name']}_supervised.pkl"
    supervised_params = pickle.load(open(supervised_path, "rb"))
    supervised_model.set_context({"stage": "eval"})
    supervised_model = supervised_model.set_variables(supervised_params)

    @jax.jit
    def prediction_step(batch, key):
        key, subkey = jax.random.split(key)
        probs = supervised_model.predict(batch, subkey)
        return probs, key

    def get_batch_outputs(batch):
        nonlocal key
        probs, key = prediction_step(batch, key)
        return batch, probs

    key = jax.random.key(0)
    elements = Pipeline(dataset.element_iterator())
    batches = elements.batch(256).to_device(sharding)
    batch_outputs = batches.map(get_batch_outputs).to_host()
    batch_outputs.unbatch().map(lambda x: predictions.update(*x)).last()

    if is_val:
        metrics, thresholds = predictions.score()
        os.makedirs(persist_dir, exist_ok=True)
        pickle.dump(thresholds, open(f"{persist_dir}/{config['name']}_thresholds.pkl", "wb"))
        print(metrics)


def train_ensemble():
    configs = get_configs()
    for config_name, config in configs.items():
        pretrain(config)
        train(config)
        predict(config)
    compute_ensemble_thresholds(configs)


def test_ensemble():
    configs = get_configs()
    print("Loading test videos...", flush=True)
    test_videos = load_videos(mode="test", use_cached=True)
    print(f"Loaded {len(test_videos)} test videos", flush=True)
    test_predictions = Predictions(test_videos)
    for i, (config_name, config) in enumerate(configs.items()):
        t0 = time.time()
        print(f"[{i+1}/{len(configs)}] Running config: {config_name}...", flush=True)
        predict(config, predictions=test_predictions)
        print(f"[{i+1}/{len(configs)}] {config_name} done in {time.time()-t0:.1f}s", flush=True)
    print("Generating submission...", flush=True)
    submission_df = test_predictions.to_submission_df()
    submission_path = os.path.join(project_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}", flush=True)


# Module-level path globals. Entry points reassign these (run_allbehaviors_perlab.py points
# dataset_dir/working_dir at whichever dataset is being run), so they stay plain strings rather
# than becoming lookups; see hidra/paths.py for how the defaults are resolved.
project_dir = str(paths.repo_root() or paths.PKG_DIR)
dataset_dir = str(paths.dataset_dir())
working_dir = str(paths.work_root() / "tmp")
persist_dir = str(paths.models_dir())

if __name__ == "__main__":
    mode = "test"  # 'train' requires TPU v5e-8, 'test' requires P100

    if mode == "train":
        train_ensemble()

    if mode == "test":
        test_ensemble()