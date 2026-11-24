# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resuming the olmo_mm loader from a saved state gives the batches a full replay gives."""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from maxtext.input_pipeline import olmo_mm_loader_state as ls


class _FakeLoader:
  """A stateful stream with a buffer, like the packed mixture loader: batch n depends on everything before it."""

  def __init__(self, tag: int = 7, epoch_len: int = 0):
    self.tag = tag
    self.cursor = 0
    self.buffer = [0, 0, 0]
    self.epoch_len = epoch_len  # > 0: iteration ends every epoch_len batches, as a finite loader's does
    self.epoch = 1
    self.built = 0

  def __iter__(self):
    n = 0
    while not self.epoch_len or n < self.epoch_len:
      self.buffer = self.buffer[1:] + [self.cursor * self.tag + self.epoch]
      self.cursor += 1
      n += 1
      self.built += 1
      yield (self.cursor - 1, tuple(self.buffer))

  def reshuffle(self):
    pass

  def state_dict(self):
    return {"cursor": self.cursor, "buffer": tuple(self.buffer), "tag": self.tag}

  def load_state_dict(self, state):
    if state["tag"] != self.tag:
      raise ValueError("state is for another loader")
    self.cursor = state["cursor"]
    self.buffer = list(state["buffer"])


def _take(batches, n):
  return [next(batches) for _ in range(n)]


class LoaderStateStoreTest(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)
    self.logs = []

  def _store(self, **kwargs):
    values = {"rank": 3, "world": 8, "period": 5, "keep": 4, "log": self.logs.append}
    values.update(kwargs)
    return ls.LoaderStateStore(self._tmp.name, **values)

  def test_save_latest_and_prune(self):
    store = self._store()
    self.assertEqual(store.steps(), [])
    self.assertIsNone(store.latest_at_or_before(100))
    for step in range(5, 45, 5):
      self.assertTrue(store.save(step, {"cursor": step}))
    self.assertEqual(store.steps(), [25, 30, 35, 40])  # the newest four
    self.assertEqual(store.latest_at_or_before(37), (35, {"cursor": 35}))
    self.assertEqual(store.latest_at_or_before(40), (40, {"cursor": 40}))
    self.assertIsNone(store.latest_at_or_before(24))
    self.assertTrue(os.path.isdir(os.path.join(self._tmp.name, "world_8", "rank_00003")))
    self.assertFalse(any(name.startswith(".tmp") for name in os.listdir(store.dir)))

  def test_unreadable_state_falls_back_to_an_older_one(self):
    store = self._store()
    store.save(5, {"cursor": 5})
    store.save(10, {"cursor": 10})
    (store.dir / "step_00000010.pkl").write_bytes(b"truncated")
    self.assertEqual(store.latest_at_or_before(12), (5, {"cursor": 5}))
    self.assertTrue(any("unreadable" in line for line in self.logs))

  def test_a_failed_save_does_not_raise(self):
    store = self._store()
    with mock.patch.object(ls.pickle, "dumps", side_effect=OSError("disk full")):
      self.assertFalse(store.save(5, {"cursor": 5}))
    self.assertTrue(any("could not save" in line for line in self.logs))

  def test_store_from_env(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      store = ls.store_from_env(checkpoint_dir="gs://bucket/runs/r1/checkpoints/", rank=0, world=2)
      self.assertEqual(str(store.dir), "gs://bucket/runs/r1/olmo_mm_loader_state/world_2/rank_00000")
      self.assertEqual((store.period, store.keep), (25, 40))
      self.assertIsNone(ls.store_from_env(checkpoint_dir="", rank=0, world=2))
    with mock.patch.dict(os.environ, {"OLMO_MM_LOADER_STATE_PERIOD": "0"}, clear=True):
      self.assertIsNone(ls.store_from_env(checkpoint_dir="/ckpt", rank=0, world=2))
    env = {"OLMO_MM_LOADER_STATE_DIR": "/states", "OLMO_MM_LOADER_STATE_PERIOD": "10", "OLMO_MM_LOADER_STATE_KEEP": "3"}
    with mock.patch.dict(os.environ, env, clear=True):
      store = ls.store_from_env(checkpoint_dir="", rank=1, world=2)
      self.assertEqual((str(store.dir), store.period, store.keep), ("/states/world_2/rank_00001", 10, 3))


class ResumableBatchesTest(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)
    self.logs = []

  def _store(self, **kwargs):
    values = {"rank": 0, "world": 1, "period": 5, "keep": 40, "log": self.logs.append}
    values.update(kwargs)
    return ls.LoaderStateStore(self._tmp.name, **values)

  def _reference(self, n, **kwargs):
    return _take(ls.ResumableBatches(lambda: _FakeLoader(**kwargs), log=self.logs.append), n)

  def test_restore_matches_replay_at_every_resume_step(self):
    reference = self._reference(40)
    first = ls.ResumableBatches(_FakeLoader, store=self._store(), log=self.logs.append)
    self.assertEqual(_take(first, 23), reference[:23])
    self.assertEqual(self._store().steps(), [5, 10, 15, 20])
    for start in (5, 6, 9, 10, 17, 20, 21, 23):
      built = []

      def build():
        built.append(_FakeLoader())
        return built[-1]

      resumed = ls.ResumableBatches(build, start_step=start, store=self._store(), log=self.logs.append)
      self.assertEqual(resumed.restored_from, start - start % 5)
      self.assertEqual(resumed.taken, max(start, resumed.restored_from + 1))
      self.assertEqual(_take(resumed, 40 - start), reference[start:], msg=f"start={start}")
      # Only the batches after the state were built again, plus the 40 - start that were asked for.
      self.assertEqual(built[-1].built, 40 - resumed.restored_from, msg=f"start={start}")

  def test_resume_without_a_state_replays_and_saves_states_on_the_way(self):
    reference = self._reference(30)
    resumed = ls.ResumableBatches(_FakeLoader, start_step=12, store=self._store(), log=self.logs.append)
    self.assertIsNone(resumed.restored_from)
    self.assertEqual(self._store().steps(), [5, 10])
    self.assertEqual(_take(resumed, 18), reference[12:])
    self.assertTrue(any("replaying every batch" in line for line in self.logs))
    # The states written during that replay serve the next resume.
    again = ls.ResumableBatches(_FakeLoader, start_step=12, store=self._store(), log=self.logs.append)
    self.assertEqual(again.restored_from, 10)
    self.assertEqual(_take(again, 18), reference[12:])

  def test_a_state_the_loader_rejects_falls_back_to_replay(self):
    first = ls.ResumableBatches(lambda: _FakeLoader(tag=7), store=self._store(), log=self.logs.append)
    _take(first, 12)
    reference = self._reference(30, tag=9)
    resumed = ls.ResumableBatches(lambda: _FakeLoader(tag=9), start_step=12, store=self._store(), log=self.logs.append)
    self.assertIsNone(resumed.restored_from)
    self.assertEqual(_take(resumed, 18), reference[12:])
    self.assertTrue(any("could not restore" in line for line in self.logs))

  def test_restore_off_replays_every_batch_and_still_saves(self):
    reference = self._reference(30)
    _take(ls.ResumableBatches(_FakeLoader, store=self._store(keep=2), log=self.logs.append), 12)
    self.assertEqual(self._store().steps(), [5, 10])
    built = []

    def build():
      built.append(_FakeLoader())
      return built[-1]

    resumed = ls.ResumableBatches(build, start_step=12, store=self._store(keep=2), restore=False, log=self.logs.append)
    self.assertIsNone(resumed.restored_from)
    self.assertEqual(_take(resumed, 18), reference[12:])
    self.assertEqual(built[-1].built, 30)  # every batch from 0
    self.assertEqual(self._store().steps(), [25, 30])
    self.assertTrue(any("OLMO_MM_LOADER_STATE_RESTORE=0" in line for line in self.logs))

  def test_restore_from_env(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertTrue(ls.restore_from_env())
    with mock.patch.dict(os.environ, {"OLMO_MM_LOADER_STATE_RESTORE": "0"}, clear=True):
      self.assertFalse(ls.restore_from_env())

  def test_without_a_store_it_replays(self):
    reference = self._reference(20)
    resumed = ls.ResumableBatches(_FakeLoader, start_step=7, log=self.logs.append)
    self.assertEqual(_take(resumed, 13), reference[7:])

  def test_finite_loader_reshuffles_and_continues(self):
    batches = ls.ResumableBatches(lambda: _FakeLoader(epoch_len=4), store=self._store(), log=self.logs.append)
    self.assertEqual([b[0] for b in _take(batches, 10)], list(range(10)))
    self.assertEqual(batches.taken, 10)


class MixtureDataLoaderResumeTest(unittest.TestCase):
  """The same comparison on OLMo-core's grouped, buffered-packing ``MixtureDataLoader`` (the 256K recipe's
  mode), with and without prefetching. Skipped where OLMo-core is not installed."""

  def setUp(self):
    try:
      import torch  # pylint: disable=import-outside-toplevel,unused-import
      from olmo_core.data.multimodal import mixture_data_loader  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError as error:
      self.skipTest(f"OLMo-core not available: {error}")
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)

  def _build(self, rank=0, **kwargs):
    from olmo_core.data.multimodal.collator import MultimodalCollator  # pylint: disable=import-outside-toplevel
    from olmo_core.data.multimodal.mixture_data_loader import MixtureDataLoader  # pylint: disable=import-outside-toplevel

    class Dataset:
      def __init__(self, tag):
        self.tag = tag
        self.content_fingerprint = f"source-{tag}"

      def __len__(self):
        return 19

      def get(self, index, epoch):
        size = 3 + index % 3
        tokens = np.full(size, self.tag + index + 1000 * epoch, dtype=np.int64)
        return {
            "input_ids": tokens,
            "labels": tokens.copy(),
            "loss_masks": np.full(size, 0.5, dtype=np.float32),
            "position_ids": np.arange(size, dtype=np.int64),
            "token_type_ids": np.zeros(size, dtype=np.int64),
            "images": np.zeros((0, 2, 3), dtype=np.float32),
            "pooled_patches_idx": np.zeros((0, 1), dtype=np.int64),
        }

    values = {
        "work_dir": os.path.join(self._tmp.name, "work"),
        "global_batch_size": 8 * 16,
        "seed": 17,
        "epoch_instances": 240,
        "pack": True,
        "pack_max_crops": 2,
        "pack_buffer_size": 4,
        "continuous_stream": True,
        "dp_world_size": 2,
        "dp_rank": rank,
        "dataset_names": ["replay", "caption", "pointing"],
        "source_groups": {"replay": "text", "caption": "vision", "pointing": "vision"},
        "group_sequence_quotas": {"vision": 6, "text": 2},
    }
    values.update(kwargs)
    loader = MixtureDataLoader(
        [Dataset(10), Dataset(100), Dataset(200)],
        [0.001, 0.399, 0.6],
        MultimodalCollator(pad_token_id=0, pad_sequence_length=16),
        **values,
    )
    loader.reshuffle(epoch=1)
    return loader

  def _assert_same(self, actual, expected, msg):
    import torch  # pylint: disable=import-outside-toplevel

    self.assertEqual(len(actual), len(expected), msg=msg)
    for a, e in zip(actual, expected):
      self.assertEqual(a.keys(), e.keys(), msg=msg)
      for key in a:
        if isinstance(a[key], torch.Tensor):
          self.assertTrue(torch.equal(a[key], e[key]), msg=f"{msg}: {key}")
        else:
          self.assertEqual(a[key], e[key], msg=f"{msg}: {key}")

  def _check(self, **loader_kwargs):
    logs = []
    for rank in (0, 1):
      build = lambda rank=rank: self._build(rank=rank, **loader_kwargs)
      store = lambda rank=rank: ls.LoaderStateStore(
          os.path.join(self._tmp.name, "states"), rank=rank, world=2, period=4, log=logs.append
      )
      reference = _take(ls.ResumableBatches(build, log=logs.append), 30)
      # A first run that stops after 14 batches, having saved states at 4, 8 and 12.
      _take(ls.ResumableBatches(build, store=store(), log=logs.append), 14)
      self.assertEqual(store().steps(), [4, 8, 12])
      for start in (4, 7, 12, 13, 14):
        resumed = ls.ResumableBatches(build, start_step=start, store=store(), log=logs.append)
        self.assertEqual(resumed.restored_from, start - start % 4, msg="\n".join(logs))
        self._assert_same(_take(resumed, 30 - start), reference[start:], msg=f"rank={rank} start={start}")
      # Replay with no state (what a run without saved states does) agrees too.
      replayed = ls.ResumableBatches(build, start_step=13, log=logs.append)
      self._assert_same(_take(replayed, 17), reference[13:], msg=f"rank={rank} replay")
      for name in os.listdir(store().dir):
        os.remove(os.path.join(store().dir, name))

  def test_grouped_buffered_packing(self):
    self._check()

  def test_grouped_buffered_packing_with_batch_prefetch(self):
    self._check(batch_prefetch_depth=2)

  def test_grouped_buffered_packing_with_thread_workers(self):
    self._check(prefetch_workers=2, batch_prefetch_depth=2)

  def test_grouped_buffered_packing_with_process_workers(self):
    # The producer's settings (olmo_mm_producer.py): process workers behind a batch producer process.
    self._check(prefetch_workers=2, prefetch_processes=True, batch_prefetch_depth=2, batch_prefetch_process=True)


if __name__ == "__main__":
  unittest.main()
