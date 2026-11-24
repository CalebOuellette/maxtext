# Copyright 2023–2026 Google LLC
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

"""Saved positions of the live ``dataset_type=olmo_mm`` loader, so a resumed run does not replay its data.

Without this a resume at step S builds and discards S batches (image decode, packing and collation included) to
put the loader back where it was: about 2 s a batch at 256K, 3 hours at step 5,625 (hero run, 2026-11-24), and
longer the later the checkpoint. OLMo-core's ``MixtureDataLoader`` can instead be restored from its
``state_dict()``: a cursor into the example stream plus the references of the examples waiting in the pack buffer.

``LoaderStateStore`` keeps those states as small files, one per loader rank every ``period`` batches:

  <state_dir>/world_<W>/rank_<R>/step_<N>.pkl      N = batches taken from the loader when the state was read

On resume at step S the newest state with N <= S is loaded and only the S - N batches after it are replayed
(fewer than ``period``). With no usable state the loader falls back to the full replay, and writes states as it
goes, so a run started before this existed pays the replay once.

The files are pickles: the loader compares the saved values (sizes, weights, fingerprints, group quotas) with its
own by equality, and a JSON round trip would turn tuples into lists. They are read only from the run's own
directory.
"""

import os
import pickle
import re
from typing import Any, Callable, Optional, Tuple

from etils import epath

_STEP_FILE = re.compile(r"^step_(\d+)\.pkl$")


class LoaderStateStore:
  """States of one loader rank under ``state_dir`` (local or ``gs://``)."""

  def __init__(
      self,
      state_dir: str,
      *,
      rank: int,
      world: int,
      period: int = 25,
      keep: int = 40,
      log: Callable[[str], None] = print,
  ):
    if period < 1:
      raise ValueError(f"period must be at least 1, got {period}")
    self.dir = epath.Path(state_dir) / f"world_{int(world)}" / f"rank_{int(rank):05d}"
    self.period = int(period)
    self.keep = max(1, int(keep))
    self._log = log
    self._save_errors = 0

  def _path(self, step: int) -> epath.Path:
    return self.dir / f"step_{int(step):08d}.pkl"

  def steps(self) -> list[int]:
    """Steps that have a saved state, ascending."""
    if not self.dir.exists():
      return []
    found = []
    for p in self.dir.iterdir():
      m = _STEP_FILE.match(p.name)
      if m:
        found.append(int(m.group(1)))
    return sorted(found)

  def due(self, step: int) -> bool:
    return step > 0 and step % self.period == 0

  def save(self, step: int, state: dict[str, Any]) -> bool:
    """Write the state for ``step`` and drop all but the newest ``keep``. Never raises: a failed save costs a
    longer replay on some later resume, which is not worth stopping training for."""
    try:
      self.dir.mkdir(parents=True, exist_ok=True)
      final = self._path(step)
      data = pickle.dumps({"step": int(step), "state": state}, protocol=pickle.HIGHEST_PROTOCOL)
      if str(self.dir).startswith("gs://"):
        final.write_bytes(data)  # an object appears whole or not at all
      else:
        tmp = self.dir / f".tmp_{os.getpid()}_{final.name}"
        tmp.write_bytes(data)
        tmp.rename(final)
      for old in self.steps()[: -self.keep]:
        self._path(old).unlink(missing_ok=True)
      return True
    except Exception as error:  # pylint: disable=broad-exception-caught
      self._save_errors += 1
      if self._save_errors <= 5:
        self._log(f"olmo_mm loader state: could not save step {step} under {self.dir}: {error!r}")
      return False

  def latest_at_or_before(self, step: int) -> Optional[Tuple[int, dict[str, Any]]]:
    """The newest readable state taken at or before ``step``, as ``(its step, state)``."""
    for candidate in reversed([s for s in self.steps() if s <= step]):
      try:
        saved = pickle.loads(self._path(candidate).read_bytes())
        if int(saved["step"]) != candidate:
          raise ValueError(f"file holds step {saved['step']}")
        return candidate, saved["state"]
      except Exception as error:  # pylint: disable=broad-exception-caught
        self._log(f"olmo_mm loader state: skipping unreadable {self._path(candidate)}: {error!r}")
    return None


def store_from_env(
    *,
    checkpoint_dir: str,
    rank: int,
    world: int,
    log: Callable[[str], None] = print,
) -> Optional[LoaderStateStore]:
  """The store for a training loader, or None when states are off.

  ``OLMO_MM_LOADER_STATE_DIR``     where to keep them; default ``<run dir>/olmo_mm_loader_state``, next to
                                   ``checkpoint_dir``. No directory (no checkpointing): states are off.
  ``OLMO_MM_LOADER_STATE_PERIOD``  batches between states (default 25; 0 turns states off).
  ``OLMO_MM_LOADER_STATE_KEEP``    states kept per rank (default 40).
  """
  period = int(os.environ.get("OLMO_MM_LOADER_STATE_PERIOD", "25") or 0)
  if period <= 0:
    return None
  state_dir = (os.environ.get("OLMO_MM_LOADER_STATE_DIR", "") or "").strip()
  if not state_dir:
    ckpt = (checkpoint_dir or "").strip().rstrip("/")
    if not ckpt:
      return None
    state_dir = ckpt.rsplit("/", 1)[0] + "/olmo_mm_loader_state"
  keep = int(os.environ.get("OLMO_MM_LOADER_STATE_KEEP", "40") or 40)
  return LoaderStateStore(state_dir, rank=rank, world=world, period=period, keep=keep, log=log)


class ResumableBatches:
  """Batches of a loader built by ``build()``, positioned at ``start_step`` and saving its state as it goes.

  ``build()`` returns a fresh loader with ``__iter__``, ``reshuffle()``, ``state_dict()`` and
  ``load_state_dict()`` (OLMo-core's ``MixtureDataLoader``). ``next()`` returns its batches from number
  ``start_step`` on; ``taken`` is how many batches have been drawn from the loader since batch 0.
  """

  def __init__(
      self,
      build: Callable[[], Any],
      *,
      start_step: int = 0,
      store: Optional[LoaderStateStore] = None,
      log: Callable[[str], None] = print,
  ):
    self._store = store
    self._log = log
    self._stash = None
    self.taken = 0
    self.restored_from: Optional[int] = None
    self.loader = build()
    self._iter = None
    start_step = int(start_step)

    found = store.latest_at_or_before(start_step) if (store is not None and start_step > 0) else None
    if found is not None:
      step, state = found
      try:
        self.loader.load_state_dict(state)
        self.loader.reshuffle()  # hands the restored epoch to the datasets
        self._iter = iter(self.loader)
        self.taken = step
        # The loader checks the state against its own configuration when it builds its first batch.
        first = self._draw()
        if self.taken > start_step:
          self._stash = first
        self.restored_from = step
        log(f"olmo_mm loader state: restored step {step} from {store.dir}; replaying {max(0, start_step - step)} batches")
      except Exception as error:  # pylint: disable=broad-exception-caught
        log(f"olmo_mm loader state: could not restore step {step} from {store.dir} ({error!r}); replaying from 0")
        close = getattr(self._iter, "close", None)
        if close is not None:
          close()  # stops the half-restored loader's prefetching
        self.loader = build()
        self._iter = None
        self._stash = None
        self.taken = 0
    elif start_step > 0:
      where = f"no saved state at or before it under {store.dir}" if store is not None else "loader states are off"
      log(f"olmo_mm loader: resuming at step {start_step} by replaying every batch ({where})")

    if self._iter is None:
      self._iter = iter(self.loader)
    while self.taken < start_step:
      self._draw()

  def _draw(self):
    try:
      batch = next(self._iter)
    except StopIteration:
      self.loader.reshuffle()
      self._iter = iter(self.loader)
      batch = next(self._iter)
    self.taken += 1
    if self._store is not None and self._store.due(self.taken):
      # Read on this thread right after the batch: with prefetching the loader reports the state that goes with
      # the last batch handed over, not with what its producer has built ahead.
      self._store.save(self.taken, self.loader.state_dict())
    return batch

  def __iter__(self):
    return self

  def __next__(self):
    if self._stash is not None:
      batch, self._stash = self._stash, None
      return batch
    return self._draw()
