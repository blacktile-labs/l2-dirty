"""Candidate D2 for a model v5: model v4 (LRU + dirty cap) with *clean-first* victim selection --
ESTIMATES only; a hypothesis under test in B6c (HYPOTHESES.md "B6c pre-registration").

Why: model v4 over-predicted DRAM writes of B5 mat4 / sweep_warm by 0.4-0.5 arrays when the
working set exceeds L2 (D-038). One candidate explanation (post hoc) is that the real L2 prefers
to evict clean lines, so dirty lines live longer and are written back less often. Finn (2026,
RTX 4090 blog) describes a replacement scan that defers dirty victims and keeps looking for a
clean one; this is the simplest version of that idea.

D1 (unchanged from cache_sim_v4): at most `dirty_cap` dirty lines; a store that exceeds it cleans
    the least-recently-stored dirty line (written back, stays resident, keeps its recency).
D2 (new): on a miss in a full cache, evict the least-recently-used *clean* line; only when every
    resident line is dirty, evict the least-recently-used line (and write it back).
With D1 at half the capacity a clean line always exists, so D2 never evicts a dirty line.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict

from cache_sim_v4 import DIRTY_CAP_FRACTION


class CleanFirstDirtyCap:
    def __init__(self, capacity_lines: int, dirty_cap_lines: int | None = None):
        if capacity_lines <= 0:
            raise ValueError("capacity must be > 0")
        self.cap = capacity_lines
        self.dcap = int(capacity_lines * DIRTY_CAP_FRACTION) if dirty_cap_lines is None else dirty_cap_lines
        if self.dcap <= 0:
            raise ValueError("dirty cap must be > 0")
        self.t = 0
        self.ts: dict[int, int] = {}        # resident line -> last access time
        self.dirty: dict[int, bool] = {}
        self.stores: OrderedDict[int, None] = OrderedDict()
        self.all_heap: list[tuple[int, int]] = []    # (time, line), lazy
        self.clean_heap: list[tuple[int, int]] = []  # (time, line) of clean lines, lazy
        self.reads = self.writebacks = self.cleanings = 0

    def reset_counters(self) -> None:
        self.reads = self.writebacks = self.cleanings = 0

    def _touch(self, line: int) -> None:
        self.t += 1
        self.ts[line] = self.t
        heapq.heappush(self.all_heap, (self.t, line))
        if not self.dirty[line]:
            heapq.heappush(self.clean_heap, (self.t, line))

    def _remove(self, line: int) -> None:
        del self.ts[line]
        if self.dirty.pop(line):
            self.writebacks += 1
            del self.stores[line]

    def _evict(self) -> None:
        while self.clean_heap:
            t, ln = heapq.heappop(self.clean_heap)
            if self.ts.get(ln) == t and not self.dirty[ln]:
                self._remove(ln)
                return
        while True:
            t, ln = heapq.heappop(self.all_heap)
            if self.ts.get(ln) == t:
                self._remove(ln)
                return

    def _store(self, line: int) -> None:
        self.stores[line] = None
        self.stores.move_to_end(line)
        while len(self.stores) > self.dcap:
            old, _ = self.stores.popitem(last=False)
            self.dirty[old] = False
            heapq.heappush(self.clean_heap, (self.ts[old], old))  # keeps its recency
            self.writebacks += 1
            self.cleanings += 1

    def access(self, line: int, write: bool) -> None:
        if line in self.ts:
            if write:
                self.dirty[line] = True
            self._touch(line)
            if write:
                self._store(line)
            return
        if not write:
            self.reads += 1
        if len(self.ts) >= self.cap:
            self._evict()
        self.dirty[line] = write
        self._touch(line)
        if write:
            self._store(line)
