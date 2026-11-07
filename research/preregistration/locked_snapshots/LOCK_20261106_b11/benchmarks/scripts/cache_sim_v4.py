"""Model v4 L2: write-back LRU (as cache_sim.Cache "lru") + a dirty-occupancy cap -- ESTIMATES only.

Why: B6/B6b (DECISION_LOG D-034, D-036) measured with DRAM counters on the RTX 3060 Laptop that
the L2 writes dirty data back *without evicting it* once the number of dirty lines exceeds about
half of the L2: steady write-back per launch was 0.014 / 0.194 / 0.886 / 0.991 of the written
bytes at a dirty footprint of 0.4375 / 0.5 / 0.5625 / 0.625 L2, independent of the clean footprint
(up to 0.69 L2), counted in lines (one dirty sector per line triggers it too), and the cleaned
lines stayed resident. The locked simulator (cache_sim.py, assumption S3) only writes back on
eviction, so it predicts zero for all of these.

Model (one new assumption, D1; everything else as cache_sim.Cache "lru"):
  D1  the cache holds at most `dirty_cap` dirty lines. A store that makes the dirty count exceed
      the cap cleans the least-recently-*stored* dirty line: its 128 B are written back
      (counted in `writebacks`) and the line stays resident and clean, keeping its LRU position.
      dirty_cap = capacity // 2 (fixed from B6/B6b; NOT fitted to B4/B5 counter data).
The real L2 is set-associative with a hashed index; a single global cap makes the step perfectly
sharp (0 at <= 0.5, 1 above), whereas B6b measured 0.19 at exactly 0.5 and 0.89 at 0.5625. This
boundary error is a declared limitation.
"""

from __future__ import annotations

from collections import OrderedDict

LINE = 128
DIRTY_CAP_FRACTION = 0.5


class DirtyCapLRU:
    def __init__(self, capacity_lines: int, dirty_cap_lines: int | None = None):
        if capacity_lines <= 0:
            raise ValueError("capacity must be > 0")
        self.cap = capacity_lines
        self.dcap = int(capacity_lines * DIRTY_CAP_FRACTION) if dirty_cap_lines is None else dirty_cap_lines
        if self.dcap <= 0:
            raise ValueError("dirty cap must be > 0")
        self.lru: OrderedDict[int, bool] = OrderedDict()   # line -> dirty, in recency order
        self.stores: OrderedDict[int, None] = OrderedDict()  # dirty lines, oldest store first
        self.reads = 0
        self.writebacks = 0      # eviction write-backs + cleanings
        self.cleanings = 0       # the D1 part of writebacks

    def reset_counters(self) -> None:
        self.reads = self.writebacks = self.cleanings = 0

    def _store(self, line: int) -> None:
        self.stores[line] = None
        self.stores.move_to_end(line)
        while len(self.stores) > self.dcap:
            old, _ = self.stores.popitem(last=False)
            self.lru[old] = False  # clean; assignment keeps its LRU position
            self.writebacks += 1
            self.cleanings += 1

    def access(self, line: int, write: bool) -> None:
        d = self.lru
        if line in d:
            d.move_to_end(line)
            if write:
                d[line] = True
                self._store(line)
            return
        if not write:
            self.reads += 1
        d[line] = write
        if len(d) > self.cap:
            victim, was_dirty = d.popitem(last=False)
            if was_dirty:
                self.writebacks += 1
                del self.stores[victim]
        if write:
            self._store(line)
