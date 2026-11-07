"""Candidate replacement models for a model v5 -- ESTIMATES only; hypotheses under test in B8
(research/HYPOTHESES.md "B8 pre-registration").

Why: near L2 capacity every model so far mispredicts in both directions (D-038 stencil, D-045 B7).
B6c (D-040) showed, under capacity pressure, that written lines are written back every launch
(WB ~ 1) yet mostly stay resident (readback misses 0.03-0.31 where LRU predicts 0.33-1.0), while
read lines are all evicted (RD = 1). Finn (2026, RTX 4090 blog) describes exactly this: a dirty
victim is not evicted at once -- it is written back and stays readable -- and the replacement scan
continues to a clean victim. (B6c's "clean-first" model did NOT write the dirty line back, so B6c
rejected that variant only.)

All models keep D1 (dirty cap, cache_sim_v4) and differ in what happens when the replacement
candidate (the LRU line) is dirty:
  "lru"       write it back and evict it                       (= cache_sim_v4.DirtyCapLRU)
  "dvsc_mru"  write it back, keep it, move it to MRU (a full second life), evict the next clean
              LRU candidate                                    (dirty-victim second chance)
  "dvsc_skip" write it back, keep it in place (clean, still LRU-most), evict the next candidate;
              it goes next time unless it is touched again    (one-fill reprieve)
  "dvsc_life" write it back, keep it resident in a write-back FIFO for `life_frac` x capacity
              further fills (still readable; a hit returns it to the LRU list), then evict it.
              Finn: a dirty victim "survives 2D-1 extra fills" (per set, write-back buffer depth D);
              one global lifetime is the fully-associative approximation. life_frac is fitted.
  `cap_frac` < 1 shrinks the usable capacity (an "effective capacity" hypothesis).
"""

from __future__ import annotations

from collections import OrderedDict

from cache_sim_v4 import DIRTY_CAP_FRACTION

POLICIES = ("lru", "dvsc_mru", "dvsc_skip", "dvsc_life")


class CandidateL2:
    def __init__(self, capacity_lines: int, policy: str = "dvsc_mru", cap_frac: float = 1.0,
                 dirty_cap_lines: int | None = None, life_frac: float = 0.0):
        if policy not in POLICIES:
            raise ValueError(policy)
        self.full = capacity_lines
        self.cap = max(1, int(capacity_lines * cap_frac))
        self.dcap = int(capacity_lines * DIRTY_CAP_FRACTION) if dirty_cap_lines is None else dirty_cap_lines
        self.policy = policy
        self.lru: OrderedDict[int, bool] = OrderedDict()     # line -> dirty, recency order
        self.stores: OrderedDict[int, None] = OrderedDict()  # dirty lines, store order
        self.reads = self.writebacks = self.cleanings = 0
        self.life = max(1, int(life_frac * capacity_lines))
        self.fills = 0
        self.wbq: OrderedDict[int, int] = OrderedDict()      # dvsc_life: line -> fill count at entry

    def reset_counters(self) -> None:
        self.reads = self.writebacks = self.cleanings = 0

    def _clean(self, line: int) -> None:
        self.lru[line] = False  # keeps its recency position
        del self.stores[line]
        self.writebacks += 1

    def _store(self, line: int) -> None:
        self.stores[line] = None
        self.stores.move_to_end(line)
        while len(self.stores) > self.dcap:            # D1
            old = next(iter(self.stores))
            self._clean(old)
            self.cleanings += 1

    def _evict(self) -> None:
        d = self.lru
        if self.policy == "dvsc_life":
            if self.wbq:                                # expired write-back buffer entry goes first
                old, t = next(iter(self.wbq.items()))
                if self.fills - t >= self.life:
                    del self.wbq[old]
                    return
            while d:
                victim, dirty = next(iter(d.items()))
                del d[victim]
                if not dirty:
                    return
                self.writebacks += 1                    # written back, parked in the FIFO
                del self.stores[victim]
                self.wbq[victim] = self.fills
            old = next(iter(self.wbq))                  # everything resident is parked: evict oldest
            del self.wbq[old]
            return
        if self.policy == "lru":
            victim, dirty = d.popitem(last=False)
            if dirty:
                self.writebacks += 1
                del self.stores[victim]
            return
        skipped = []
        while True:
            victim, dirty = next(iter(d.items()))
            if not dirty:
                del d[victim]
                break
            self._clean(victim)                         # written back, stays resident
            if self.policy == "dvsc_mru":
                d.move_to_end(victim)
            else:                                       # dvsc_skip: set aside, restore at LRU end
                del d[victim]
                skipped.append(victim)
        for v in reversed(skipped):
            d[v] = False
            d.move_to_end(v, last=False)

    def access(self, line: int, write: bool) -> None:
        d = self.lru
        if line in self.wbq:                            # dvsc_life: hit in the write-back FIFO
            del self.wbq[line]
            d[line] = False
        if line in d:
            d.move_to_end(line)
            if write:
                d[line] = True
                self._store(line)
            return
        if not write:
            self.reads += 1
        if len(d) + len(self.wbq) >= self.cap:
            self._evict()
        self.fills += 1
        d[line] = write
        if write:
            self._store(line)
