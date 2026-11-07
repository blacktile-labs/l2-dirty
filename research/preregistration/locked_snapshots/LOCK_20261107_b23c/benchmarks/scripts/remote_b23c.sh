#!/usr/bin/env bash
# B23c (research/HYPOTHESES.md "B23c amendment") on a rented vast.ai instance: try to lock the SM clock at 75 % of
# the GPU's max SM clock (nvidia-smi -lgc), record whether the host allows it, run remote_b23.sh unchanged, and
# reset the clock (nvidia-smi -rgc) before the run is reported done, so a lock never outlives the rental.
set -uo pipefail
cd "$HOME/dmkl"
mkdir -p pilot_out
CL=pilot_out/clock_lock.txt
LOCKED=0
reset_clock() { [ "$LOCKED" = 1 ] && { nvidia-smi -rgc >> "$CL" 2>&1; echo "reset rc=$?" >> "$CL"; LOCKED=0; }; }
trap reset_clock EXIT
{ date -u; nvidia-smi -q -d CLOCK; } > pilot_out/clock_before.txt 2>&1
MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
if [ -n "$MAX" ]; then
  T=$((MAX * 3 / 4))
  { echo "max SM clock $MAX MHz, lock target $T MHz"; nvidia-smi -lgc "$T,$T"; echo "lock rc=$?"; } > "$CL" 2>&1
  grep -q "^lock rc=0" "$CL" && grep -qi "clocks set" "$CL" && LOCKED=1
else
  echo "max SM clock not readable; no lock attempted" > "$CL"
fi
echo "locked=$LOCKED target_mhz=${T:-}" >> "$CL"
bash benchmarks/scripts/remote_b23.sh
RC=$?
reset_clock
{ date -u; nvidia-smi -q -d CLOCK; } > pilot_out/clock_after.txt 2>&1
tar czf "$HOME/pilot_results.tgz" pilot_out benchmarks/results 2>/dev/null
exit $RC
