#!/usr/bin/env bash
# B21 on a rented vast.ai instance (docker container or VM): e2e_probe.sh. Timing only: no ncu,
# so containers are fine. Runs in ~/dmkl, started detached by benchmarks/scripts/vast/vast_run.sh.
# torch 2.7.1 + triton 3.3.1 (cu126 wheels; cu128 for compute capability >= 12, i.e. Blackwell),
# bitsandbytes 0.50.2, liger-kernel 0.6.2 (installed --no-deps so it cannot change torch/triton).
#   0. environment; stop (exit 4) if nvidia-smi lists no GPU
#   1. python venv (apt only for python3, venv, headers and gcc, which Triton's launcher build needs)
#   2. smoke run of the probe
#   3. e2e_probe.sh (3 runs, scored on the instance and again locally)
#   4. pack into ~/pilot_results.tgz
set -uo pipefail
cd "$HOME/dmkl"
mkdir -p pilot_out
LOG=pilot_out/remote.log
step() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
pack() { tar czf "$HOME/pilot_results.tgz" pilot_out benchmarks/results 2>/dev/null; }
trap pack EXIT
SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo -n"
APT=($SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y -q)
export PIP_NO_CACHE_DIR=1
gpu_ok() { nvidia-smi -L 2>/dev/null | grep -q "^GPU [0-9]"; }

step "0 environment"
{ date -u; id; uname -a; cat /etc/os-release; nproc; free -g; nvidia-smi; nvidia-smi -q | head -60
  echo "== counter access (RmProfilingAdminOnly=0 means ncu could read counters on this host)"
  grep -i RmProfilingAdminOnly /proc/driver/nvidia/params; } > pilot_out/environment.txt 2>&1
gpu_ok || { step "no working NVIDIA driver"; echo NO_DRIVER > pilot_out/STATUS; exit 4; }

step "1 toolchain"
$SUDO systemctl stop unattended-upgrades apt-daily.service apt-daily-upgrade.service apt-daily.timer \
  apt-daily-upgrade.timer > /dev/null 2>&1
if ! command -v python3 > /dev/null || ! python3 -c "import ensurepip" 2>/dev/null || ! command -v gcc > /dev/null ||
   ! python3 -c "import os, sysconfig; assert os.path.exists(sysconfig.get_paths()['include'] + '/Python.h')" 2>/dev/null; then
  "${APT[@]}" update > pilot_out/apt.log 2>&1
  "${APT[@]}" install python3 python3-venv python3-dev build-essential >> pilot_out/apt.log 2>&1
  step "apt rc=$?"
fi
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); MAJ=${CC%%.*}
TIDX=cu126; [ "${MAJ:-0}" -ge 12 ] && TIDX=cu128
step "compute capability $CC -> torch index $TIDX"
VENV="$HOME/venv"
# liger-kernel goes in the same resolve as the pinned torch/triton (its only deps), with retries: on the
# B17 pilot host PyPI reset the connection for a separate liger-kernel install (D-076)
python3 -m venv "$VENV"
for attempt in 1 2 3; do
  "$VENV/bin/pip" install -q --retries 10 --timeout 60 --index-url "https://download.pytorch.org/whl/$TIDX" \
    --extra-index-url https://pypi.org/simple torch==2.7.1 triton==3.3.1 numpy==2.2.6 bitsandbytes==0.50.2 \
    packaging liger-kernel==0.6.2 >> pilot_out/torch_install.log 2>&1 && break
  step "pip attempt $attempt failed"; sleep 15
done
"$VENV/bin/python" -c "import torch, triton, importlib.metadata as m; print(torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.is_available(), 'bnb', m.version('bitsandbytes'), 'liger', m.version('liger-kernel'))" \
  | tee -a pilot_out/toolchain.txt | tee -a "$LOG"

step "2 smoke"
(cd benchmarks/scripts && "$VENV/bin/python" e2e_probe.py --smoke --meta ../../pilot_out/smoke_meta.json) \
  > pilot_out/smoke.txt 2>&1
step "smoke rc=$? $(tail -1 pilot_out/smoke.txt | cut -c1-300)"

step "3 B21"
PY="$VENV/bin/python" bash benchmarks/scripts/e2e_probe.sh > pilot_out/b21.txt 2>&1
step "B21 rc=$?"
echo DONE > pilot_out/STATUS
step "4 pack"
