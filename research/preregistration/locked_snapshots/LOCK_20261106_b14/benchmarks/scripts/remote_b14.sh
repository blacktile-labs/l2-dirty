#!/usr/bin/env bash
# B14 on a rented vast.ai VM: the quantized-GEMV speedup-bias protocol (quant_bias_probe.sh). Timing only: no ncu.
# Runs ON the VM in ~/dmkl, started detached by benchmarks/scripts/vast/vast_run.sh.
# torch 2.7.1+cu126, triton 3.3.1 (Triton compiles its kernels itself; no CUDA toolkit needed).
#   0. environment; stop (exit 4) if nvidia-smi lists no GPU
#   1. python venv (apt only for python3-venv)
#   2. smoke run of the probe
#   3. quant_bias_probe.sh (3 runs, scored on the VM and again locally)
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
{ date -u; id; uname -a; cat /etc/os-release; nproc; free -g; nvidia-smi
  grep -i RmProfilingAdminOnly /proc/driver/nvidia/params; } > pilot_out/environment.txt 2>&1
gpu_ok || { step "no working NVIDIA driver"; echo NO_DRIVER > pilot_out/STATUS; exit 4; }

step "1 toolchain"
# unattended-upgrades can hold the dpkg lock past apt's 600 s timeout right after boot (B12 on the
# 2060S, D-062): stop it before using apt
$SUDO systemctl stop unattended-upgrades apt-daily.service apt-daily-upgrade.service apt-daily.timer \
  apt-daily-upgrade.timer > /dev/null 2>&1
# Triton JIT-compiles a C launcher per kernel: it needs gcc and Python.h besides python3-venv
if ! python3 -c "import ensurepip" 2>/dev/null || ! command -v gcc > /dev/null ||
   ! python3 -c "import os, sysconfig; assert os.path.exists(sysconfig.get_paths()['include'] + '/Python.h')" 2>/dev/null; then
  "${APT[@]}" update > pilot_out/apt.log 2>&1
  "${APT[@]}" install python3-venv python3-dev build-essential >> pilot_out/apt.log 2>&1
  step "apt rc=$?"
fi
VENV="$HOME/venv"
python3 -m venv "$VENV" && "$VENV/bin/pip" install -q --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple torch==2.7.1 triton==3.3.1 numpy==2.2.6 > pilot_out/torch_install.log 2>&1
"$VENV/bin/python" -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.is_available())" \
  | tee -a pilot_out/toolchain.txt | tee -a "$LOG"

step "2 smoke"
"$VENV/bin/python" benchmarks/scripts/quant_bias_probe.py --smoke --meta pilot_out/smoke_meta.json \
  > pilot_out/smoke.txt 2>&1
step "smoke rc=$? $(tail -1 pilot_out/smoke.txt)"

step "3 B14"
PY="$VENV/bin/python" bash benchmarks/scripts/quant_bias_probe.sh > pilot_out/b14.txt 2>&1
step "B14 rc=$?"
echo DONE > pilot_out/STATUS
step "4 pack"
