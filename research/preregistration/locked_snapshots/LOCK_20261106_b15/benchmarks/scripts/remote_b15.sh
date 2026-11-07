#!/usr/bin/env bash
# B15 on a rented vast.ai VM: (1) the absorption protocol with counters (absorb_probe.sh) and
# (2) the unchanged B14 protocol (quant_bias_probe.sh) as a replication on this GPU.
# Runs ON the VM in ~/dmkl, started detached by benchmarks/scripts/vast/vast_run.sh.
# Toolchain as B12-B14: Nsight Compute 2024.2.1 (cuda-nsight-compute-12-5), torch 2.7.1+cu126,
# triton 3.3.1 (+ gcc and Python.h for Triton's launcher).
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
$SUDO systemctl stop unattended-upgrades apt-daily.service apt-daily-upgrade.service apt-daily.timer \
  apt-daily-upgrade.timer > /dev/null 2>&1
if [ ! -x /usr/local/cuda-12.5/bin/ncu ] || ! command -v gcc > /dev/null ||
   ! python3 -c "import os, sysconfig; assert os.path.exists(sysconfig.get_paths()['include'] + '/Python.h')" 2>/dev/null; then
  curl -fsSL -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb &&
    $SUDO dpkg -i /tmp/cuda-keyring.deb > pilot_out/apt.log 2>&1
  "${APT[@]}" update >> pilot_out/apt.log 2>&1
  "${APT[@]}" install python3-venv python3-dev build-essential cuda-nsight-compute-12-5 >> pilot_out/apt.log 2>&1
  step "apt rc=$?"
fi
export PATH=/usr/local/cuda-12.5/bin:$PATH
ncu --version > pilot_out/toolchain.txt 2>&1
VENV="$HOME/venv"
python3 -m venv "$VENV" && "$VENV/bin/pip" install -q --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple torch==2.7.1 triton==3.3.1 numpy==2.2.6 > pilot_out/torch_install.log 2>&1
"$VENV/bin/python" -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.is_available())" \
  | tee -a pilot_out/toolchain.txt | tee -a "$LOG"

NCURUN=($SUDO env "PATH=$PATH")
step "2 counter check (absorb smoke under ncu)"
"${NCURUN[@]}" ncu --target-processes application-only --replay-mode app-range \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum "$VENV/bin/python" benchmarks/scripts/absorb_probe.py \
  --mode range --smoke --meta pilot_out/smoke_meta.json > pilot_out/counter_check.txt 2>&1
if grep -q "ERR_NVGPUCTRPERM" pilot_out/counter_check.txt; then
  step "counters BLOCKED"; echo BLOCKED > pilot_out/STATUS; exit 3
fi
if grep -E "dram__bytes_(read|write)\.sum" pilot_out/counter_check.txt | grep -q "n/a"; then
  step "DRAM metrics n/a"; echo DRAM_NA > pilot_out/STATUS; exit 3
fi
grep -c "dram__bytes_write.sum" pilot_out/counter_check.txt | sed 's/^/counter rows: /' | tee -a "$LOG"

step "3 B15 absorb"
"${NCURUN[@]}" PY="$VENV/bin/python" bash benchmarks/scripts/absorb_probe.sh > pilot_out/b15.txt 2>&1
step "B15 rc=$?"
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "4080 SUPER"; then
  step "4 B14 replication skipped (RTX 4080 SUPER measured in B14; budget)"
else
  step "4 B14 replication"
  PY="$VENV/bin/python" bash benchmarks/scripts/quant_bias_probe.sh > pilot_out/b14.txt 2>&1
  step "B14 rc=$?"
fi
[ -n "$SUDO" ] && $SUDO chown -R "$(id -u):$(id -g)" pilot_out benchmarks/results 2>/dev/null
echo DONE > pilot_out/STATUS
step "5 pack"
