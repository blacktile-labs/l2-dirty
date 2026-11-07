#!/usr/bin/env bash
# B12 on a rented vast.ai VM: the do_bench flush-bias protocol (dobench_flush_probe.sh).
# Runs ON the VM in ~/dmkl, started detached by benchmarks/scripts/vast/vast_run.sh.
# Toolchain as in B10/B11 for Turing/Ampere/Ada: Nsight Compute 2024.2.1 (cuda-nsight-compute-12-5),
# torch 2.7.1+cu126, triton 3.3.1. No CUDA compiler is needed (torch/cuBLAS kernels only).
#   0. environment; stop (exit 4) if nvidia-smi lists no GPU
#   1. ncu + python venv
#   2. smoke range run under ncu: stop (exit 3) on ERR_NVGPUCTRPERM or if DRAM metrics read n/a
#   3. dobench_flush_probe.sh (3 range + 3 timing runs, scored on the VM and again locally)
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
if [ ! -x /usr/local/cuda-12.5/bin/ncu ]; then
  curl -fsSL -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb &&
    $SUDO dpkg -i /tmp/cuda-keyring.deb > pilot_out/apt.log 2>&1
  "${APT[@]}" update >> pilot_out/apt.log 2>&1
  "${APT[@]}" install python3-venv cuda-nsight-compute-12-5 >> pilot_out/apt.log 2>&1
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
step "2 counter check (smoke range run)"
"${NCURUN[@]}" ncu --target-processes application-only --replay-mode app-range \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum "$VENV/bin/python" benchmarks/scripts/dobench_flush_probe.py \
  --mode range --smoke --meta pilot_out/smoke_meta.json > pilot_out/counter_check.txt 2>&1
if grep -q "ERR_NVGPUCTRPERM" pilot_out/counter_check.txt; then
  step "counters BLOCKED"; echo BLOCKED > pilot_out/STATUS; exit 3
fi
if grep -E "dram__bytes_(read|write)\.sum" pilot_out/counter_check.txt | grep -q "n/a"; then
  step "DRAM metrics n/a"; echo DRAM_NA > pilot_out/STATUS; exit 3
fi
grep -c "dram__bytes_write.sum" pilot_out/counter_check.txt | sed 's/^/counter rows: /' | tee -a "$LOG"

step "3 B12"
"${NCURUN[@]}" PY="$VENV/bin/python" bash benchmarks/scripts/dobench_flush_probe.sh > pilot_out/b12.txt 2>&1
step "B12 rc=$?"
[ -n "$SUDO" ] && $SUDO chown -R "$(id -u):$(id -g)" pilot_out benchmarks/results 2>/dev/null
echo DONE > pilot_out/STATUS
step "4 pack"
