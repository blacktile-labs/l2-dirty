#!/usr/bin/env bash
# B10 pilot on a rented vast.ai VM (RTX 3090, GA102, sm_86): replicate B6b, B8 and B9b with the same
# code, cells, rules and locked predictions as on the RTX 3060 Laptop (research/HYPOTHESES.md "B10").
# Runs ON the VM in ~/dmkl (unpacked benchmarks/src, benchmarks/scripts, research/preregistration and
# the laptop's B6 reference binary). Started detached by benchmarks/scripts/vast/vast_run.sh.
# Works as root or as a sudo-capable user: every ncu run goes through $SUDO, because counters need
# admin rights on a native Linux driver.
#   0. record the environment; stop (exit 4) if there is no working GPU driver
#   1. toolchain pinned to the laptop's: CUDA 12.5 nvcc/cudart/cuobjdump + Nsight Compute from
#      cuda-nsight-compute-12-5 (laptop: nvcc V12.5.82, ncu 2024.2.1); torch venv in the background
#   2. build wb_probe for sm_86 with the laptop's build.sh; dump its SASS and the SASS of the laptop's
#      B6 binary (wb_probe.binary, e0c82a53...) for a local diff
#   3. counter permission check: ncu on wb_probe --smoke; stop (exit 3) if ERR_NVGPUCTRPERM
#   4. wb_probe_b6b.sh, wb_probe_b8.sh   (scored locally afterwards as well)
#   5. queued_wb_probe.sh (B9b) if the torch venv installed
#   6. pack everything into ~/pilot_results.tgz
set -uo pipefail
cd "$HOME/dmkl"
mkdir -p pilot_out
LOG=pilot_out/remote.log
step() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
pack() { tar czf "$HOME/pilot_results.tgz" pilot_out benchmarks/results benchmarks/build/*.txt benchmarks/build/*.log 2>/dev/null; }
trap pack EXIT
SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo -n"
APT=($SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y -q)
export PIP_NO_CACHE_DIR=1   # the offer has a 28 GB disk
REF_B6=benchmarks/results/wb_probe_20261105-042441/wb_probe.binary

step "0 environment"
{ date -u; id; uname -a; cat /etc/os-release; nproc; free -g; df -h "$HOME"; nvidia-smi;
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,pstate --format=csv
  grep -i RmProfilingAdminOnly /proc/driver/nvidia/params; ls /usr/local | grep -i cuda; } > pilot_out/environment.txt 2>&1
nvidia-smi -L > /dev/null 2>&1 || { step "no working NVIDIA driver"; echo NO_DRIVER > pilot_out/STATUS; exit 4; }
step "user $(id -un), sudo='${SUDO}'"

step "1 toolchain"
if [ ! -x /usr/local/cuda-12.5/bin/nvcc ] || [ ! -x /usr/local/cuda-12.5/bin/ncu ]; then
  curl -fsSL -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb &&
    $SUDO dpkg -i /tmp/cuda-keyring.deb > pilot_out/apt.log 2>&1
  "${APT[@]}" update >> pilot_out/apt.log 2>&1
  "${APT[@]}" install build-essential python3-venv cuda-nvcc-12-5 cuda-cudart-dev-12-5 cuda-cuobjdump-12-5 \
    cuda-nsight-compute-12-5 >> pilot_out/apt.log 2>&1
  step "apt rc=$?"
fi
export PATH=/usr/local/cuda-12.5/bin:$PATH
{ nvcc --version; ncu --version; } > pilot_out/toolchain.txt 2>&1
cat pilot_out/toolchain.txt | tee -a "$LOG"
VENV="$HOME/venv"
( python3 -m venv "$VENV" && "$VENV/bin/pip" install -q --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple torch==2.7.1 triton==3.3.1 numpy==2.2.6 &&
  "$VENV/bin/python" -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.is_available())"
) > pilot_out/torch_install.log 2>&1 &
TORCH_PID=$!

step "2 build"
NVCC=/usr/local/cuda-12.5/bin/nvcc CUDA_ARCH=sm_86 ONLY=wb_probe bash benchmarks/scripts/build.sh > pilot_out/build.txt 2>&1
step "build rc=$?"
sha256sum benchmarks/build/wb_probe | tee -a "$LOG"
cuobjdump -sass benchmarks/build/wb_probe > pilot_out/wb_probe_sass.txt 2>&1
[ -f "$REF_B6" ] && cuobjdump -sass "$REF_B6" > pilot_out/laptop_b6_sass.txt 2>&1

# ncu needs admin rights; run it (and the protocols that call it) as root, keeping PATH
NCURUN=($SUDO env "PATH=$PATH")
step "3 counter permission check"
"${NCURUN[@]}" ncu --target-processes application-only --metrics dram__bytes_read.sum,dram__bytes_write.sum \
  benchmarks/build/wb_probe --smoke > pilot_out/counter_check.txt 2>&1
if grep -q "ERR_NVGPUCTRPERM" pilot_out/counter_check.txt; then
  step "counters BLOCKED (ERR_NVGPUCTRPERM)"; echo BLOCKED > pilot_out/STATUS; exit 3
fi
grep -c "dram__bytes_read.sum" pilot_out/counter_check.txt | sed 's/^/counter rows: /' | tee -a "$LOG"

step "4 B6b"
"${NCURUN[@]}" bash benchmarks/scripts/wb_probe_b6b.sh > pilot_out/b6b.txt 2>&1
step "B6b rc=$?"
step "4 B8"
"${NCURUN[@]}" bash benchmarks/scripts/wb_probe_b8.sh > pilot_out/b8.txt 2>&1
step "B8 rc=$?"

step "5 B9b (waiting for the torch venv)"
if wait $TORCH_PID && "$VENV/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  "${NCURUN[@]}" PY="$VENV/bin/python" bash benchmarks/scripts/queued_wb_probe.sh > pilot_out/b9b.txt 2>&1
  step "B9b rc=$?"
else
  step "torch venv failed: B9b skipped"
fi
[ -n "$SUDO" ] && $SUDO chown -R "$(id -u):$(id -g)" pilot_out benchmarks/results 2>/dev/null
echo DONE > pilot_out/STATUS
step "6 pack"
