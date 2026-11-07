#!/usr/bin/env bash
# B10 / B11 on a rented vast.ai VM: replicate B6b, B8 and B9b with the same code, cells, rules and
# locked predictions as on the RTX 3060 Laptop (research/HYPOTHESES.md "B10", "B11"), plus the B10b
# first-cell test (wb_probe_firstcell.sh). CUDA_ARCH (from vast_run.sh) picks the toolchain:
#   sm_120 (Blackwell)  -> CUDA 12.9 (nvcc + ncu of 12.9) and torch 2.7.1+cu128 (12.5 cannot build sm_120)
#   anything else       -> CUDA 12.5 (laptop: nvcc V12.5.82, ncu 2024.2.1) and torch 2.7.1+cu126
# Runs ON the VM in ~/dmkl (unpacked benchmarks/src, benchmarks/scripts, research/preregistration and
# the laptop's B6 reference binary). Started detached by benchmarks/scripts/vast/vast_run.sh.
# Works as root or as a sudo-capable user: every ncu run goes through $SUDO, because counters need
# admin rights on a native Linux driver.
#   0. record the environment; stop (exit 4) if there is no working GPU driver
#   1. toolchain from NVIDIA's apt repo (nvcc/cudart/cuobjdump/profiler-api + Nsight Compute);
#      torch venv in the background
#   2. build wb_probe for $CUDA_ARCH with the laptop's build.sh; dump its SASS and the SASS of the
#      laptop's B6 binary (wb_probe.binary, e0c82a53...) for a local diff (meaningful for sm_86 only)
#   3. counter permission check: ncu on wb_probe --smoke; stop (exit 3) if ERR_NVGPUCTRPERM
#   4. wb_probe_b6b.sh, wb_probe_b8.sh, wb_probe_firstcell.sh   (scored locally afterwards as well)
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
export PIP_NO_CACHE_DIR=1   # small VM disks
ARCH="${CUDA_ARCH:-sm_86}"
case "$ARCH" in sm_12*) CV=12.9; TIDX=cu128 ;; *) CV=12.5; TIDX=cu126 ;; esac
CUDA="/usr/local/cuda-$CV"; PKV="${CV/./-}"
REF_B6=benchmarks/results/wb_probe_20261105-042441/wb_probe.binary

step "0 environment"
{ date -u; id; uname -a; cat /etc/os-release; nproc; free -g; df -h "$HOME"; nvidia-smi;
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,pstate --format=csv
  grep -i RmProfilingAdminOnly /proc/driver/nvidia/params; ls /usr/local | grep -i cuda; } > pilot_out/environment.txt 2>&1
# nvidia-smi -L exits 0 even when it prints "No devices were found" (B11 attempt 2, D-060),
# so a working GPU is one that nvidia-smi lists as "GPU <n>: ...".
gpu_ok() { nvidia-smi -L 2>/dev/null | grep -q "^GPU [0-9]"; }
# Blackwell GPUs need NVIDIA's open kernel modules; the VM image ships the closed nvidia-driver-575
# (B11 attempt 1 on an RTX 5060 Ti: "requires use of the NVIDIA open kernel modules", D-060).
# Swap in the open variant of the same driver branch and reload the modules. The trigger is "no
# usable GPU although an NVIDIA PCI device is present": the dmesg hint appeared too late in B11
# attempt 3 (the script runs ~2 min after boot), D-060.
nv_pci() { grep -qi "0x10de" /sys/bus/pci/devices/*/vendor 2>/dev/null; }
if ! gpu_ok && nv_pci; then
  MAJ=$(dpkg-query -W -f='${Package}\n' 'nvidia-driver-[0-9]*' 2>/dev/null | sed -n 's/^nvidia-driver-\([0-9]*\)$/\1/p' | head -1)
  [ -n "$MAJ" ] || MAJ=$(sed -n 's/.*Kernel Module *\([0-9]*\)\..*/\1/p' /proc/driver/nvidia/version 2>/dev/null)
  step "NVIDIA PCI device but no usable GPU: installing nvidia-driver-$MAJ-open"
  { $SUDO dmesg | grep -iE "nvrm|nvidia" | tail -20; echo "--- swap"
    "${APT[@]}" update
    # the image holds its nvidia packages at 575.51.03 while nvidia-driver-575-open is 575.57.08, so
    # apt refuses ("held packages", B11 attempt 4): unhold, then move driver + userspace together;
    # fall back to the open kernel module of the installed version only
    HELD=$(apt-mark showhold | grep -E '^(nvidia|libnvidia|xserver-xorg-video-nvidia)'); echo "held: $HELD"
    [ -n "$HELD" ] && $SUDO apt-mark unhold $HELD
    "${APT[@]}" install "nvidia-driver-$MAJ-open"; rc=$?; echo "apt rc=$rc"
    if [ "$rc" != 0 ]; then
      VER=$(dpkg-query -W -f='${Version}' "nvidia-driver-$MAJ")
      "${APT[@]}" install "nvidia-dkms-$MAJ-open=$VER" "nvidia-kernel-source-$MAJ-open=$VER"; echo "apt pinned rc=$?"
    fi
    $SUDO systemctl stop nvidia-persistenced
    for m in nvidia_uvm nvidia_drm nvidia_modeset nvidia; do $SUDO rmmod "$m"; done
    $SUDO modprobe nvidia && $SUDO modprobe nvidia_uvm
    cat /proc/driver/nvidia/version; nvidia-smi; } > pilot_out/driver_open.log 2>&1
  step "after open-module swap: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -1)"
  nvidia-smi >> pilot_out/environment.txt 2>&1
fi
gpu_ok || { step "no working NVIDIA driver"; echo NO_DRIVER > pilot_out/STATUS; exit 4; }
step "user $(id -un), sudo='${SUDO}', arch $ARCH, CUDA $CV, torch index $TIDX"

step "1 toolchain"
# unattended-upgrades can hold the dpkg lock past apt's 600 s timeout right after boot (B12 on the
# 2060S, D-062): stop it before using apt
$SUDO systemctl stop unattended-upgrades apt-daily.service apt-daily-upgrade.service apt-daily.timer \
  apt-daily-upgrade.timer > /dev/null 2>&1
if [ ! -x "$CUDA/bin/nvcc" ] || [ ! -x "$CUDA/bin/ncu" ] || [ ! -f "$CUDA/include/cuda_profiler_api.h" ]; then
  curl -fsSL -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb &&
    $SUDO dpkg -i /tmp/cuda-keyring.deb > pilot_out/apt.log 2>&1
  "${APT[@]}" update >> pilot_out/apt.log 2>&1
  # cuda-profiler-api: wb_probe.cu includes cuda_profiler_api.h (missing in B10 attempt 1, D-057)
  "${APT[@]}" install build-essential python3-venv "cuda-nvcc-$PKV" "cuda-cudart-dev-$PKV" "cuda-cuobjdump-$PKV" \
    "cuda-profiler-api-$PKV" "cuda-nsight-compute-$PKV" >> pilot_out/apt.log 2>&1
  step "apt rc=$?"
fi
export PATH="$CUDA/bin:$PATH"
{ nvcc --version; ncu --version; } > pilot_out/toolchain.txt 2>&1
cat pilot_out/toolchain.txt | tee -a "$LOG"
VENV="$HOME/venv"
( python3 -m venv "$VENV" && "$VENV/bin/pip" install -q --index-url "https://download.pytorch.org/whl/$TIDX" \
    --extra-index-url https://pypi.org/simple torch==2.7.1 triton==3.3.1 numpy==2.2.6 &&
  "$VENV/bin/python" -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.is_available())"
) > pilot_out/torch_install.log 2>&1 &
TORCH_PID=$!

step "2 build"
NVCC="$CUDA/bin/nvcc" CUDA_ARCH="$ARCH" ONLY=wb_probe bash benchmarks/scripts/build.sh > pilot_out/build.txt 2>&1
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
step "4 B10b first-cell"
"${NCURUN[@]}" bash benchmarks/scripts/wb_probe_firstcell.sh > pilot_out/firstcell.txt 2>&1
step "B10b rc=$?"

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
