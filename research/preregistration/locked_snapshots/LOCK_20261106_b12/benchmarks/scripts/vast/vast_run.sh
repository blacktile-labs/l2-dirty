#!/usr/bin/env bash
# vast_run.sh -- rent one vast.ai instance, run a remote protocol, fetch results, ALWAYS destroy.
#
#   bash benchmarks/scripts/vast/vast_run.sh <offer_id> <image> <cuda_arch> <remote_script> <remote_archive> <tag>
#   e.g. bash benchmarks/scripts/vast/vast_run.sh 45600364 nvidia/cuda:12.2.2-devel-ubuntu22.04 sm_86 \
#             benchmarks/scripts/remote_v3.sh /tmp/dmkl_v3.tgz rtx3060
#
# Generalized from the orchestrators actually used on 2026-11-04 (kept verbatim, for provenance, in
# benchmarks/results/orchestrator/). First exercised by the B10 VM pilot (2026-11-05); check the log.
#
# Requirements: vastai CLI (VASTAI_CLI, default "vastai"; e.g. VASTAI_CLI="uv tool run --quiet vastai"),
# an API key in <repo>/.env as VASTAI_API_KEY (never printed), an SSH key registered with vast.ai
# (default ~/.ssh/vastai, override with VAST_SSH_KEY).
# Optional environment:
#   DISK (GB, default 20)  SSH_WAIT_S (default 600)  RUN_WAIT_S (default 1500)
#   UPLOAD_EXTRA  extra repo paths to package next to benchmarks/src and benchmarks/scripts
#   HARD_CAP_S    wall-clock cap for the whole rental: when it expires the instance is destroyed
#   SSH_USERS     users tried for SSH, in order (default "root")
#   ATTACH_IID    take over this existing instance instead of creating one (still destroyed on exit)
# Behaviour:
#   * never uses a vast.ai template or onstart script (a public "Ubuntu 22.04 VM" template was found
#     to run `curl ... | bash` on start, D-055): the image is given explicitly
#   * waits <= SSH_WAIT_S for SSH, retries the upload 3x
#   * runs the remote script DETACHED (nohup) and polls for completion (<= RUN_WAIT_S), so a dropped
#     SSH session does not lose the run
#   * downloads <remote_archive> into $WORK_DIR (default <repo>/.vast_work/<tag>, gitignored)
#   * trap EXIT/INT/TERM destroys the instance and verifies it is gone (3 attempts)
# Every rental must be approved by the PI first (CLAUDE.md).
set -uo pipefail
[ $# -eq 6 ] || { sed -n '2,6p' "$0"; exit 64; }
OFFER="$1"; IMAGE="$2"; ARCH="$3"; RSCRIPT="$4"; RARCHIVE="$5"; TAG="$6"
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORK="${WORK_DIR:-$PROJ/.vast_work/$TAG}"; rm -rf "$WORK"; mkdir -p "$WORK" "$PROJ/.vast_work"
KEY="${VAST_SSH_KEY:-$HOME/.ssh/vastai}"
DISK="${DISK:-20}"; SSH_WAIT_S="${SSH_WAIT_S:-600}"; RUN_WAIT_S="${RUN_WAIT_S:-1500}"
[ -f "$PROJ/.env" ] || { echo "missing $PROJ/.env (VASTAI_API_KEY)"; exit 65; }
[ -f "$PROJ/$RSCRIPT" ] || { echo "missing remote script $RSCRIPT"; exit 66; }
set -a; . "$PROJ/.env" >/dev/null 2>&1; set +a
read -r -a CLI <<< "${VASTAI_CLI:-vastai}"
VA=("${CLI[@]}" --api-key "$VASTAI_API_KEY")
SSHO=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$PROJ/.vast_work/known_hosts"
      -o ConnectTimeout=15 -o BatchMode=yes -o ServerAliveInterval=30)
log() { echo "[$TAG $(date -u +%H:%M:%S)] $*" | tee -a "$WORK/orchestrator.log"; }

# package exactly the code under test
(cd "$PROJ" && find benchmarks -name __pycache__ -prune -exec rm -rf {} + &&
 tar czf "$WORK/upload.tgz" benchmarks/src benchmarks/scripts ${UPLOAD_EXTRA:-})
log "package sha256 $(sha256sum "$WORK/upload.tgz" | cut -c1-16), remote script sha256 $(sha256sum "$PROJ/$RSCRIPT" | cut -c1-16)"

IID=""
cleanup() {
  if [ -n "$IID" ]; then
    log "destroying instance $IID"
    left=""
    for k in 1 2 3; do
      timeout 60 "${VA[@]}" destroy instance "$IID" -y --raw >/dev/null 2>&1
      left=$(timeout 60 "${VA[@]}" show instances --raw 2>/dev/null |
             python3 -c "import json,sys; print(sum(1 for i in json.load(sys.stdin) if i.get('id')==$IID))" 2>/dev/null)
      [ "$left" = "0" ] && { log "destroyed (verified)"; break; }
      sleep 10
    done
    [ "$left" = "0" ] || log "WARNING: destroy NOT verified for $IID -- check the vast.ai console now"
    IID=""
  fi
  [ -n "${WATCH:-}" ] && kill "$WATCH" 2>/dev/null
}
trap cleanup EXIT
trap 'log "signal: stopping"; exit 130' INT TERM
if [ -n "${HARD_CAP_S:-}" ]; then
  ( sleep "$HARD_CAP_S"; echo "[$TAG] HARD_CAP_S=$HARD_CAP_S reached" >> "$WORK/orchestrator.log"; kill -TERM $$ ) &
  WATCH=$!
fi

if [ -n "${ATTACH_IID:-}" ]; then
  # take over an instance created by an earlier vast_run.sh (e.g. to wait longer for a slow VM boot);
  # it is destroyed on exit exactly like a created one
  IID="$ATTACH_IID"
  log "attached to existing instance $IID (no create; offer $OFFER, image $IMAGE)"
else
  timeout 120 "${VA[@]}" create instance "$OFFER" --image "$IMAGE" --disk "$DISK" --ssh --direct --raw \
    > "$WORK/create.json" 2> "$WORK/create.err"
  IID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('new_contract') or '')" "$WORK/create.json" 2>/dev/null)
  [ -n "$IID" ] || { log "create failed: $(head -c 300 "$WORK/create.json" "$WORK/create.err" | tr '\n' ' ')"; exit 1; }
  log "created instance $IID (offer $OFFER, image $IMAGE, disk $DISK GB)"
fi

HOST=""; PORT=""; RUSER=""; end=$((SECONDS + SSH_WAIT_S))
while [ $SECONDS -lt $end ]; do
  info=$(timeout 60 "${VA[@]}" show instance "$IID" --raw 2>/dev/null | python3 -c "
import json,sys
i=json.load(sys.stdin); p=(i.get('ports') or {}).get('22/tcp') or [{}]
print(i.get('actual_status'), i.get('public_ipaddr') or '-', p[0].get('HostPort') or '-', i.get('ssh_host') or '-', i.get('ssh_port') or '-', i.get('dph_total') or '-')" 2>/dev/null)
  set -- $info
  [ -n "${6:-}" ] && [ "${6:-}" != - ] && [ -z "${DPH_LOGGED:-}" ] && { log "price dph_total = \$$6/h"; DPH_LOGGED=1; }
  if [ "${1:-}" = running ]; then
    for hp in "$2 $3" "$4 $5"; do set -- $hp; [ "$1" = - ] && continue
      for u in ${SSH_USERS:-root}; do
        if timeout 25 ssh "${SSHO[@]}" -p "$2" "$u@$1" 'echo OK' 2>/dev/null | grep -q OK; then HOST=$1; PORT=$2; RUSER=$u; break 3; fi
      done
    done
  fi
  sleep 20
done
[ -n "$HOST" ] || { log "SSH not reachable within $SSH_WAIT_S s"; exit 2; }
log "ssh ok ($RUSER@$HOST:$PORT)"
RH="$RUSER@$HOST"
RHOME=$(timeout 30 ssh "${SSHO[@]}" -p "$PORT" "$RH" 'echo $HOME' 2>/dev/null)

up=0
for a in 1 2 3; do
  if timeout 300 scp "${SSHO[@]}" -P "$PORT" "$WORK/upload.tgz" "$RH:$RHOME/dmkl_upload.tgz" >>"$WORK/scp.log" 2>&1; then up=1; break; fi
  log "upload attempt $a failed"; sleep 15
done
[ $up = 1 ] || { log "upload failed"; exit 3; }

timeout 120 ssh "${SSHO[@]}" -p "$PORT" "$RH" "mkdir -p ~/dmkl && cd ~/dmkl && tar xzf ~/dmkl_upload.tgz && sha256sum $RSCRIPT | cut -c1-16 > ~/SHA && (nohup bash -c 'CUDA_ARCH=$ARCH bash $RSCRIPT > ~/run.log 2>&1; echo rc=\$? > ~/DONE' > /dev/null 2>&1 &)" >> "$WORK/remote_stdout.txt" 2>&1
log "started detached run of $RSCRIPT"
done_ok=0; pend=$((SECONDS + RUN_WAIT_S)); last=""
while [ $SECONDS -lt $pend ]; do
  st=$(timeout 40 ssh "${SSHO[@]}" -p "$PORT" "$RH" "cat ~/DONE 2>/dev/null" 2>/dev/null | grep -E "^rc=")
  [ -n "$st" ] && { done_ok=1; break; }
  cur=$(timeout 40 ssh "${SSHO[@]}" -p "$PORT" "$RH" "tail -1 ~/dmkl/pilot_out/remote.log 2>/dev/null" 2>/dev/null)
  [ -n "$cur" ] && [ "$cur" != "$last" ] && { log "remote: $cur"; last="$cur"; }
  sleep 20
done
timeout 60 ssh "${SSHO[@]}" -p "$PORT" "$RH" "cat ~/SHA ~/run.log ~/DONE" >> "$WORK/remote_stdout.txt" 2>&1
log "remote finished (done=$done_ok): $(grep -E '^rc=' "$WORK/remote_stdout.txt" | tail -1)"
[ $done_ok = 1 ] || timeout 120 ssh "${SSHO[@]}" -p "$PORT" "$RH" "cd ~/dmkl && tar czf $RARCHIVE pilot_out benchmarks/results 2>/dev/null" >/dev/null 2>&1
if timeout 300 scp "${SSHO[@]}" -P "$PORT" "$RH:$RARCHIVE" "$WORK/" >>"$WORK/scp.log" 2>&1; then
  log "downloaded $(basename "$RARCHIVE") -> $WORK"
else
  log "download failed"
fi
exit 0
