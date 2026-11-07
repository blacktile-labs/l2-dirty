#!/usr/bin/env bash
# usage: rent_run_destroy.sh <offer_id> <image> <cuda_arch> <tag>
# Creates an instance, waits <=10 min for SSH, uploads the package, runs remote_v3.sh,
# downloads results, and ALWAYS destroys the instance (trap EXIT). Never prints secrets.
set -uo pipefail
OFFER="$1"; IMAGE="$2"; ARCH="$3"; TAG="$4"
PROJ=/home/rungrod/data-move-kernel-lab
SP="${SP:-/tmp/scratch}"   # scratch directory (original session path removed)
OUT="$SP/remote_v3_$TAG"; rm -rf "$OUT"; mkdir -p "$OUT"
set -a; . "$PROJ/.env" >/dev/null 2>&1; set +a
VA=(vastai --api-key "$VASTAI_API_KEY")
SSHO=(-i "$HOME/.ssh/vastai" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$SP/known_hosts_vast" -o ConnectTimeout=15 -o BatchMode=yes -o ServerAliveInterval=30)
log() { echo "[$TAG $(date -u +%H:%M:%S)] $*"; }
IID=""
cleanup() {
  if [ -n "$IID" ]; then
    log "destroying instance $IID"
    for k in 1 2 3; do
      timeout 60 "${VA[@]}" destroy instance "$IID" -y --raw >/dev/null 2>&1
      left=$(timeout 60 "${VA[@]}" show instances --raw 2>/dev/null | python3 -c "import json,sys; print(sum(1 for i in json.load(sys.stdin) if i.get('id')==$IID))" 2>/dev/null)
      [ "$left" = "0" ] && { log "destroyed (verified)"; break; }
      sleep 10
    done
    [ "$left" = "0" ] || log "WARNING: destroy not verified for $IID"
  fi
}
trap cleanup EXIT
IID=$(timeout 120 "${VA[@]}" create instance "$OFFER" --image "$IMAGE" --disk 20 --ssh --direct --raw 2>/dev/null | python3 -c "import json,sys; r=json.load(sys.stdin); print(r.get('new_contract') or '')" 2>/dev/null)
[ -n "$IID" ] || { log "create failed"; exit 1; }
log "created instance $IID"
HOST=""; PORT=""; end=$((SECONDS+600))
while [ $SECONDS -lt $end ]; do
  info=$(timeout 60 "${VA[@]}" show instance "$IID" --raw 2>/dev/null | python3 -c "
import json,sys
i=json.load(sys.stdin); p=(i.get('ports') or {}).get('22/tcp') or [{}]
print(i.get('actual_status'), i.get('public_ipaddr') or '-', p[0].get('HostPort') or '-', i.get('ssh_host') or '-', i.get('ssh_port') or '-')" 2>/dev/null)
  set -- $info
  if [ "${1:-}" = running ]; then
    for hp in "$2 $3" "$4 $5"; do set -- $hp; [ "$1" = - ] && continue
      if timeout 25 ssh "${SSHO[@]}" -p "$2" "root@$1" 'echo OK' 2>/dev/null | grep -q OK; then HOST=$1; PORT=$2; break 2; fi
    done
  fi
  sleep 20
done
[ -n "$HOST" ] || { log "SSH not reachable within 10 min"; exit 2; }
log "ssh ok $HOST:$PORT"
up=0
for a in 1 2 3; do
  if timeout 180 scp "${SSHO[@]}" -P "$PORT" "$SP/dmkl_upload6.tgz" "root@$HOST:/root/dmkl_upload.tgz" >>"$OUT/scp_upload.log" 2>&1; then up=1; break; fi
  log "upload attempt $a failed (rc=$?); see $OUT/scp_upload.log"; sleep 15
done
[ $up = 1 ] || { log "upload failed"; exit 3; }
# Run detached on the host (survives SSH drops), then poll for completion (<= 25 min).
timeout 120 ssh "${SSHO[@]}" -p "$PORT" "root@$HOST" "mkdir -p /root/dmkl && cd /root/dmkl && tar xzf /root/dmkl_upload.tgz && sha256sum benchmarks/scripts/remote_v3.sh | cut -c1-16 > /root/SHA && (nohup bash -c 'CUDA_ARCH=$ARCH bash benchmarks/scripts/remote_v3.sh > /root/run.log 2>&1; echo rc=\$? > /root/DONE' > /dev/null 2>&1 &)" >> "$OUT/remote_stdout.txt" 2>&1
log "started detached run"
done_ok=0; pend=$((SECONDS+1500))
while [ $SECONDS -lt $pend ]; do
  st=$(timeout 40 ssh "${SSHO[@]}" -p "$PORT" "root@$HOST" "cat /root/DONE 2>/dev/null; tail -1 /root/run.log 2>/dev/null" 2>/dev/null | grep -vE "Welcome|Have fun")
  if echo "$st" | grep -q "^rc="; then done_ok=1; break; fi
  sleep 20
done
timeout 60 ssh "${SSHO[@]}" -p "$PORT" "root@$HOST" "cat /root/SHA /root/run.log /root/DONE" >> "$OUT/remote_stdout.txt" 2>&1
log "remote finished (done=$done_ok): $(grep -E '^rc=' "$OUT/remote_stdout.txt" | tail -1)"
timeout 180 scp "${SSHO[@]}" -P "$PORT" "root@$HOST:/tmp/dmkl_v3.tgz" "$OUT/" >/dev/null 2>&1 && log "downloaded $(stat -c %s "$OUT/dmkl_v3.tgz") bytes" || log "download failed"
exit 0
