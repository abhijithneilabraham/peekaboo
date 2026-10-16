#!/bin/bash
# Unattended run on a RunPod pod: run every step, push the results to GitHub every 10 minutes, and terminate the
# pod when everything is done or when MAX_HOURS is reached (so it never bills past the run).
#   setsid nohup bash autopilot.sh > output/autopilot.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/root/hf}
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
REPO=$(git rev-parse --show-toplevel)
DEADLINE=$(( $(date +%s) + ${MAX_HOURS:-5} * 3600 ))
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
push() {   # results only: JSON, markdown and logs (.gitignore keeps feature caches and dataset rows out)
  git -C "$REPO" add -A experiments/peekaboo_remaining/output
  git -C "$REPO" commit -q -m "$1" && log "committed: $1"
  git -C "$REPO" pull -q --rebase --autostash && git -C "$REPO" push -q && log "pushed"
}
terminate() {
  log "terminating pod $RUNPOD_POD_ID"
  runpodctl remove pod "$RUNPOD_POD_ID" 2>&1 | tail -1
  curl -s -X POST https://api.runpod.io/graphql -H 'Content-Type: application/json' -H "Authorization: Bearer $RUNPOD_API_KEY" \
       -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}"
}
( while true; do sleep 600; push "RunPod: Peekaboo results so far ($(date -u +%H:%M) UTC)"; done ) & PUSHER=$!
for attempt in 1 2 3; do
  git -C "$REPO" pull -q --rebase --autostash && log "code at $(git -C "$REPO" log --oneline -1)"   # fixes pushed meanwhile
  left=$(( DEADLINE - $(date +%s) )); [ $left -le 60 ] && break
  log "run.py, attempt $attempt, ${left}s left"
  timeout $left python run.py
  python run.py --check > /dev/null; code=$?
  log "check exit $code"; [ $code -eq 0 ] && break
done
python table.py > /dev/null 2>&1
kill $PUSHER 2>/dev/null
push "RunPod: final Peekaboo results"
terminate
