#!/bin/bash
# Unattended run of the v2 grid on a fresh RunPod pod, pushing results to GitHub every 10 minutes and terminating the
# pod when done (or after MAX_HOURS). Expects: the repo at /workspace/peekaboo, a push-capable deploy key in ~/.ssh,
# the v1 items_*.json for public/jevbench/decisionbench/typed in output_v2/, and RUNPOD_API_KEY + RUNPOD_POD_ID in
# /workspace/pod.env. Run:  nohup bash autopilot.sh > /workspace/autopilot.log 2>&1 &
set -u
cd /workspace/peekaboo
source /workspace/pod.env
MAX_HOURS=${MAX_HOURS:-7}
OUT=/workspace/peekaboo/experiments/peekaboo_remaining/output_v2
export VEGA_OUT=$OUT HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
git config user.name abhijithneilabraham; git config user.email abhijithneilabrahampk@gmail.com
git config pull.rebase true

push() {
  cd /workspace/peekaboo
  git add -A experiments/peekaboo_remaining/output_v2 >/dev/null 2>&1
  git commit -qm "RunPod v2: results so far ($(date -u +%H:%M) UTC)" >/dev/null 2>&1 || true
  for i in 1 2 3; do git pull -q --rebase origin main && git push -q origin main && break; sleep 20; done
}

terminate() {
  push
  curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H 'Content-Type: application/json' \
       -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}"
}

cd experiments/peekaboo_remaining
mkdir -p $OUT
# the grid, detached; the pusher and the watchdog below run beside it
nohup python run.py --grid v2 --out $OUT --jobs 8 >> $OUT/run.log 2>&1 &
RUN=$!
START=$(date +%s)
while kill -0 $RUN 2>/dev/null; do
  sleep 600
  push
  if (( $(date +%s) - START > MAX_HOURS * 3600 )); then
    echo "MAX_HOURS reached; stopping" >> $OUT/run.log
    pkill -f "run.py --grid v2" ; sleep 5; pkill -f adapt.py; pkill -f laya_peekaboo.py; pkill -f suite_cache.py
    break
  fi
done
sleep 5
echo "autopilot: run.py finished at $(date -u)" >> $OUT/run.log
terminate
