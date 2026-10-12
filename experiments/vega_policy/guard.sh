#!/bin/bash
# Memory safety for every experiment on this 8 GB Mac.
#   free_pct          system-wide free memory, percent
#   wait_for_memory   block until at least $MIN_FREE % is free before starting a run
#   watchdog          kill any experiment process above $MAX_MB resident or when free drops under $KILL_FREE %
MIN_FREE=${MIN_FREE:-40}; KILL_FREE=${KILL_FREE:-20}; MAX_MB=${MAX_MB:-2500}
free_pct(){ memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'; }
wait_for_memory(){
  while [ "$(free_pct)" -lt "$MIN_FREE" ]; do echo "GUARD waiting: only $(free_pct)% free"; sleep 30; done
}
watchdog(){
  while true; do
    for p in $(pgrep -f "train.py|run.py|extract.py|vtrain.py"); do
      mb=$(( $(ps -o rss= -p $p 2>/dev/null || echo 0) / 1024 ))
      f=$(free_pct)
      if [ "$mb" -gt "$MAX_MB" ] || [ "$f" -lt "$KILL_FREE" ]; then
        echo "GUARD killed pid $p (rss ${mb}MB, free ${f}%)"; kill -9 $p
      fi
    done
    sleep 3
  done
}
# ./guard.sh watch  -> run the watchdog in the foreground (the pipeline backgrounds it)
if [ "$1" = "watch" ]; then watchdog; fi
