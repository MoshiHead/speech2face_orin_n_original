#!/usr/bin/env bash
# run_watchdog_x86.sh -- keep ./run_x86.sh up: restart it if it exits or hits a CUDA device-side assert.
# Log: run.log (current), crashlogs/ (saved logs of crashed runs).   Stop: pkill -f run_watchdog_x86.sh; pkill -f speech2face/src
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"; mkdir -p crashlogs
exec 9>"$HERE/.watchdog_x86.lock"; flock -n 9 || { echo "watchdog already running"; exit 1; }
while true; do
  ./run_x86.sh > run.log 2>&1 &
  PID=$!; echo "[$(date +%F_%T)] started pid $PID" >> crashlogs/watchdog.log
  sleep 60
  while kill -0 $PID 2>/dev/null; do
    if grep -q "device-side assert" run.log; then
      cp run.log crashlogs/crash_$(date +%m%d_%H%M%S).log
      echo "[$(date +%F_%T)] device-side assert -> restarting" >> crashlogs/watchdog.log
      kill -9 $PID; break
    fi
    sleep 5
  done
  wait $PID 2>/dev/null; sleep 5
done
