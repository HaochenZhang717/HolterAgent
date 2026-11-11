#!/usr/bin/env bash
# Run cloop rounds one after another for a track, unattended (2026-10-17).
#   RSI_TRACK=q27 bash RSI/cloop_driver.sh Q1 Q2 OPEN Q3 ...
# Waits until the track has a base on its board. A round whose evaluation stops early (IncompleteRun, e.g. after a
# preempted GPU pod) is resumed with `cloop.py resume <id>` up to twice. Every step goes to <cloop dir>/driver.log.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
DIR="RSI/cloop${RSI_TRACK:+_$RSI_TRACK}"
LOG="$DIR/driver.log"
log() { echo "[driver] $(date '+%F %T') $*" | tee -a "$LOG"; }
until [ -f "$DIR/board.jsonl" ] && grep -q '"status": "base"' "$DIR/board.jsonl"; do sleep 60; done
log "base found; focuses: $*"
for focus in "$@"; do
    if ! PYTHONPATH=RSI $PY -m rsi.spend --check >> "$LOG" 2>&1; then  # paid tracks: stop at the budget
        log "budget reached; stopping before focus $focus"
        break
    fi
    log "round with focus $focus starts"
    $PY RSI/cloop.py round --focus "$focus" >> "$DIR/driver_rounds.out" 2>&1
    rc=$?
    last=$(ls -d "$DIR"/rounds/round_* 2>/dev/null | sort | tail -1)
    hid=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['entry']['id'])" "$last/meta.json" 2>/dev/null)
    for attempt in 1 2; do
        [ -z "$hid" ] && break
        grep -q "\"id\": \"$hid\"" "$DIR/board.jsonl" && break
        log "round $focus ($hid) has no board entry (exit $rc); resume attempt $attempt"
        $PY RSI/cloop.py resume "$hid" >> "$DIR/driver_rounds.out" 2>&1
        rc=$?
    done
    log "round with focus $focus ended (exit $rc): $(tail -1 "$DIR/board.jsonl" | cut -c1-300)"
done
log "all focuses done"
