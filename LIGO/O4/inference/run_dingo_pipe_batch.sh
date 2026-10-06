#!/bin/bash
# Run dingo_pipe on every .ini file in a folder (default: ./inis).
#
# Usage:
#   ./run_dingo_pipe_batch.sh [inis_dir] [logs_dir]
#
# Since the .ini files have `local = True`, dingo_pipe runs each job
# in-process (no scheduler submission). Jobs are run sequentially by
# default; set PARALLEL=1 to launch them in the background instead
# (only do this if your GPU/CPU resources can handle running them
# concurrently).

set -uo pipefail

INIS_DIR="${1:-inis}"
LOGS_DIR="${2:-logs}"
PARALLEL="${PARALLEL:-0}"

mkdir -p "$LOGS_DIR"

if ! compgen -G "$INIS_DIR"/*.ini > /dev/null; then
    echo "No .ini files found in $INIS_DIR" >&2
    exit 1
fi

pids=()
failed=()

for ini_file in "$INIS_DIR"/*.ini; do
    event=$(basename "$ini_file" .ini)
    log_file="$LOGS_DIR/${event}.log"

    echo "Running dingo_pipe on $ini_file (log: $log_file)"

    if [ "$PARALLEL" = "1" ]; then
        PYTHONUNBUFFERED=1 dingo_pipe "$ini_file" > "$log_file" 2>&1 &
        pids+=("$!:$event")
    else
        if ! PYTHONUNBUFFERED=1 dingo_pipe "$ini_file" > "$log_file" 2>&1; then
            echo "  FAILED: $event (see $log_file)" >&2
            failed+=("$event")
        fi
    fi
done

if [ "$PARALLEL" = "1" ]; then
    for entry in "${pids[@]}"; do
        pid="${entry%%:*}"
        event="${entry#*:}"
        if ! wait "$pid"; then
            echo "  FAILED: $event (see $LOGS_DIR/${event}.log)" >&2
            failed+=("$event")
        fi
    done
fi

echo ""
if [ "${#failed[@]}" -eq 0 ]; then
    echo "All dingo_pipe runs completed successfully."
else
    echo "Completed with ${#failed[@]} failure(s): ${failed[*]}"
    exit 1
fi