#!/bin/bash
# Run dingo_pipe on every .ini file in a folder (default: ./inis).
#
# Usage:
#   ./run_dingo_pipe_batch.sh [inis_dir] [logs_dir]
#
# For each .ini file the `outdir` entry is read. If that directory already
# contains a .pdf file (searched recursively), the inference is considered
# done and is skipped. Otherwise the whole outdir is deleted and recreated
# by dingo_pipe, and the job is run normally.
#
# Since the .ini files have `local = True`, dingo_pipe runs each job
# in-process (no scheduler submission). Jobs are run sequentially by
# default; set PARALLEL=1 to launch them in the background instead
# (only do this if your GPU/CPU resources can handle running them
# concurrently).

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

set -uo pipefail

INIS_DIR="${1:-inis}"
LOGS_DIR="${2:-logs}"
PARALLEL="${PARALLEL:-0}"

mkdir -p "$LOGS_DIR"

if ! compgen -G "$INIS_DIR"/*.ini > /dev/null; then
    echo "No .ini files found in $INIS_DIR" >&2
    exit 1
fi

# Extract the value of `outdir` from an .ini file (ignores comments/quotes).
get_outdir() {
    local val
    val=$(grep -E '^[[:space:]]*outdir[[:space:]]*=' "$1" | head -n 1 \
          | sed -E 's/^[^=]*=[[:space:]]*//; s/[[:space:]]*#.*$//; s/^["'\'']//; s/["'\'']?[[:space:]]*$//')
    # expand a leading ~
    echo "${val/#\~/$HOME}"
}

pids=()
failed=()
skipped=()

for ini_file in "$INIS_DIR"/*.ini; do
    event=$(basename "$ini_file" .ini)
    log_file="$LOGS_DIR/${event}.log"

    outdir=$(get_outdir "$ini_file")

    if [ -z "$outdir" ]; then
        echo "  FAILED: $event (no 'outdir' found in $ini_file)" >&2
        failed+=("$event")
        continue
    fi

    # Skip if a .pdf already exists in the outdir
    if [ -d "$outdir" ] && [ -n "$(find "$outdir" -type f -name '*.pdf' -print -quit)" ]; then
        echo "Skipping $event: .pdf already exists in $outdir"
        skipped+=("$event")
        continue
    fi

    # Otherwise wipe the outdir (with a safety guard) and run normally
    if [ -e "$outdir" ]; then
        case "$(realpath -m "$outdir")" in
            /|"$HOME"|"$(pwd)")
                echo "  FAILED: $event (refusing to delete unsafe outdir: $outdir)" >&2
                failed+=("$event")
                continue
                ;;
        esac
        echo "Removing existing outdir (no .pdf found): $outdir"
        rm -rf "$outdir"
    fi

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
echo "Skipped ${#skipped[@]} run(s) with existing .pdf: ${skipped[*]:-none}"
if [ "${#failed[@]}" -eq 0 ]; then
    echo "All dingo_pipe runs completed successfully."
else
    echo "Completed with ${#failed[@]} failure(s): ${failed[*]}"
    exit 1
fi