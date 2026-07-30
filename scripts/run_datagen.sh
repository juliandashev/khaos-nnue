#!/usr/bin/env bash
# Launch parallel self-play data generation, one process per core.
#
# The engine's datagen is single-threaded on purpose: each process gets its own
# transposition table, so games never contaminate each other's scores. Scaling
# is therefore by process, not by thread.
#
#   ./scripts/run_datagen.sh <engine> <out-dir> <games-per-shard> [nodes] [jobs]
#
# Example, 32 shards of 20k games at 5k nodes:
#   ./scripts/run_datagen.sh ../chess-engine/bin/KhaosChess data 20000 5000 32
#
# Shards are written as <out-dir>/shard_NN.txt and can be concatenated or passed
# to prepare.py as a glob. Safe to interrupt: datagen appends and flushes
# periodically, so whatever finished is usable.

set -euo pipefail

ENGINE="${1:?usage: run_datagen.sh <engine> <out-dir> <games-per-shard> [nodes] [jobs]}"
OUT_DIR="${2:?missing out-dir}"
GAMES="${3:?missing games-per-shard}"
NODES="${4:-5000}"
JOBS="${5:-$(nproc)}"

if [[ ! -x "$ENGINE" ]]; then
    echo "error: '$ENGINE' is not executable" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "engine:  $ENGINE"
echo "shards:  $JOBS x $GAMES games at $NODES nodes/move"
echo "output:  $OUT_DIR/shard_*.txt"
echo

pids=()
for ((i = 0; i < JOBS; i++)); do
    shard=$(printf "%s/shard_%02d.txt" "$OUT_DIR" "$i")
    log=$(printf "%s/shard_%02d.log" "$OUT_DIR" "$i")

    # Each shard needs a distinct seed or every process plays identical games.
    seed=$((1000 + i))

    printf 'datagen games %s nodes %s seed %s randply 8 out %s report 200\nquit\n' \
        "$GAMES" "$NODES" "$seed" "$shard" \
        | "$ENGINE" > /dev/null 2> "$log" &
    pids+=($!)
done

echo "started ${#pids[@]} workers; progress is in $OUT_DIR/shard_*.log"
echo "waiting..."

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done

echo
if (( failed > 0 )); then
    echo "warning: $failed worker(s) exited non-zero; check the logs" >&2
fi

echo "positions generated:"
wc -l "$OUT_DIR"/shard_*.txt | tail -1
