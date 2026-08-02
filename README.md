# khaos-nnue

NNUE training pipeline for [KhaosChess](https://github.com/FMI-OpenSource-Lab/chess-engine).

The engine repo holds the inference side (accumulator, quantized forward pass,
`.nnue` loader) and the self-play data generator. This repo holds the trainer,
which needs PyTorch and so lives separately to keep the engine dependency-free.

**If you were handed this to run the training: start at [Runbook](#runbook).**

---

## What you need

| | |
|---|---|
| **CPU** | The more cores the better - data generation scales linearly by process. This is the long pole. |
| **GPU** | Any CUDA card. The net is small (768×256×32×1); an hour of GPU time is plenty per run. Trains on CPU too, just slower. |
| **Disk** | ~100 bytes per position of text, ~72 bytes packed. 100M positions ≈ 10 GB text + 7 GB packed. |
| **Python** | 3.9+ with PyTorch and numpy. |

Rough shape of the work: **data generation is days, training is hours.**

---

## Runbook

### 0. Build the engine

```bash
git clone https://github.com/FMI-OpenSource-Lab/chess-engine
cd chess-engine
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
cd ../..
```

Confirm it works and that the NNUE plumbing is sound:

```bash
cd chess-engine && ./bin/tests/nnue_tests && cd ..
```

All 7 tests must pass. They check that the incrementally updated accumulator
matches a from-scratch rebuild over random games - if that fails, stop, because
every net you train afterwards will be evaluated with a corrupted accumulator.

### 1. Set up this repo

```bash
git clone <this repo> khaos-nnue
cd khaos-nnue
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Sanity-check the format contract between Python and C++ before spending days on
data - this needs no torch:

```bash
python3 scripts/make_test_net.py nets/test.nnue --seed 1
python3 scripts/check_engine.py ../chess-engine/bin/KhaosChess nets/test.nnue
```

Expected: `all 8 compared positions agree exactly (4 skipped: ...)`. This means
the engine and the trainer compute byte-identical evaluations from the same
weights. **If it disagrees, nothing downstream is trustworthy** - the feature
indexing or quantization has drifted and a trained net will play badly for
reasons no amount of training fixes.

### 2. Generate data

```bash
./scripts/run_datagen.sh ../chess-engine/bin/KhaosChess data 20000 5000 $(nproc)
```

Arguments: `<engine> <out-dir> <games-per-shard> <nodes-per-move> <parallel-jobs>`.

One process per core, each writing `data/shard_NN.txt`. Progress lands in
`data/shard_NN.log`. Interrupting is safe - datagen appends and flushes, so
whatever finished is usable.

**How much?** The `768 → 256 → 32 → 1` net has more parameters than a flat
single-layer one, so it wants more data to fill: aim for **100-200M positions**
rather than the ~50M a flat net gets by on. At 5000 nodes/move a core produces
very roughly 30-60k positions/hour, so 32 cores gets you ~100M in two to three
days. Start a small run (say `2000` games/shard), walk the whole pipeline to a
working net, *then* commit to the big run.

Node count is the quality/quantity trade-off. 5000 is a reasonable start. Going
below ~2000 makes the labels too noisy to be worth much.

Each retained position is one line:

```
<fen> | <score> | <result>
```

`score` is the search score **in engine units, not centipawns** - a pawn is
about 410 here. `result` is the game result, both from White's point of view.
Positions in check, or where the best move is a capture or promotion, are
filtered out during generation: a static evaluation should not be trained to
predict tactics its own quiescence search exists to resolve.

### 3. Pack it

```bash
python3 prepare.py 'data/shard_*.txt' -o data/train.bin --shuffle --jobs $(nproc)
```

Parses once into fixed 72-byte records so training can memory-map instead of
re-parsing FENs every epoch. `--shuffle` matters: consecutive positions come
from the same game and are highly correlated.

`--shuffle` holds everything in RAM. If the set is too big for that, drop the
flag and shuffle the text shards first (`sort -R`, or `shuf`) instead.

### 4. Train

```bash
./venv/bin/python train.py data/train.bin --epochs 30 --out nets/run1
```

Watch the validation loss. It should fall steadily; if it flattens after a few
epochs you need more data rather than more epochs.

A `.nnue` is exported on every improvement, so `nets/run1/best.nnue` is always
loadable while training continues.

Knobs worth touching:

- `--blend` (default 0.7) - weight on the search score vs the game result.
  1.0 trains purely on what the search thought, 0.0 purely on how games ended.
  The score is dense and low-variance but inherits the search's blind spots; the
  result is unbiased but very noisy per position. If the net plateaus, try 0.5.
- `--batch` (default 8192), `--lr` (default 1e-3).
- `--resume nets/run1/epoch_12.pt` to continue an interrupted run.

### 5. Verify the exported net

```bash
python3 scripts/check_engine.py ../chess-engine/bin/KhaosChess nets/run1/best.nnue
```

Same exact-agreement check as step 1, now on a real net. Always run this before
judging a net's strength - it separates "the net is weak" from "the net is being
evaluated wrongly."

### 6. Play it

```bash
cd ../chess-engine
printf 'setoption name EvalFile value ../khaos-nnue/nets/run1/best.nnue\nposition startpos\ngo depth 12\nquit\n' | ./bin/KhaosChess
```

Then match it against the hand-crafted evaluation. The net is the *only*
difference, so this is a clean A/B:

```bash
# NNUE build vs the same binary with the net switched off
fastchess \
  -engine cmd=./bin/KhaosChess name=nnue \
      option.EvalFile=../khaos-nnue/nets/run1/best.nnue \
  -engine cmd=./bin/KhaosChess name=hce option.UseNNUE=false \
  -each tc=8+0.08 -rounds 400 -repeat -concurrency 8 \
  -openings file=8moves_v3.pgn format=pgn order=random
```

**Expect the first net to lose.** A net trained on a few million positions is
worse than a Texel-tuned hand-crafted evaluation. It takes real data volume
before NNUE overtakes a well-tuned HCE. Judge progress by whether the gap
closes as you add data, not by the first result.

---

## Engine-side UCI additions

| Command | Effect |
|---|---|
| `setoption name EvalFile value <path>` | Load a net. Empty or `<empty>` unloads and reverts to the hand-crafted evaluation. |
| `setoption name UseNNUE value false` | Keep the net loaded but ignore it. Useful for A/B matches from one binary. |
| `eval` | Prints `static eval: <n> (nnue\|hce\|endgame)` - the number the search actually uses, and which path produced it. |
| `datagen games N nodes N seed N randply N maxply N out PATH report N` | Self-play generation. Blocks until done. |

With no net loaded the engine behaves exactly as before, so this is all
opt-in.

---

## Troubleshooting

**Training aborts immediately with `TypeError: can't assign a numpy.float32 to a
torch.FloatTensor`.** Recent torch refuses a numpy-2 scalar written straight into
a tensor element. Fixed in `dataset.py` (the result target is coerced to a Python
float); if you hit this you are on an older checkout - pull.

**`CUDA error: no kernel image is available for execution on the device`.** The
installed torch wheel has no kernels compiled for your GPU's compute capability.
`torch.cuda.is_available()` can still return `True` in this state, so confirm the
GPU with a real kernel launch rather than the flag:

```bash
./venv/bin/python -c "import torch; x=torch.randn(4000,4000,device='cuda'); print((x@x).sum().item())"
```

A printed number means the GPU is usable. The kernel error means the wheel and
card disagree: reinstall the torch build matching the card (older Pascal-era
cards, compute 6.1, want a `cu118` wheel; newer cards a recent `cuXXX` one - see
requirements.txt). This never blocks progress: data generation and every
verification step are CPU-only.

---

## Design notes

**Architecture is `768 → 256 → 32 → 1`, not HalfKP/HalfKA.** The input is plain
"piece of type T on square S, from perspective P", 2 x 6 x 64. Deliberately no
king bucket, because that keeps a king move an ordinary two-feature update. The
accumulator therefore never needs a mid-search refresh, which is what lets the
engine maintain it as a side effect of `place_piece`/`remove_piece`/`move_piece`
with no accumulator stack at all: `undo_move` replays the inverse piece
mutations, and integer adds are exactly invertible, so it cannot drift. Only the
feature transformer (the accumulator) is incremental; the two perspectives are
concatenated and run through one more hidden layer (`L2`) before the output,
evaluated fresh each call.

King buckets are the obvious generation-2 upgrade and worth a solid chunk of
Elo, but they require a refresh path and an accumulator stack. Get generation 1
working first.

**Quantization.** A clipped activation in `[0, 255]` represents a float in
`[0, 1]`. Feature weights are `round(w * 255)` in int16; every post-accumulator
layer stores weights as `round(w * 64)` and a bias pre-scaled by `255 * 64`, and
dividing that layer's int32 sum by 64 returns to the activation scale. The final
output maps back to engine units with `* 1640 / (255 * 64)`. The int16
accumulator is the binding constraint: 32 pieces' worth of feature weights must
fit, which is why the trainer clamps them to ±1.98 after every optimizer step
and why `export.py` refuses a net whose worst case exceeds 32767.

**Endgames still bypass the net.** The engine consults its KPK bitbase and
specialized mate evaluators before asking the network, because those are exact
results and small nets are notoriously bad at exactly those positions. That is
why `check_engine.py` skips them.

**The evaluation is not antisymmetric by construction.** `forward(acc, WHITE)`
and `forward(acc, BLACK)` are unrelated values for an arbitrary net - the output
layer has independent weights per perspective half plus a bias. A trained net
learns approximate antisymmetry from data. Nothing in the search relies on it
being exact.

**Speed is the usual way a first NNUE fails.** A net that evaluates better but
10× slower loses Elo. The incremental accumulator is what makes this viable; the
current forward pass is plain scalar C++, so if the net is clearly stronger per
node but weaker per second, hand-vectorizing the accumulator update and the
dense layers with AVX2 is the next lever.

---

## Layout

```
khaosnnue/
  quant.py      quantization constants - must match include/nnue.h
  features.py   FEN -> feature indices          (stdlib only)
  format.py     .nnue read/write                (stdlib only)
  refeval.py    reference integer forward pass  (stdlib only)
  model.py      torch model + quantizer
  dataset.py    memory-mapped packed dataset
prepare.py      text shards -> packed binary
train.py        training loop
export.py       checkpoint -> .nnue
scripts/
  run_datagen.sh    parallel self-play generation
  make_test_net.py  deterministic random net     (stdlib only)
  check_engine.py   engine vs python cross-check (stdlib only)
```

Everything defining the C++ contract is dependency-free on purpose, so it can be
tested and the engine cross-checked without a training environment installed.
