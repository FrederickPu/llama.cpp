#!/bin/bash
#SBATCH --job-name=premise-loop
#SBATCH --partition=general
#SBATCH --time=2-00:00:00         # check the partition limit: sinfo -p general -o "%l"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16        # extract_data elaborates 8 files at once
#SBATCH --mem=128G                # training holds all of premises.jsonl in memory
#SBATCH --output=%x-%j.out
#SBATCH --open-mode=append        # a requeued run appends to the same log
#SBATCH --requeue
#SBATCH --signal=B:USR1@600       # warn 10 minutes before the time limit

# Mine Mathlib premises and train a premise selector from all-distilroberta-v1.
# Submit from tools/training; ENTRY picks the Lean files and extra arguments go to premise_loop.py:
#   sbatch premise_loop.sbatch --rounds 3
#   ENTRY=.lake/packages/mathlib/Mathlib/Order/Defs sbatch --time=2:00:00 premise_loop.sbatch --rounds 1 --out out/smoke
# The job requeues itself near the time limit and resumes from --out.
#
# One-time setup on a login node (lake itself runs in the job):
#   curl -sSf https://elan.lean-lang.org/elan-init.sh | sh -s -- -y --default-toolchain none
#   python3 -m venv ~/venvs/premise && source ~/venvs/premise/bin/activate
#   pip install torch  # not requirements-convert_hf_to_gguf.txt, which pins the CPU wheel
#   pip install transformers==4.57.6 huggingface_hub safetensors sentencepiece protobuf numpy pyyaml tqdm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f premise_loop.py ]] || { echo "submit from tools/training"; exit 1; }

source "${VENV:-$HOME/venvs/premise}/bin/activate"
export PATH="$HOME/.elan/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# Lean starts a worker per core on the node, not per core allocated to the job
export LEAN_NUM_THREADS="$SLURM_CPUS_PER_TASK"

lake exe cache get
lake build extract_data

# Canonical hardcodes the premise server to 127.0.0.1:8766, so two jobs on one node would share it
if curl -sf --max-time 2 http://127.0.0.1:8766/health > /dev/null; then
    echo "port 8766 is already in use on $(hostname)"
    exit 1
fi

trap 'echo "time limit near, requeueing"; scontrol requeue "$SLURM_JOB_ID"' USR1
python3 premise_loop.py "${ENTRY:-.lake/packages/mathlib/Mathlib}" --model sentence-transformers/all-distilroberta-v1 "$@" &
wait $!
