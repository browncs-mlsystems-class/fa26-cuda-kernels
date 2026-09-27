# Runs `make check <suite>...` on a Hydra GPU node via Slurm, instead of on this machine.
# Same suite names as make check (saxpy_correct, sgemm3, all, ...); output lands in logs/.
#
# Usage: uv run runner.py <suite>...

import sys
import os
import stat
import subprocess

from run_tests import OPTIONS, NO_COLOR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Configuration ──────────────────────────────────────────────────────
VALID_SUITES = sorted(OPTIONS - {NO_COLOR})

# Currently constricts to one GPU type
GPU_CONSTRAINT="gtx_2080_ti|titan_rtx"

LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
JOB_TIME_HOURS = 10

def main():
    suites = sys.argv[1:]
    if not suites:
        print("Usage: uv run runner.py <suite>...")
        print(f"  suite: {', '.join(VALID_SUITES)}")
        sys.exit(1)

    unknown = sorted(set(suites) - OPTIONS)
    if unknown:
        print(f"Unknown suite(s): {', '.join(unknown)}")
        print(f"  suite: {', '.join(VALID_SUITES)}")
        sys.exit(1)

    options = " ".join(suites)
    job_name = "check_" + "_".join(suites)
    log_file = os.path.join(LOG_DIR, f"{job_name}.out")
    job_script = os.path.join(LOG_DIR, f"{job_name}.sh")

    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Running: make check {options}")
    print(f"  Submitting: {job_name} (GPU: {GPU_CONSTRAINT})")

    # Generate SBATCH job script
    slurm_script = f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --time={JOB_TIME_HOURS}:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output={log_file}
#SBATCH --error={log_file}
#SBATCH --constraint={GPU_CONSTRAINT}

cd {SCRIPT_DIR}

if ! command -v nvcc > /dev/null; then
  echo "ERROR: nvcc not found. Add the CUDA exports from the setup task to your ~/.bashrc."
  exit 1
fi

echo "=== GPU ==="
nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=name,compute_cap --format=csv,noheader

echo "=== Compiling ==="
make all || exit 1

echo "=== Running: make check {options} ==="
uv run python3 -u run_tests.py {options} {NO_COLOR}
"""

    with open(job_script, "w") as f:
        f.write(slurm_script)
    os.chmod(job_script, os.stat(job_script).st_mode | stat.S_IEXEC)

    result = subprocess.run(
        ["sbatch", "--parsable", job_script],
        capture_output=True, text=True, check=True,
    )
    job_id = result.stdout.strip()
    print(f"    Job ID: {job_id}")

    print()
    print("Monitor with: squeue -u $(whoami)")
    print(f"Logs in: {LOG_DIR}/")


if __name__ == "__main__":
    main()
