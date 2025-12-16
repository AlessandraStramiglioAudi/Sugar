import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

OBJ_ROOT = "/leonardo_work/IscrC_GEN-X3D/GS/ShapeSplat-Gaussian_MAE/render_scripts/objaverse_renders"
UID_FILE = "/leonardo_work/IscrC_GEN-X3D/MeshAnything/MeshAnythingV2/val_uids.txt"
SCRIPT = "train_full_pipeline.py"

# how many runs in parallel on this node
MAX_WORKERS = 4 

BASE_CMD = [
    "python",
    SCRIPT,
    "-r", "dn_consistency",
    "--high_poly", "True",
    "--export_obj", "True",
]

def run_one(scene_path: str, gpu_id: int = 0):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = BASE_CMD + ["-s", scene_path]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    return scene_path, result.returncode

def read_ids(path: str):
    with open(path, "r") as f:
        for line in f:
            uid = line.strip()
            if not uid:
                continue
            yield uid

def main(num_workers: int = MAX_WORKERS, start_idx: int = 0):
    all_uids = list(read_ids(UID_FILE))
    print(f"Loaded {len(all_uids)} UIDs")

    all_uids = all_uids[start_idx:]
    # print("Using batch:", batch_uids)

    scene_paths = []
    for uid in all_uids:
        scene_dir = os.path.join(OBJ_ROOT, uid)
        if os.path.isdir(scene_dir):
            scene_paths.append(scene_dir)
        else:
            print(f"Warning: missing directory for {uid} -> {scene_dir}")

    print(f"Found {len(scene_paths)} scenes")

    # simple GPU assignment: job i -> GPU (i % NUM_GPUS)
    NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {}
        for i, sp in enumerate(scene_paths):
            gpu_id = i % NUM_GPUS
            fut = ex.submit(run_one, sp, gpu_id)
            futures[fut] = sp

        for fut in as_completed(futures):
            sp, code = fut.result()
            print(f"Finished {sp} with exit code {code}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch train sugar on multiple scenes from Objaverse"
    )
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS,
                        help="Maximum number of parallel workers")
    parser.add_argument("--start_idx", type=int, default=40,
                        help="Starting index for processing UIDs")
    args = parser.parse_args()

    main(num_workers=args.max_workers, start_idx=args.start_idx)
