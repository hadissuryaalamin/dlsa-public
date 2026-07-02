"""
Runner untuk dua eksperimen:
  1. CNN+Transformer (paper baseline, tanpa GNN)
  2. CNN+GNN+Transformer (ekstensi)

Jalankan:
    python run_experiments.py --gpu 0 1 2 3
    python run_experiments.py          # auto-select GPU
    python run_experiments.py --dry-run  # hanya validasi config
"""
import argparse
import logging
import yaml

from run_train_test import run


EXPERIMENTS = [
    ("configs/cnntransformer-paper.yaml", "paper-cnn-transformer"),
    ("configs/cnntransformer-gnn.yaml",   "gnn-cnn-transformer"),
]


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run CNN+Trans vs CNN+GNN+Trans experiments.")
    parser.add_argument("--gpu", "-g", nargs="*", type=int,
                        help="GPU device IDs (e.g. --gpu 0 1 2 3). Default: auto-select.")
    parser.add_argument("--only", choices=["paper", "gnn"],
                        help="Run only one experiment (paper or gnn).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load configs and print summary without training.")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.only == "paper":
        experiments = [EXPERIMENTS[0]]
    elif args.only == "gnn":
        experiments = [EXPERIMENTS[1]]

    for config_path, run_id in experiments:
        config = load_config(config_path)
        if args.dry_run:
            print(f"\n=== DRY RUN: {run_id} ===")
            print(f"  config: {config_path}")
            print(f"  model:  {config['model_name']}")
            print(f"  use_gnn: {config['model'].get('use_gnn', False)}")
            print(f"  batch_size: {config['batch_size']}")
            print(f"  num_epochs: {config['num_epochs']}")
            print(f"  length_training: {config['length_training']}")
            print(f"  factor_models: {list(config['factor_models'].keys())}")
            continue

        logging.info(f"\n{'='*60}")
        logging.info(f"STARTING EXPERIMENT: {run_id}")
        logging.info(f"{'='*60}")

        run(
            config=config,
            run_id=run_id,
            gpu_device_ids=args.gpu,
        )

        logging.info(f"FINISHED EXPERIMENT: {run_id}\n")


if __name__ == "__main__":
    main()
