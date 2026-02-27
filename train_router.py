"""
Training script for the routing system (domain classifier + Q-learning task routers).

Usage:
    python train_router.py                          # Train both with defaults
    python train_router.py --epochs 3               # More epochs
    python train_router.py --data path/to/data.json # Custom training data
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

from moe_router.gating.components.routing_system import PromptRoutingSystem


def load_training_data(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train routing system classifiers")
    parser.add_argument("--data", type=str, default="train_combined.json",
                        help="Path to training data JSON")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs for domain classifier")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate for domain classifier")
    parser.add_argument("--skip-domain", action="store_true",
                        help="Skip domain classifier training")
    parser.add_argument("--skip-task", action="store_true",
                        help="Skip task router training")
    args = parser.parse_args()

    print("=" * 80)
    print("ROUTING SYSTEM TRAINING")
    print("=" * 80)

    # Initialize in training mode (skips loading old checkpoints)
    print("\nInitializing routing system (training_mode=True)...")
    system = PromptRoutingSystem(training_mode=True)

    stats = system.get_system_stats()
    print(f"Domains: {stats['domains']}")
    print(f"Total tasks: {stats['total_tasks']}")

    # Load training data
    print(f"\nLoading training data from: {args.data}")
    training_data = load_training_data(args.data)
    print(f"Loaded {len(training_data)} samples")

    from collections import Counter
    task_dist = Counter(d["task"] for d in training_data)
    print(f"Task distribution: {dict(task_dist)}")

    # Train domain classifier
    if not args.skip_domain:
        print("\n" + "=" * 80)
        print("TRAINING DOMAIN CLASSIFIER")
        print("=" * 80)
        system.train_domain_classifier(
            training_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    # Train Q-learning task routers
    if not args.skip_task:
        print("\n" + "=" * 80)
        print("TRAINING Q-LEARNING TASK ROUTERS")
        print("=" * 80)
        system.train_q_routers(training_data)

    # Save all models
    print("\nSaving models...")
    system.save_all_models()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
