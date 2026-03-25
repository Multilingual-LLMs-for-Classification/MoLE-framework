"""
spawn_workers — create named ExpertWorkerActors on the Ray cluster.

Called once at coordinator startup (via app/main.py lifespan).
Can also be run standalone to pre-warm workers before the coordinator starts.

What this replaces
------------------
Old setup:
  - Each worker machine ran a separate Docker container with a FastAPI app
  - expert_machine_mapping.json held host:port for every worker
  - Workers had to be started manually on each machine before the coordinator

New setup:
  - Ray cluster is already running (head + worker nodes joined)
  - This script asks Ray to schedule one ExpertWorkerActor per model
  - Ray automatically picks a node with a free GPU — no manual host/port config
  - Workers are "detached" — they survive coordinator restarts

Improvements over the first version
-------------------------------------
1. Ray-native config (ray_worker_registry.json)
   A separate, clean config file is read when USE_RAY=true.  It has no url/port
   fields (those are irrelevant for Ray actors).  The old expert_machine_mapping.json
   is still used for HTTP mode — neither file is modified.

2. num_replicas
   Each worker entry can declare num_replicas > 1.  spawn_workers creates that
   many actor instances behind the same model_key, each with its own GPU slot.
   RayGatewayService then round-robins requests across replicas.

3. Placement group hints
   An optional placement_group_strategy field (STRICT_PACK / PACK / SPREAD /
   STRICT_SPREAD) lets you pin a model's replicas to specific nodes for
   deterministic scheduling and reproducibility.

Usage
-----
Standalone:
    python -m ray_cluster.spawn_workers \\
        --ray-registry config/ray_worker_registry.json \\
        --expert-registry moe_router/experts/config/experts_registry.json

From code (called by routing_service.initialize):
    from ray_cluster.spawn_workers import spawn_workers
    model_to_actors = spawn_workers(
        ray_registry_path=settings.ray_worker_registry_path,
        expert_registry_path=settings.expert_registry_path,
    )
    # model_to_actors: Dict[str, List[str]]
    #   e.g. {"llama-2-7b-hf": ["expert-worker-worker-0-r0"],
    #          "qwen2.5-7b-instruct": ["expert-worker-worker-1-r0"]}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import ray


# ---------------------------------------------------------------------------
# Placement group helpers
# ---------------------------------------------------------------------------

_PG_STRATEGIES = {"STRICT_PACK", "PACK", "SPREAD", "STRICT_SPREAD"}


def _make_placement_group(strategy: str, num_gpus: float, num_cpus: int):
    """Create a Ray placement group for one actor replica."""
    from ray.util.placement_group import placement_group

    pg = placement_group(
        [{"GPU": num_gpus, "CPU": num_cpus}],
        strategy=strategy,
    )
    ray.get(pg.ready())
    return pg


# ---------------------------------------------------------------------------
# Main spawn function
# ---------------------------------------------------------------------------

def spawn_workers(
    ray_registry_path: str,
    expert_registry_path: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Spawn ExpertWorkerActors for all workers in ray_worker_registry.json.

    Parameters
    ----------
    ray_registry_path : str
        Path to config/ray_worker_registry.json.
    expert_registry_path : str, optional
        Path to experts_registry.json passed to each actor so it can build
        its TaskExpert instances.  Defaults to the standard in-repo location.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of base_model_key → list of Ray actor names.
        List length equals num_replicas for that model.
        Pass directly to RayGatewayService().
    """
    from ray_cluster.worker_actor import ExpertWorkerActor

    path = Path(ray_registry_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Ray worker registry not found: {ray_registry_path}\n"
            "Set RAY_WORKER_REGISTRY_PATH or pass the correct path."
        )

    with open(path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    model_to_actors: Dict[str, List[str]] = {}

    for worker_id, worker_cfg in registry["workers"].items():
        # Skip workers explicitly disabled (machine not yet in the cluster)
        if not worker_cfg.get("enabled", True):
            print(f"[spawn_workers] Skipping '{worker_id}' (enabled=false).")
            continue

        model_key    = worker_cfg["base_model_key"]
        num_replicas = int(worker_cfg.get("num_replicas", 1))
        num_gpus     = float(worker_cfg.get("num_gpus", 1))
        pg_strategy  = worker_cfg.get("placement_group_strategy", "").upper() or None

        if pg_strategy and pg_strategy not in _PG_STRATEGIES:
            raise ValueError(
                f"Worker '{worker_id}': invalid placement_group_strategy "
                f"'{pg_strategy}'. Valid values: {sorted(_PG_STRATEGIES)}"
            )

        actor_names: List[str] = []

        for replica_idx in range(num_replicas):
            actor_name = f"expert-worker-{worker_id}-r{replica_idx}"

            # If the actor already exists (coordinator restarted), reuse it.
            try:
                ray.get_actor(actor_name)
                print(
                    f"[spawn_workers] Actor '{actor_name}' already running — reusing."
                )
                actor_names.append(actor_name)
                continue
            except ValueError:
                pass  # Does not exist yet — create below.

            print(
                f"[spawn_workers] Spawning '{actor_name}' "
                f"(model='{model_key}', num_gpus={num_gpus}, "
                f"replica {replica_idx + 1}/{num_replicas}) ..."
            )

            actor_options = {
                "name": actor_name,
                "lifetime": "detached",   # survives coordinator restarts
                "max_restarts": -1,       # Ray restarts indefinitely on crash
                "num_gpus": num_gpus,
            }

            # Optional placement group for deterministic GPU pinning
            if pg_strategy:
                pg = _make_placement_group(pg_strategy, num_gpus, num_cpus=4)
                actor_options["placement_group"] = pg
                actor_options["placement_group_bundle_index"] = 0
                print(
                    f"[spawn_workers] Placement group strategy='{pg_strategy}' "
                    f"applied to '{actor_name}'."
                )

            ExpertWorkerActor.options(**actor_options).remote(
                model_key=model_key,
                worker_id=f"{worker_id}-r{replica_idx}",
                registry_path=expert_registry_path,
            )

            actor_names.append(actor_name)
            print(
                f"[spawn_workers] Actor '{actor_name}' scheduled "
                f"(model will load in the background on the assigned GPU node)."
            )

        # If a model_key appears under multiple worker entries (unusual but
        # valid — e.g. different quantisation versions), merge the lists.
        if model_key in model_to_actors:
            model_to_actors[model_key].extend(actor_names)
        else:
            model_to_actors[model_key] = actor_names

    total = sum(len(v) for v in model_to_actors.values())
    print(
        f"[spawn_workers] Done — {len(model_to_actors)} model(s), "
        f"{total} actor replica(s) registered: {model_to_actors}"
    )
    return model_to_actors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spawn MoLE ExpertWorkerActors on a running Ray cluster."
    )
    parser.add_argument(
        "--ray-registry",
        default="config/ray_worker_registry.json",
        help="Path to ray_worker_registry.json (default: config/ray_worker_registry.json)",
    )
    parser.add_argument(
        "--expert-registry",
        default=None,
        help="Path to experts_registry.json (default: auto-resolved inside repo)",
    )
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Ray cluster address (default: auto)",
    )
    args = parser.parse_args()

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    result = spawn_workers(
        ray_registry_path=args.ray_registry,
        expert_registry_path=args.expert_registry,
    )
    print("\nmodel_to_actors mapping:")
    for model, actors in result.items():
        print(f"  {model}: {actors}")
    sys.exit(0)
