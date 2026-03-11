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

Usage
-----
Standalone:
    python -m ray_cluster.spawn_workers config/expert_machine_mapping.json

From code:
    from ray_cluster.spawn_workers import spawn_workers
    model_to_actor = spawn_workers(mapping_path, registry_path)
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import ray


def spawn_workers(
    mapping_path: str,
    registry_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Spawn ExpertWorkerActors for all workers in expert_machine_mapping.json.

    Parameters
    ----------
    mapping_path : str
        Path to config/expert_machine_mapping.json.
    registry_path : str, optional
        Path to experts_registry.json. Passed to each actor so it can build
        its TaskExpert instances.  Defaults to the standard location inside
        the repo if omitted.

    Returns
    -------
    dict
        Mapping of base_model_key → Ray actor name.
        Pass this directly to RayGatewayService().
    """
    from ray_cluster.worker_actor import ExpertWorkerActor

    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expert machine mapping not found: {mapping_path}\n"
            "Set EXPERT_MAPPING_PATH or pass the correct path."
        )

    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    model_to_actor: Dict[str, str] = {}

    for worker_id, worker_cfg in mapping["workers"].items():
        model_key = worker_cfg["base_model_key"]
        actor_name = f"expert-worker-{worker_id}"

        # If the actor already exists (e.g. coordinator restarted), reuse it.
        try:
            ray.get_actor(actor_name)
            print(
                f"[spawn_workers] Actor '{actor_name}' already running — reusing."
            )
            model_to_actor[model_key] = actor_name
            continue
        except ValueError:
            pass  # Actor does not exist yet; create it below.

        print(
            f"[spawn_workers] Spawning '{actor_name}' "
            f"(model='{model_key}') on Ray cluster ..."
        )

        # ExpertWorkerActor.options() lets us set:
        #   name       — allows discovery by ray.get_actor(name)
        #   lifetime   — "detached" means the actor outlives this Python process
        #   num_gpus   — declared on the @ray.remote decorator; inherited here
        ExpertWorkerActor.options(
            name=actor_name,
            lifetime="detached",    # survives coordinator restarts
            max_restarts=-1,        # Ray restarts the actor indefinitely on crash
        ).remote(
            model_key=model_key,
            worker_id=worker_id,
            registry_path=registry_path,
        )

        model_to_actor[model_key] = actor_name
        print(
            f"[spawn_workers] Actor '{actor_name}' scheduled "
            f"(model will load in the background on the assigned GPU node)."
        )

    print(
        f"[spawn_workers] Done — {len(model_to_actor)} actor(s) registered: "
        f"{list(model_to_actor.values())}"
    )
    return model_to_actor


if __name__ == "__main__":
    mapping = sys.argv[1] if len(sys.argv) > 1 else "config/expert_machine_mapping.json"
    registry = sys.argv[2] if len(sys.argv) > 2 else None

    ray.init(address="auto")
    result = spawn_workers(mapping, registry)
    print("model_to_actor mapping:", result)
