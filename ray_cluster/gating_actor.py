"""
GatingActor — Ray Actor wrapping the gating pipeline (language detection,
domain classification, Q-learning task routing).

Why a separate Ray Actor for gating?
-------------------------------------
In the original setup the coordinator runs the gating pipeline directly in its
own process on CPU (via asyncio thread pool).  That works, but:

  1. XLM-RoBERTa inference (domain classifier) and Q-learning encoder are
     transformer-based and benefit from GPU.
  2. Running gating in the coordinator process means gating + FastAPI request
     handling compete for the same CPU cores.
  3. With a Ray Actor we can declare fractional GPU resources (e.g. num_gpus=0.1)
     so 10 gating actors share one GPU alongside LLM workers without oversubscribing.

Resource declaration
---------------------
``@ray.remote`` is declared with no GPU constraint here.  The actual GPU fraction
is set at spawn time via ``.options(num_gpus=N)`` in spawn_gating_actor(), so the
coordinator can control it from config (USE_GATING_ACTOR + GATING_ACTOR_NUM_GPUS).

  GATING_ACTOR_NUM_GPUS=0.0  → CPU only (default, safe for any machine)
  GATING_ACTOR_NUM_GPUS=0.1  → 10% of one GPU (10 gating actors share a GPU)

Usage
-----
The coordinator calls:

    gating_result = await gating_actor.run_gating.remote(prompt)

Because GatingActor methods are async, the coordinator can simply ``await`` the
ObjectRef without a thread-pool hop.
"""

from dataclasses import asdict
from typing import Any, Dict, Optional

import ray


# ---------------------------------------------------------------------------
# Named constant for actor discovery
# ---------------------------------------------------------------------------
GATING_ACTOR_NAME = "mole-gating-actor"


# ---------------------------------------------------------------------------
# Ray Actor
# ---------------------------------------------------------------------------

@ray.remote
class GatingActor:
    """
    Ray Actor that holds the lightweight gating pipeline (FastText + XLM-RoBERTa
    + Q-learning) and serves run_gating() calls asynchronously.

    Lifecycle
    ---------
    1. __init__: loads PromptRoutingSystem(coordinator_only=True)
    2. run_gating(): called by the coordinator; returns a plain dict matching
       GatingResult fields so it can be reconstructed without importing the dataclass.
    3. On crash: Ray restarts the actor (gating models reload on __init__).
    """

    def __init__(self):
        from moe_router.gating.components.routing_system import PromptRoutingSystem

        print("[GatingActor] Loading gating pipeline ...")
        self._system = PromptRoutingSystem(
            training_mode=False,
            coordinator_only=True,
        )
        self._ready = True
        print("[GatingActor] Gating pipeline ready.")

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    async def run_gating(self, prompt: str) -> Dict[str, Any]:
        """
        Run language detection → domain classification → task routing.

        Returns a dict with the same fields as GatingResult so the coordinator
        can reconstruct routing without importing the dataclass.

        Parameters
        ----------
        prompt : str
            Concatenation of description + text (built by the coordinator).

        Returns
        -------
        dict with keys: language, domain, task, base_model_key, adapter_name,
                        routing_path
        """
        import asyncio
        # PromptRoutingSystem.run_gating is synchronous (transformer inference).
        # Run in executor so the actor's asyncio event loop is not blocked,
        # keeping the actor responsive to health-check calls while gating runs.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._system.run_gating, prompt)
        return asdict(result)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self._ready


# ---------------------------------------------------------------------------
# Spawn helper (called once at coordinator startup)
# ---------------------------------------------------------------------------

def spawn_gating_actor(num_gpus: float = 0.0) -> Any:
    """
    Ensure a named GatingActor is running on the Ray cluster.

    If an actor with GATING_ACTOR_NAME already exists (coordinator restarted),
    it is reused.  Otherwise a new one is created.

    Parameters
    ----------
    num_gpus : float
        Fractional GPU allocation per actor.
        0.0 = CPU only (default, always safe).
        0.1 = 10% of a GPU (10 actors share one GPU).

    Returns
    -------
    Ray actor handle
    """
    try:
        actor = ray.get_actor(GATING_ACTOR_NAME)
        print(f"[GatingActor] Existing actor '{GATING_ACTOR_NAME}' reused.")
        return actor
    except ValueError:
        pass  # Does not exist yet — create below.

    resource_label = f"num_gpus={num_gpus}" if num_gpus > 0 else "CPU only"
    print(
        f"[GatingActor] Spawning '{GATING_ACTOR_NAME}' "
        f"({resource_label}, max_restarts=-1) ..."
    )

    actor = (
        GatingActor
        .options(
            name=GATING_ACTOR_NAME,
            lifetime="detached",
            max_restarts=-1,
            num_gpus=num_gpus,
            num_cpus=2,
        )
        .remote()
    )
    print(f"[GatingActor] Actor '{GATING_ACTOR_NAME}' scheduled.")
    return actor
