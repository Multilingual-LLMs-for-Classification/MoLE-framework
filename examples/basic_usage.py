"""
Usage examples for the moe-classifier SDK — all three deployment modes.

Run from the MoLE-framework/ directory after installing:

    pip install -e .
    python examples/basic_usage.py
"""

from moe_classifier import MOEClassifier, DeploymentMode


def demo_local():
    """
    LOCAL mode — full in-process pipeline on a single GPU.

    This is the original behavior.  Loads gating models + LLM experts into
    GPU memory.  Requires a GPU and model weights on the local machine.
    """
    print("=" * 60)
    print("MODE 1: LOCAL  (single GPU, in-process)")
    print("=" * 60)

    clf = MOEClassifier()                          # defaults to deployment="local"
    # clf = MOEClassifier(deployment="local")      # explicit — same thing
    clf.initialize()
    print(f"Classifier ready: {clf}\n")

    # System info
    stats = clf.get_stats()
    print(f"  Domains   : {stats['total_domains']}  ({', '.join(stats['domains'])})")
    print(f"  Tasks     : {stats['total_tasks']}")
    print(f"  Languages : {stats['supported_languages']}\n")

    # Single classification
    result = clf.classify(
        text="This product exceeded my expectations! Great quality and fast shipping.",
        description="Rate this product review from 1 to 5 stars based on sentiment.",
    )
    print(f"  Result      : {result.result}")
    print(f"  Confidence  : {result.confidence:.2%}" if result.confidence else "  Confidence  : N/A")
    print(f"  Language    : {result.language}")
    print(f"  Domain      : {result.domain}")
    print(f"  Task        : {result.task}")
    print(f"  Route       : {result.routing_path}")
    print(f"  Time        : {result.processing_time_ms:.1f} ms\n")

    # Batch classification
    batch = clf.classify_batch([
        {"text": "Excellent service, will buy again!",   "description": "Rate 1-5."},
        {"text": "Terrible product, broke after one day.", "description": "Rate 1-5."},
        {"text": "Average experience, nothing special.",   "description": "Rate 1-5."},
    ])
    print(f"  Batch: {batch.successful} ok, {batch.failed} failed, "
          f"{batch.total_processing_time_ms:.0f} ms total\n")

    for item in batch.items:
        if item.success:
            r = item.result
            conf = f"{r.confidence:.2%}" if r.confidence else "N/A"
            print(f"  [{item.index}] result={r.result!r}  confidence={conf}")
        else:
            print(f"  [{item.index}] ERROR: {item.error}")


def demo_remote():
    """
    REMOTE mode — HTTP client to a running MoLE coordinator.

    No GPU or model weights needed locally.  Just point to the coordinator
    URL and authenticate.  Ideal for application developers consuming
    MoLE as a service.
    """
    print("\n" + "=" * 60)
    print("MODE 2: REMOTE  (HTTP client to running service)")
    print("=" * 60)

    # Option A: auto-login with username/password
    clf = MOEClassifier(
        deployment="remote",
        coordinator_url="http://localhost:8000",
        credentials={"username": "testuser", "password": "testpassword123"},
    )

    # Option B: use a pre-existing JWT token
    # clf = MOEClassifier(
    #     deployment="remote",
    #     coordinator_url="http://localhost:8000",
    #     token="eyJhbGciOiJIUzI1NiIs...",
    # )

    clf.initialize()
    print(f"Classifier ready: {clf}\n")

    result = clf.classify(
        text="Revenue increased by 20% year over year, exceeding analyst expectations.",
        description="Classify the sentiment of this financial statement on a scale of 1-5.",
    )
    print(f"  Result : {result.result}")
    print(f"  Route  : {result.routing_path}")
    print(f"  Time   : {result.processing_time_ms:.1f} ms\n")


def demo_distributed():
    """
    DISTRIBUTED mode — gating locally + dispatch to remote workers.

    Loads lightweight gating models locally (FastText, XLM-RoBERTa,
    Q-learning routers) but dispatches LLM inference to remote expert
    workers via HTTP.

    This is the programmatic equivalent of SERVICE_MODE=coordinator
    without the FastAPI layer.
    """
    print("\n" + "=" * 60)
    print("MODE 3: DISTRIBUTED  (gating locally, workers remotely)")
    print("=" * 60)

    clf = MOEClassifier(
        deployment="distributed",
        expert_mapping="config/expert_machine_mapping.json",
    )
    clf.initialize()
    print(f"Classifier ready: {clf}\n")

    result = clf.classify(
        text="Revenue increased by 20% year over year.",
        description="Classify the sentiment of this financial statement on a scale of 1-5.",
    )
    print(f"  Result : {result.result}")
    print(f"  Route  : {result.routing_path}")
    print(f"  Time   : {result.processing_time_ms:.1f} ms\n")


def main():
    print("MOE Classifier SDK — Deployment Mode Examples\n")
    print(f"Available modes: {[m.value for m in DeploymentMode]}\n")

    # Uncomment the mode you want to try:

    # --- Mode 1: Local (requires GPU + model weights) ---
    demo_local()

    # --- Mode 2: Remote (requires a running MoLE service) ---
    # demo_remote()

    # --- Mode 3: Distributed (requires GPU for gating + running workers) ---
    # demo_distributed()

    print("\nDone.")


if __name__ == "__main__":
    main()
