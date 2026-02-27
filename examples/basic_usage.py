"""
Basic usage examples for the moe-classifier SDK.

Run from the moe-classification-service/ directory after installing:

    pip install -e .
    python examples/basic_usage.py
"""

from moe_classifier import MOEClassifier


def main():
    # ------------------------------------------------------------------
    # 1. Initialize (loads all models — takes a moment on first run)
    # ------------------------------------------------------------------
    print("Initializing MOEClassifier...")
    clf = MOEClassifier()
    clf.initialize()
    print(f"Classifier ready: {clf}\n")

    # ------------------------------------------------------------------
    # 2. System info
    # ------------------------------------------------------------------
    stats = clf.get_stats()
    print("=== System Stats ===")
    print(f"  Domains   : {stats['total_domains']}  ({', '.join(stats['domains'])})")
    print(f"  Tasks     : {stats['total_tasks']}")
    print(f"  Languages : {stats['supported_languages']}")
    print()

    # ------------------------------------------------------------------
    # 3. Single classification — sentiment (finance/rating)
    # ------------------------------------------------------------------
    print("=== Single Classification ===")
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
    print(f"  Time        : {result.processing_time_ms:.1f} ms")
    print()

    # ------------------------------------------------------------------
    # 4. Single classification — with domain probabilities
    # ------------------------------------------------------------------
    print("=== Classification with Domain Probabilities ===")
    result2 = clf.classify(
        text="The invoice shows a wire transfer to an offshore account.",
        description="Detect PII entities in this financial document.",
        return_domain_probabilities=True,
    )
    print(f"  Result      : {result2.result}")
    print(f"  Route       : {result2.routing_path}")
    if result2.domain_probabilities:
        print("  Domain probs:")
        for domain, prob in sorted(result2.domain_probabilities.items(), key=lambda x: -x[1]):
            print(f"    {domain}: {prob:.2%}")
    print()

    # ------------------------------------------------------------------
    # 5. Batch classification
    # ------------------------------------------------------------------
    print("=== Batch Classification ===")
    batch_items = [
        {
            "text": "Excellent service, will buy again!",
            "description": "Rate this review from 1 to 5 stars.",
        },
        {
            "text": "Terrible product, broke after one day.",
            "description": "Rate this review from 1 to 5 stars.",
        },
        {
            "text": "Average experience, nothing special.",
            "description": "Rate this review from 1 to 5 stars.",
        },
    ]

    batch = clf.classify_batch(batch_items)
    print(f"  Total      : {len(batch.items)}")
    print(f"  Successful : {batch.successful}")
    print(f"  Failed     : {batch.failed}")
    print(f"  Total time : {batch.total_processing_time_ms:.1f} ms")
    print()

    for item in batch.items:
        if item.success:
            r = item.result
            conf = f"{r.confidence:.2%}" if r.confidence else "N/A"
            print(f"  [{item.index}] result={r.result!r}  confidence={conf}  ({r.processing_time_ms:.0f}ms)")
        else:
            print(f"  [{item.index}] ERROR: {item.error}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
