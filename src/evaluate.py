def evaluate_model(trainer):
    metrics = trainer.evaluate()
    print("\nEvaluation Results:\n", metrics)
    return metrics
