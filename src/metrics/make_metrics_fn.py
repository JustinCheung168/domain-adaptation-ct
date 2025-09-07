from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def make_metrics_fn(model):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        logits1, logits2 = preds

        labels1 = labels[0]
        labels2 = labels[1]

        # Convert to NumPy
        preds1 = np.argmax(logits1, axis=-1)
        preds2 = np.argmax(logits2, axis=-1)

        labels1 = np.array(labels1)
        labels2 = np.array(labels2)

        # # Print current lambda (from GradientReversalLayer)
        # if hasattr(model, 'grad_reverse'):
        #     print(f"[Metrics] Current Lambda: {model.grad_reverse.alpha:.4f}")

        return {
            "accuracy_branch1": accuracy_score(labels1, preds1),
            "precision_branch1": precision_score(labels1, preds1, average="macro", zero_division=0),
            "recall_branch1": recall_score(labels1, preds1, average="macro", zero_division=0),
            "f1_branch1": f1_score(labels1, preds1, average="macro", zero_division=0),
            "accuracy_branch2": accuracy_score(labels2, preds2),
            "precision_branch2": precision_score(labels2, preds2, average="macro", zero_division=0),
            "recall_branch2": recall_score(labels2, preds2, average="macro", zero_division=0),
            "f1_branch2": f1_score(labels2, preds2, average="macro", zero_division=0),
            "lambda": model.grad_reverse.alpha if hasattr(model, 'grad_reverse') else None
        }
    return compute_metrics

