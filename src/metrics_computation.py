import evaluate
import numpy as np

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)

    accuracy_result = accuracy_metric.compute(predictions=predictions, references=labels)
    accuracy = accuracy_result["accuracy"] if accuracy_result else 0.0
    
    f1_result = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    f1 = f1_result["f1"] if f1_result else 0.0

    return {"accuracy": accuracy, "f1": f1}