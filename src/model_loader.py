from transformers import AutoModelForSequenceClassification

def load_model(checkpoint: str, num_labels: int, device: str):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    model.to(device)
    return model