from src.config import MODEL
from src.data_loader import load_data
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from src.metrics_computation import compute_metrics
from datasets import Dataset
from typing import cast


def orchestrate_evaluation(model_name, model, data_collator):
    dataset_dict = load_data(MODEL[model_name]["save_tokenized_path"])
    test_data = dataset_dict["test"]

    trainer = Trainer(model=model, data_collator=data_collator)

    predictions = trainer.predict(test_data)
    logits = predictions.predictions

    metrics = compute_metrics((logits, test_data["label"]))
    print(metrics)


def orchestrate_inference():
    model_name = input("Enter the model name (distilbert or finbert): ")
    if model_name not in MODEL:
        raise ValueError(f"Invalid model name. Please choose from {list(MODEL.keys())}.")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL[model_name]["save_tokenizer_path"], local_files_only=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL[model_name]["save_final_fine_tuned"], local_files_only=True)

    print("model loaded successfully.")

    while True:
        input_text = input("Enter the input text for inference (or 'exit' to quit): ")
        if input_text.lower() == 'exit':
            print("Exiting inference.")
            break
        
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()
        predicted_label = np.argmax(logits, axis=1)[0]
        label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
        print(f"Predicted sentiment: {label_mapping[predicted_label]}")

if __name__ == "__main__":
    orchestrate_inference()