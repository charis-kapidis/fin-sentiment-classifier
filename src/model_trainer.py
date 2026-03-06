from transformers import Trainer, EarlyStoppingCallback

from src.data_loader import load_data
from src.metrics_computation import compute_metrics

def create_trainer(model, training_arguments, tokenized_datasets_path, data_collator, tokenizer):
    tokenized_datasets = load_data(tokenized_datasets_path)

    trainer = Trainer(
        model,
        training_arguments,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    return trainer
