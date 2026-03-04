import os
from pathlib import Path
from transformers import TrainingArguments

def create_training_arguments(output_path: str):
    current_path = Path.cwd()
    logging_dir = (current_path.parent / "logs").resolve()
    logging_dir.mkdir(parents=True, exist_ok=True)
    logging_dir_full = logging_dir / output_path
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(logging_dir_full)

    output_dir = (current_path.parent / "models" ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_full = output_dir / output_path

    training_args = TrainingArguments(
        output_dir=str(output_dir_full),
        
        # Training Duration and Batch Size
        per_device_train_batch_size = 4,
        num_train_epochs = 3,

        # Learning Rate & Scheduler
        learning_rate = 2e-5,
        lr_scheduler_type = "linear",
        warmup_steps=0.1,

        # Optimizer
        optim="adamw_torch",
        weight_decay=0.01,

        # Regularization & Training Stability
        gradient_accumulation_steps=4,

        # Mixed Precision Training
        fp16=True,

        # Logging & Monitoring Training
        logging_strategy="steps",
        logging_steps=50,

        # Experiment Tracking Integration
        report_to="tensorboard",

        # Evaluation
        eval_strategy="epoch",
        per_device_eval_batch_size=8,
        eval_accumulation_steps = 10,

        # Checkpointing & Saving
        save_strategy="epoch",

        # Best Model Tracking
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # Reproducibility
        seed=42,
        data_seed=42,
        )
    return training_args