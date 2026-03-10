# Financial Sentiment Classifier: LLM Fine-Tuning Portfolio Project

An end-to-end machine learning project demonstrating the fine-tuning of Large Language Models (LLMs) for domain-specific NLP tasks. 

This repository showcases the fine-tuning of **DistilBERT** and **FinBERT** on the \`FinanceMTEB/financial_phrasebank\` dataset to classify financial sentences into positive, negative, or neutral sentiments. It features a highly modular architecture, robust evaluation metrics, and a CLI tool for real-time inference.

## 🚀 Key Features
* **Modular Architecture:** Separation of concerns across data loading, preprocessing, model training, and inference.
* **Modern Tooling:** Dependency management and reproducible environments using \`uv\`.
* **Advanced Training Techniques:** Implements mixed precision training (FP16), gradient accumulation, early stopping, and dynamic padding via custom data collators.
* **Comparative Analysis:** Evaluates a general-purpose model (\`distilbert-base-uncased\`) against a domain-specific model (\`ProsusAI/finbert\`).
* **Interactive CLI:** Includes a terminal-based inference script to test the fine-tuned models on custom financial text.

## 📁 Project Structure

```text
FIN-SENTIMENT-CLASSIFIER/
├── data/                        # Raw, stratified, and tokenized datasets (generated)
├── models/                      # Saved fine-tuned model checkpoints (generated)
├── notebooks/
│   └── model_fine_tuning.ipynb  # Main orchestration notebook for training
├── src/
│   ├── __init__.py
│   ├── config.py                # Centralized path and parameter configuration
│   ├── data_collator.py         # Dynamic padding for tokenized inputs
│   ├── data_loader.py           # Downloading, concatenating, and stratified splitting
│   ├── data_preprocessor.py     # Tokenization logic
│   ├── inference.py             # CLI tool for real-time model predictions
│   ├── metrics_computation.py   # Accuracy and Macro F1 calculation via Evaluate
│   ├── model_loader.py          # Checkpoint loading for training
│   ├── model_trainer.py         # Hugging Face Trainer configuration and callbacks
│   └── training_arguments.py    # Hyperparameters (FP16, learning rate, epochs, etc.)
├── tokenizer/                   # Saved tokenizers (generated)
├── .python-version              # Specifies Python version
├── pyproject.toml               # Project metadata and dependencies
└── uv.lock                      # Deterministic dependency lockfile
```

## 🛠️ Installation & Setup

This project uses [uv](https://github.com/astral-sh/uv) for lightning-fast Python package and dependency management.

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd FIN-SENTIMENT-CLASSIFIER
   ```

2. **Sync the environment:**
   This will automatically read the `uv.lock` file, create a virtual environment (`.venv`), and install all exact dependencies.
   ```bash
   uv sync
   ```

3. **Activate the virtual environment:**
   * **Linux/macOS:** `source .venv/bin/activate`
   * **Windows:** `.venv\Scripts\activate`

## 🧠 Training the Models

The training pipeline is orchestrated within the Jupyter Notebook. It handles data downloading, stratified splitting (70/10/20 for Train/Val/Test), tokenization, training, and saving the final weights.

1. Launch Jupyter:
   ```bash
   uv run jupyter notebook
   ```
2. Open `notebooks/model_fine_tuning.ipynb`.
3. Run all cells to download the `financial_phrasebank` dataset, train both DistilBERT and FinBERT, and output the final evaluation metrics (Accuracy and F1 Score). 

*Note: Models and tokenizers will be saved automatically to the `models/` and `tokenizer/` directories.*

## 💻 Running Inference

Once the models are fine-tuned, you can interact with them directly from your terminal using the custom inference script.

Run the inference script from the root directory:
```bash
python -m src.inference
```

**Example Interaction:**
```text
Enter the model name (distilbert or finbert): finbert
model loaded successfully.
Enter the input text for inference (or 'exit' to quit): The company reported a 20% increase in Q3 revenue.
Predicted sentiment: positive
Enter the input text for inference (or 'exit' to quit): Operating costs remained flat year-over-year.
Predicted sentiment: neutral
Enter the input text for inference (or 'exit' to quit): exit
Exiting inference.
```

## 📈 Evaluation Metrics
The models were evaluated on a 20% stratified test split of the `financial_phrasebank` dataset to ensure a balanced representation of positive, negative, and neutral sentiments. 

The evaluation metrics utilized were **Accuracy** and **Macro F1-Score** (to account for any potential class imbalances).

| Model | Accuracy | Macro F1-Score |
| :--- | :--- | :--- |
| **DistilBERT** (`distilbert-base-uncased`) | 96.25% | 94.84% |
| **FinBERT** (`ProsusAI/finbert`) | 96.03% | 94.27% |

*Note: Both models demonstrate highly robust performance, with DistilBERT showing a marginal performance edge on this specific dataset.*

## 🧑‍💻 About the Author
Built as a portfolio project to demonstrate practical experience with Large Language Models, the Hugging Face ecosystem (`transformers`, `datasets`, `evaluate`), and modern Python engineering workflows.
