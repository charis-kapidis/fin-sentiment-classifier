from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

data_dir = (PROJECT_ROOT / "data").resolve()
data_dir.mkdir(parents=True, exist_ok=True)

tokenizer_dir = (PROJECT_ROOT / "tokenizer").resolve()
tokenizer_dir.mkdir(parents=True, exist_ok=True)

model_dir = (PROJECT_ROOT / "models").resolve()
model_dir.mkdir(parents=True, exist_ok=True)


DATA = {
    "raw": {
        "download_path": "FinanceMTEB/financial_phrasebank",
        "save_path": f"{data_dir}/financial_phrasebank"
    },
    "stratified": {
        "save_path": f"{data_dir}/financial_phrasebank_stratified"
    },
    "split_ratios": {
        "train": 0.7, "val": 0.1, "test": 0.2
    }
}

MODEL = {
    "distilbert": {
        "checkpoint": "distilbert-base-uncased",
        "num_labels": 3,
        "save_tokenized_path": f"{data_dir}/tokenized_distilbert_financial_phrasebank_stratified",
        "save_tokenizer_path": f"{tokenizer_dir}/distilbert_tokenizer",
        "save_fine_tuned_path": f"{model_dir}/fine_tuned_distilbert_financial_phrasebank_stratified",
        "save_final_fine_tuned": f"{model_dir}/fine_tuned_distilbert_financial_phrasebank_stratified/final_model",
    },
    "finbert": {
        "checkpoint": "ProsusAI/finbert",
        "num_labels": 3,
        "save_tokenized_path": f"{data_dir}/tokenized_finbert_financial_phrasebank_stratified",
        "save_tokenizer_path": f"{tokenizer_dir}/finbert_tokenizer",
        "save_fine_tuned_path": f"{model_dir}/fine_tuned_finbert_financial_phrasebank_stratified",
        "save_final_fine_tuned": f"{model_dir}/fine_tuned_finbert_financial_phrasebank_stratified/final_model",
    },
}