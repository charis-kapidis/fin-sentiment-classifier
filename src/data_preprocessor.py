from src.data_loader import load_data, save_data
from transformers import AutoTokenizer
from src.config import DATA, MODEL

def tokenize_data(checkpoint, dataset_dict):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)
    
    dataset_dict = dataset_dict.map(tokenize_function, batched=True)
    return dataset_dict, tokenizer

def orchestrate_data_preprocess(model_name: str):
    dataset_dict = load_data(DATA["stratified"]["save_path"])
    dataset_dict, tokenizer = tokenize_data(MODEL[model_name]["checkpoint"], dataset_dict)
    save_data(dataset_dict, save_path=MODEL[model_name]["save_tokenized_path"])
    print(f"Tokenized dataset dictionary: {dataset_dict}")
    return tokenizer