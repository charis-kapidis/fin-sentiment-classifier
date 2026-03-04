from src.data_loader import load_final_data, save_tokenized_data
from transformers import AutoTokenizer

def tokenize_data(checkpoint, dataset_dict):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)
    
    dataset_dict = dataset_dict.map(tokenize_function, batched=True)
    return dataset_dict, tokenizer

def orchestrate_data_preprocess(data_path: str, checkpoint: str, save_path: str):
    dataset_dict = load_final_data(data_path)
    dataset_dict, tokenizer = tokenize_data(checkpoint, dataset_dict)
    save_tokenized_data(dataset_dict, save_path=f"tokenized_{save_path}")
    print(f"Tokenized dataset dictionary: {dataset_dict}")
    return tokenizer