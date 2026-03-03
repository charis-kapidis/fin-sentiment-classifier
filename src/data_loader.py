from datasets import load_dataset, concatenate_datasets, ClassLabel, DatasetDict
from pathlib import Path

def download_raw_data(data_path: str):
    dataset = load_dataset(data_path)
    return dataset

def concatenate_raw_data(datasets: list):
    full_dataset = concatenate_datasets(datasets)
    return full_dataset

def cast_datatypes(dataset):
    dataset = dataset.cast_column("label", ClassLabel(num_classes=len(dataset.unique("label"))))           
    return dataset

def resplit_raw_data(dataset, split_ratios):
    temp_split = dataset.train_test_split(
        test_size=split_ratios['test'], 
        seed=42, 
        stratify_by_column="label"
    )
    final_split = temp_split['train'].train_test_split(
        test_size=split_ratios['val'], 
        seed=42, 
        stratify_by_column="label"
    )
    dataset_dict = DatasetDict({
        'train': final_split['train'],
        'val': final_split['test'],
        'test': temp_split['test']
    })
    return dataset_dict

def save_final_data(dataset_dict, save_path):
    current_path = Path.cwd()
    data_dir = (current_path.parent / "data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path_full = data_dir / save_path
    dataset_dict.save_to_disk(str(save_path_full))


def orchestrate_data_download(data_path: str, split_ratios={'train': 0.7, 'val': 0.1, 'test': 0.2}, save_path="financial_phrasebank_stratified"):
    dataset = download_raw_data(data_path)
    full_dataset = concatenate_raw_data([dataset["train"], dataset["test"]])
    full_dataset = cast_datatypes(full_dataset)
    dataset_dict = resplit_raw_data(full_dataset, split_ratios=split_ratios)
    save_final_data(dataset_dict, save_path=save_path)
