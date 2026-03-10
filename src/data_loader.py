from datasets import load_dataset, concatenate_datasets, ClassLabel, DatasetDict
from src.config import DATA

def download_raw_data(data_path: str):
    dataset = load_dataset(data_path)
    return dataset

def concatenate_raw_data(datasets: list):
    full_dataset = concatenate_datasets(datasets)
    return full_dataset

def cast_datatypes(dataset):
    num_classes = len(dataset.unique("label"))
    dataset = dataset.cast_column("label", ClassLabel(num_classes=num_classes))           
    return dataset, num_classes

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

def save_data(dataset_dict, save_path):
    dataset_dict.save_to_disk(str(save_path))

def load_data(path: str) -> DatasetDict:
    dataset_dict = DatasetDict.load_from_disk(str(path))
    return dataset_dict

def orchestrate_data_download():

    # Step 1: Download raw data and save it to disk
    dataset = download_raw_data(DATA["raw"]["download_path"])
    save_data(dataset, save_path=DATA["raw"]["save_path"])

    # Step 2: Load the raw data, concatenate it, cast datatypes, resplit it, and save the stratified dataset to disk
    dataset = load_data(DATA["raw"]["save_path"])
    full_dataset = concatenate_raw_data([dataset["train"], dataset["test"]])
    full_dataset, num_classes = cast_datatypes(full_dataset)
    dataset_dict = resplit_raw_data(full_dataset, split_ratios=DATA["split_ratios"])
    save_data(dataset_dict, save_path=f"{DATA['stratified']['save_path']}")

    print(f"Dataset dictionary: {dataset_dict}")
    return dataset_dict
