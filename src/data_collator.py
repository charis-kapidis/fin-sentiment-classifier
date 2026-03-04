from transformers import DataCollatorWithPadding

def create_data_collator(tokenizer):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator