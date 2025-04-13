from datasets import load_dataset
from src.config import TRAIN_SIZE, EVAL_SIZE

def load_data():
    dataset = load_dataset("squad")
    train_dataset = dataset["train"].select(range(TRAIN_SIZE))
    eval_dataset = dataset["validation"].select(range(EVAL_SIZE))
    return train_dataset, eval_dataset

def preprocess(example):
    prompt = f"Question: {example['question']} Context: {example['context']} Answer:"
    return {"input_ids": prompt, "labels": example["answers"]["text"][0] if example["answers"]["text"] else ""}
