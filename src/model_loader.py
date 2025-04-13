from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.config import MODEL_ID, QUANT_CONFIG

def load_model():
    bnb_config = BitsAndBytesConfig(**QUANT_CONFIG)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return model

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
