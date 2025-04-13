MODEL_ID = "meta-llama/Llama-3.2-1B"

# BitsAndBytes Quantization
QUANT_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4"
}

# LoRA Config
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Training Args
TRAINING_ARGS = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "warmup_steps": 2,
    "max_steps": 50,
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 5,
    "output_dir": "outputs",
    "report_to": ["wandb"]
}

# Dataset Params
TRAIN_SIZE = 1000
EVAL_SIZE = 200
