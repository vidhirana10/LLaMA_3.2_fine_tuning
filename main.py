from src.auth import wandb_login, hf_login
from src.data_loader import load_data, preprocess
from src.model_loader import load_model, load_tokenizer
from src.trainer import setup_lora_model, train_model
from src.evaluate import evaluate_model
import os
from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    # Replace with your actual keys before running
    
    wandb_api_key = os.getenv("WANDB_API_KEY")
    hf_token = os.getenv("HF_TOKEN")

    wandb_login(wandb_api_key)
    hf_login(hf_token)

    # Load and preprocess data
    train_data, eval_data = load_data()
    train_data = train_data.map(preprocess)
    eval_data = eval_data.map(preprocess)

    # Load model and tokenizer
    tokenizer = load_tokenizer()
    model = load_model()

    # Setup LoRA and train
    model = setup_lora_model(model)
    trainer = train_model(model, tokenizer, train_data, eval_data)

    # Evaluate
    evaluate_model(trainer)
