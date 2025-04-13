import wandb
from huggingface_hub import login

def wandb_login(api_key: str):
    wandb.login(key=api_key)

def hf_login(token: str):
    login(token=token)
