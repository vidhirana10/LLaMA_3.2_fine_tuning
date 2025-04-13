Got it! Here's your updated `README.md` with `.env` usage instructions included under the **How to Run** section:

---

```markdown
# LLaMA 3.2 1B LoRA Fine-Tuning

This project fine-tunes Meta's LLaMA 3.2 1B model on the SQuAD dataset using PEFT (LoRA) and 4-bit quantization.

## Features
- BitsAndBytes 4-bit quantization
- PEFT (LoRA) for parameter-efficient fine-tuning
- Preprocessing SQuAD into a prompt-completion format
- HuggingFace `Trainer` for training and evaluation
- Integrated with W&B for logging

## How to Run

1. Clone the repo and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Create a `.env` file in the root directory with the following content:
    ```env
    WANDB_API_KEY=your_wandb_api_key_here
    HF_TOKEN=your_huggingface_token_here
    ```

3. Run the pipeline:
    ```bash
    python main.py
    ```

## ⚙️ CUDA Requirements (for running on personal machine)

- A compatible NVIDIA GPU (e.g., RTX 30xx series)
- Latest [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx) installed
- CUDA 12.1 or higher supported
- Recommended: Use **WSL2 with Ubuntu** for CUDA support if on Windows

To verify your setup:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

If `False`, check your CUDA toolkit and PyTorch installation:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Structure
```
llama_lora_finetune/
├── src/
│   ├── auth.py
│   ├── config.py
│   ├── data_loader.py
│   ├── model_loader.py
│   ├── trainer.py
│   └── evaluate.py
├── main.py
├── .env
├── requirements.txt
├── README.md
└── .gitignore
```

---

Built with ❤️ by Vidhi Rana
```

