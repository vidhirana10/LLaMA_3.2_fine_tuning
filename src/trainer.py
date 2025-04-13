from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from src.config import LORA_CONFIG, TRAINING_ARGS

def setup_lora_model(model):
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, config)
    return model

def train_model(model, tokenizer, train_dataset, eval_dataset):
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(**TRAINING_ARGS)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer
