import os
from types import SimpleNamespace

import wandb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

from utils import parse_args

WANDB_PROJECT = "mixtral"
WANDB_ENTITY = "capecape"

config = SimpleNamespace(
    model_id = "mistralai/Mixtral-8x7B-v0.1",
    batch_size = 1, 
    use_flash_attention_2=False,
    load_in_4bit=True,
)


def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, 
        use_flash_attention_2=config.use_flash_attention_2,
        load_in_4bit=config.load_in_4bit,
    )

    text = "Tell me a story about planes and cats, make it Tolkien-esque: "
    inputs = tokenizer(text, return_tensors="pt").to(0)

    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    parse_args(config)
    main(config)