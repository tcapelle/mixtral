import time
from types import SimpleNamespace

import wandb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utils import parse_args

WANDB_PROJECT = "mixtral"
WANDB_ENTITY = "capecape"

config = SimpleNamespace(
    model_id = "mistralai/Mixtral-8x7B-v0.1",
    text = "Tell me a story about planes and cats, make it Tolkien-esque: ",
    use_flash_attention_2=False,
    load_in_4bit=True,
    load_in_8bit=False,
    use_cache=True,
    max_new_tokens=100,
)


def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, 
        use_flash_attention_2=config.use_flash_attention_2,
        load_in_4bit=config.load_in_4bit and not config.load_in_8bit,
        load_in_8bit=config.load_in_8bit and not config.load_in_4bit,
        use_cache=config.use_cache,
        device_map="auto",
    )

    print(model)
    
    inputs = tokenizer(config.text, return_tensors="pt").to(0)
    t = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
    tf = time.perf_counter() - t
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print(f"Total Tokens {len(outputs[0])}\nTime taken: {tf:.2f}s\ntokens/s: {len(outputs[0])/tf:.2f}")

if __name__ == "__main__":
    parse_args(config)
    main(config)