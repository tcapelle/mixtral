import argparse
from ast import literal_eval

from transformers import Trainer

def str2bool(v):
    "Fix Argparse to process bools"
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(config):
    print("Running with the following config")
    parser = argparse.ArgumentParser(description='Run training baseline')
    for k,v in config.__dict__.items():
        parser.add_argument('--'+k, type=type(v) if type(v) is not bool else str2bool, 
                            default=v, 
                            help=f"Default: {v}")
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        try:
            # attempt to eval it it (e.g. if bool, number, or etc)
            attempt = literal_eval(v)
        except (SyntaxError, ValueError):
            # if that goes wrong, just use the string
            attempt = v
        setattr(config, k, attempt)
        print(f"--{k}:{v}")


def debug_trainer_data(trainer: Trainer):
    """Print a bunch of debug info about how the packed dataset is being constructed.
    We set everythin to finite to avoid iterating forever"""
    print("Computing Dataset Stats...")
    train_ds = trainer.train_dataset
    len_train_ds = sum(1 for _ in train_ds)
    print(
        f"  len(train_ds): {len_train_ds}\n"
    )
    train_dl = trainer.get_train_dataloader()
    train_dl.dataset.infinite = False
    len_train_dl = sum(1 for _ in train_dl)
    b = next(iter(train_dl))
    input_ids, labels = b["input_ids"], b["labels"]
    
    print(
        f"  len(train_dl): {len_train_dl}\n"
        f"  batch_shape  : {input_ids.shape}\n"
    )
    tokenizer = trainer.tokenizer
    decoded_ids = tokenizer.decode(input_ids[0])[0:80]
    decoded_labels = tokenizer.decode(labels[0])[0:80]
    print("First batch:\n"
          f"input_ids:\n{decoded_ids}\n"
          f"labels:\n{decoded_labels}\n")


def _prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n").format_map(row)

def _prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n").format_map(row)

def create_alpaca_prompt(row):
    return _prompt_no_input(row) if row["input"] == "" else _prompt_input(row)

def create_alpaca_prompt_with_response(row):
    instruct = _prompt_no_input(row) if row["input"] == "" else _prompt_input(row)
    return instruct + row["output"]