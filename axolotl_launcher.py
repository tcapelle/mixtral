"""
CLI to run training on a model
"""
import logging, os, yaml
from pathlib import Path
import wandb
import fire
import transformers

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_datasets,
    print_axolotl_text_art,
    validate_config,
    prepare_optim_env,
    normalize_config,
    setup_wandb_env_vars
)
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.dict import DictDefault

LOG = logging.getLogger("axolotl.cli.train")


def load_cfg(config: Path = Path("examples/"), **kwargs):
    if Path(config).is_dir():
        config = choose_config(config)

    # load the config from the yaml file
    with open(config, encoding="utf-8") as file:
        cfg: DictDefault = DictDefault(yaml.safe_load(file))
    cfg.axolotl_config_path = config
    # if there are any options passed in the cli, if it is something that seems valid from the yaml,
    # then overwrite the value
    cfg_keys = cfg.keys()
    for k, _ in kwargs.items():
        # if not strict, allow writing to cfg even if it's not in the yml already
        if k in cfg_keys or not cfg.strict:
            # handle booleans
            if isinstance(cfg[k], bool):
                cfg[k] = bool(kwargs[k])
            else:
                cfg[k] = kwargs[k]

    validate_config(cfg)

    prepare_optim_env(cfg)

    normalize_config(cfg)

    setup_wandb_env_vars(cfg)
    return cfg

def do_cli(config: Path = Path("examples/"), **kwargs):
    # pylint: disable=duplicate-code
    parsed_cfg = load_cfg(config, **kwargs)
    print(f"*********Parsed Args********* \n{parsed_cfg}\n***************************")
    print_axolotl_text_art()
    check_accelerate_default_config()
    check_user_token()
    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    dataset_meta = load_datasets(cfg=parsed_cfg, cli_args=parsed_cli_args)
    if int(os.environ["RANK"]) == 0:
        print(f"We are in rank {os.environ['RANK']}, initializing wandb")
        wandb.init(project=parsed_cfg.wandb_project, entity=parsed_cfg.wandb_entity, config=parsed_cfg)
        
        # enable wandb injection of config
        parsed_cfg = DictDefault(wandb.config.as_dict())

    train(cfg=parsed_cfg, cli_args=parsed_cli_args, dataset_meta=dataset_meta)

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(do_cli)