# Usage: python launch_script.py --cmd "python dummy_train.py --arg1=42 --arg3='hole'" --log True 
# This will execute the command python dummy_train.py --arg1=42 --arg3='hole' and log the output to W&B.
# Now this script is launchable with W&B Launch from the UI.

import wandb
import simple_parsing
from typing import Optional
from dataclasses import dataclass
import subprocess
from ast import literal_eval

def exec(cmd):
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print("Error")

def wandb_artifact_from_file(file_path: str, name: Optional[str] = None, type: Optional[str] = None):
    if name is None:
        name = file_path.split("/")[-1]
    if type is None:
        type = name
    artifact = wandb.Artifact(name=name, type=type)
    artifact.add_file(file_path)
    return artifact

def parse_config(argv):
    args, kwargs = [], {}
    for arg in argv:
        if '=' in arg:
            key, value = arg.split('=', 1)
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(value)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = value
            kwargs[key[2:]] = attempt
        else:
            args.append(arg)
    return args, kwargs

@dataclass
class LaunchOverride(simple_parsing.Serializable):
    project: str = "launch_intercept"  # the wandb project to log to
    entity: str = "capecape"           # the wandb entity to log to
    output: Optional[str] = None # Save the output back to W&B as an artifact.
    log: bool = False # Log to W&B
    cmd: str = "echo 'Hello World'"

args: LaunchOverride = simple_parsing.parse(LaunchOverride)
script_args, script_kwargs = parse_config(args.cmd.split())
print("Script Arguments:", script_args)
print("Script Keyword Arguments:", script_kwargs)
config = {"cmd_args": " ".join(script_args), "cmd_kwargs": script_kwargs}
if args.log:
    wandb.init(project=args.project, entity=args.entity, config=config)
    config = wandb.config
parsed_cmd = config["cmd_args"] + " " + " ".join([f"{k}={v}" for k,v in config["cmd_kwargs"].items()])
print("Executing:", parsed_cmd)

exec(args.cmd.split())

if args.output is not None and args.log:
    artifact = wandb_artifact_from_file(args.output_artifact)
    wandb.log_artifact(artifact)