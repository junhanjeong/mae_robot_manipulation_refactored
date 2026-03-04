import json
import os
import time

import torch
import wandb

from model.forward_model_mae import MAE_Translation
from scripts.push_trainer import PushTrainer
from transformers import ViTMAEForPreTraining


DEFAULTS = {
    "pretrain_path": "",
    "train_dir": "",
    "val_dir": "",
    "test_dir": "",
    "which_gpu": 0,
    "save_dir": "results",
    "exp_name": "todo",
    "batch_size": 16,
    "pretrained": 0,
    "lr": 1e-4,
    "net_type": "alex",
    "feat_dim": 256,
    "epochs": 200,
    "model": "policy",
    "env": "trash",
    "history": 1,
    "mult": 0,
    "mirror": 0,
    "rot": "6d-d",
    "data_size": 1.0,
    "l1": 0.0,
    "l2": 1.0,
    "l3": 0.0,
    "lg": 0.0,
    "rad": "None",
    "xyz_normed": 0,
    "train": 0,
    "grip_file": "",
    "rand": 0,
}


def build_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="path to downstream checkpoint (.pt) saved by PushTrainer",
    )
    parser.add_argument(
        "--params_json",
        type=str,
        default="",
        help="optional path to params.json from a previous downstream run",
    )

    for key, value in DEFAULTS.items():
        arg_name = f"--{key}"
        if isinstance(value, int):
            parser.add_argument(arg_name, type=int, default=None)
        elif isinstance(value, float):
            parser.add_argument(arg_name, type=float, default=None)
        else:
            parser.add_argument(arg_name, type=str, default=None)
    return parser


def merge_params(args):
    params = dict(DEFAULTS)

    if args.params_json:
        with open(args.params_json, "r") as f:
            loaded = json.load(f)
        params.update(loaded)

    for key in DEFAULTS:
        value = getattr(args, key)
        if value is not None:
            params[key] = value

    required = ["pretrain_path", "train_dir", "val_dir", "test_dir"]
    missing = [k for k in required if not params.get(k)]
    if missing:
        raise ValueError(
            "Missing required arguments (or params_json fields): "
            + ", ".join(missing)
        )
    return params


def load_checkpoint_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("Missing keys while loading checkpoint:", missing)
    if unexpected:
        print("Unexpected keys while loading checkpoint:", unexpected)

    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    params = merge_params(args)

    logdir_prefix = "trashpolicy_" + params["exp_name"] + "_" + params["net_type"] + "_eval"
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), params["save_dir"])
    os.makedirs(data_path, exist_ok=True)

    logdir = logdir_prefix + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    os.makedirs(logdir, exist_ok=True)
    params["logdir"] = logdir

    with open(os.path.join(logdir, "params.json"), "w") as outfile:
        json.dump(params, outfile, indent=4, separators=(",", ": "), sort_keys=True)
        outfile.write("\n")

    device = torch.device(
        "cuda:" + str(params["which_gpu"]) if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    print("Test-only logdir:", logdir)

    model_path = os.path.join(params["pretrain_path"], "transformer-mae-model")
    pretrained_mae = ViTMAEForPreTraining.from_pretrained(model_path)
    model = MAE_Translation(
        pretrained_mae=pretrained_mae,
        embed_dim=pretrained_mae.config.hidden_size,
        out_dim=3,
        freeze_encoder=True,
        frozen_layer_num=6,
    ).to(device)

    load_checkpoint_weights(model, args.checkpoint_path, device)

    trainer = PushTrainer(model, params)
    trainer.model.eval()
    trainer.wandb_dict = {}
    with torch.no_grad():
        trainer.run_loop(trainer.test_loader, "test", epoch=0, logdir=logdir, early_stopping=None)

    wandb.log(trainer.wandb_dict)

    print("\nFinal test metrics")
    for key in sorted(trainer.wandb_dict):
        if key.startswith("Test "):
            print(f"{key}: {trainer.wandb_dict[key]}")


if __name__ == "__main__":
    main()
