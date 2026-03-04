import json
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.PushDataset import PushDataset
from model.forward_model_mae import MAE_Translation
from transformers import ViTMAEForPreTraining


DEFAULTS = {
    "pretrain_path": "mae_checkpoint_6",
    "test_dir": "/home/junhan/data/test",
    "which_gpu": 0,
    "batch_size": 16,
    "mirror": 0,
    "data_size": 1.0,
    "rad": "None",
    "xyz_normed": 0,
    "rand": 0,
    "num_workers": 1,
}


def build_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/junhan/mae_robot_manipulation_refactored/results/trashpolicy_todo_alex_04-03-2026_10-26-44/policy_earlystop.pt",
        help="path to downstream checkpoint (.pt) saved by PushTrainer",
    )
    parser.add_argument(
        "--params_json",
        type=str,
        default="/home/junhan/mae_robot_manipulation_refactored/results/trashpolicy_todo_alex_04-03-2026_10-26-44/params.json",
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

    required = ["pretrain_path", "test_dir"]
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


def build_test_loader(params):
    test_set = PushDataset("test", params)
    num_workers = params["num_workers"]
    loader_kwargs = {
        "dataset": test_set,
        "batch_size": params["batch_size"],
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(**loader_kwargs)
    return loader, len(test_set)


def run_inference(model, loader, device):
    total_samples = 0
    batch_errors = []

    start = time.time()
    model.eval()
    with torch.inference_mode():
        for _, batch_imgs, batch_actions, _, _ in tqdm(
            loader, desc="Inference", total=len(loader)
        ):
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)
            pred_pos = model(batch_imgs)
            real_positions = batch_actions[:, 0]

            pred_pos = pred_pos.reshape(pred_pos.size(0), -1)
            real_positions = real_positions.reshape(real_positions.size(0), -1).float()
            batch_error = torch.mean((pred_pos - real_positions) ** 2)
            batch_errors.append(batch_error.item())

            batch_size = pred_pos.size(0)
            total_samples += batch_size

    elapsed = time.time() - start
    speed = total_samples / elapsed if elapsed > 0 else 0.0
    test_error = sum(batch_errors) / len(batch_errors) if batch_errors else float("nan")
    print(f"Test Error: {test_error:.6f}")
    print(f"Inference done: {total_samples} samples, {elapsed:.2f}s, {speed:.2f} samples/s")


def main():
    parser = build_parser()
    args = parser.parse_args()
    params = merge_params(args)

    device = torch.device(
        "cuda:" + str(params["which_gpu"]) if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    print(
        f"DataLoader config: batch_size={params['batch_size']}, "
        f"num_workers={params['num_workers']}"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

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

    test_loader, num_test_samples = build_test_loader(params)
    print(f"Loaded test samples: {num_test_samples}")
    run_inference(
        model=model,
        loader=test_loader,
        device=device,
    )


if __name__ == "__main__":
    main()
