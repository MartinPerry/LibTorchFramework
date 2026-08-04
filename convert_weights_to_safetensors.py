import argparse
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, torch.nn.Module):
        checkpoint = checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

    for key in ("state_dict", "model_state_dict", "model", "weights"):
        value = checkpoint.get(key)

        if isinstance(value, torch.nn.Module):
            checkpoint = value.state_dict()
            break

        if isinstance(value, dict) and any(isinstance(item, torch.Tensor) for item in value.values()):
            checkpoint = value
            break

    state_dict = {
        str(name): tensor.detach().cpu().contiguous().clone()
        for name, tensor in checkpoint.items()
        if isinstance(tensor, torch.Tensor)
    }

    if not state_dict:
        raise ValueError("No tensors were found in the checkpoint.")

    return state_dict


def verify(original: dict[str, torch.Tensor], loaded: dict[str, torch.Tensor]) -> None:
    if original.keys() != loaded.keys():
        missing = original.keys() - loaded.keys()
        extra = loaded.keys() - original.keys()
        raise RuntimeError(
            f"Tensor name mismatch.\nMissing: {missing}\nExtra: {extra}"
        )

    for name in original:
        a = original[name]
        b = loaded[name]

        if a.dtype != b.dtype:
            raise RuntimeError(f"{name}: dtype mismatch ({a.dtype} != {b.dtype})")

        if a.shape != b.shape:
            raise RuntimeError(f"{name}: shape mismatch ({a.shape} != {b.shape})")

        if not torch.equal(a, b):
            diff = (a != b).sum().item() if a.dtype != torch.float16 else "?"
            raise RuntimeError(f"{name}: tensor data mismatch ({diff} differing elements)")

    print(f"Verification successful ({len(original)} tensors).")


def convert(input_path: Path, output_path: Path, unsafe_load: bool) -> None:
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=not unsafe_load)
    state_dict = extract_state_dict(checkpoint)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, output_path, metadata={"format": "pt"})

    loaded = load_file(str(output_path), device="cpu")
    verify(state_dict, loaded)

    print(f"Converted {len(state_dict)} tensors")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PyTorch weights to SafeTensors.")
    parser.add_argument("input", type=Path, help="Input .pt, .pth or .bin file")
    parser.add_argument("output", type=Path, nargs="?", help="Output .safetensors file")
    parser.add_argument(
        "--unsafe-load",
        action="store_true",
        help="Allow loading checkpoints containing custom Python objects. Use only with trusted files.",
    )
    args = parser.parse_args()

    output_path = args.output or args.input.with_suffix(".safetensors")
    convert(args.input, output_path, args.unsafe_load)


if __name__ == "__main__":
    main()