import rich
import torch
import json
from contextlib import nullcontext
import gc


def wrap_cot_with_answer(cot: str, answer: str) -> str:
    return f"{cot}\n</think> <answer>{str(answer)}</answer>"


def get_device(
    verbose: bool = True, rank: int = 0, use_mps: bool = True
) -> torch.device:
    if torch.cuda.is_available():
        if verbose:
            print_color(f"Using CUDA device cuda:{rank}", "blue")
        return torch.device(f"cuda:{rank}")
    elif use_mps and torch.backends.mps.is_available():
        if verbose:
            print_color("Using MPS device", "blue")
        return torch.device("mps")
    else:
        if verbose:
            print_color("Using CPU device", "blue")
        return torch.device("cpu")


def print_color(text: str, color: str = "red"):
    rich.print(f"[{color}]{text}[/{color}]")


def print_rich_dict(data: dict) -> None:
    from rich.pretty import pprint

    """Pretty print dictionary with colors using rich."""
    pprint(data, expand_all=True)


def load_dataset(path: str, prompt_template_path: str = ""):
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        prompt_template = f.read().strip()

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    prompts = []
    cots = []
    answers = []
    for row in rows:
        prompts.append(prompt_template.format(question=row["question"]))
        cots.append(row["cot"])
        answers.append(row["answer"])

    return prompts, cots, answers


def cycle_dataloader(data_loader):
    while True:
        for batch in data_loader:
            yield batch


def get_ctx(use_mixed: bool, device: torch.device, verbose: bool = True):
    if use_mixed and device.type == "cuda":
        if verbose:
            print_color("Using mixed precision on CUDA with BFloat16", "blue")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif use_mixed and device.type == "mps":
        if verbose:
            print_color("Using mixed precision on MPS with Float16", "blue")
        return torch.autocast(device_type="mps", dtype=torch.float16)
    elif use_mixed and device.type == "cpu":
        if verbose:
            print_color("Using mixed precision on CPU with Float16", "blue")
        return torch.autocast(device_type="cpu", dtype=torch.float16)
    else:
        if verbose:
            print_color("Not using mixed precision", "blue")
        return nullcontext()


def print_color(content: str, color: str = "green"):
    print(f"[{color}]{content}[/{color}]")


def to_float(x):
    if isinstance(x, torch.Tensor):
        return x.float().item()
    elif isinstance(x, str):
        return float(x.strip())

    return float(x)


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
