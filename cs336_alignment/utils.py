import rich
import torch
import json


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
