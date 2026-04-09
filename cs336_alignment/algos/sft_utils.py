import torch
import torch.nn.functional as F

from typing import Callable
from cs336_alignment.vllm_utils import generate_responses


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer
) -> dict[str, torch.Tensor]:
    """
    Args:
        prompt_strs: list[str] List of prompt strings.
        output_strs: list[str] List of output strings.
        tokenizer: PreTrainedTokenizer from transformers.
    Returns:
        dict[str, torch.Tensor] Dictionary with keys "input_ids", "labels", "attention_mask"
    """

    prompt_tokens = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )

    output_tokens = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )

    input_ids = []
    response_masks = []
    for prompt_ids, output_ids in zip(
        prompt_tokens["input_ids"], output_tokens["input_ids"]
    ):
        input_ids.append(prompt_ids + output_ids)
        response_masks.append([False] * len(prompt_ids) + [True] * len(output_ids))

    MAX_LEN = max(len(ids) for ids in input_ids)

    def pad_to(x, value):
        return x + [value] * (MAX_LEN - len(x))

    padding_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    padded_input_ids = torch.tensor([pad_to(ids, padding_id) for ids in input_ids])
    padded_response_masks = torch.tensor(
        [pad_to(mask, False) for mask in response_masks]
    )

    assert padded_input_ids.shape == padded_response_masks.shape, (
        "Input ids and response masks must have the same shape"
    )

    input_ids = padded_input_ids[:, :-1].contiguous()
    labels = padded_input_ids[:, 1:].contiguous()
    response_mask = padded_response_masks[:, 1:].contiguous()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits: torch.Tensor, shape = (batch_size, sequence_length)

    Implementation:
        H = logsumexp(x) - Sum(Probability_i * logit_i)
    """
    lse = torch.logsumexp(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)

    sum_weighted_logits = torch.sum(probs * logits, dim=-1)
    entropy = lse - sum_weighted_logits
    return entropy


def get_response_log_probs(
    model,  # PreTrainedModel
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Args:
        model: PreTrainedModel
        input_ids: torch.Tensor, shape = (batch_size, sequence_length)
        labels: torch.Tensor, shape = (batch_size, sequence_length)
        return_token_entropy: bool = False,
    Returns:
        dict[str, torch.Tensor] Dictionary with keys "response_log_probs", "token_entropies" (optional)
    """
    logits = model(input_ids).logits

    log_all_probs = F.log_softmax(logits, dim=-1)

    log_probs = torch.gather(log_all_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(
        -1
    )

    res = {
        "log_probs": log_probs,
    }

    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        res["token_entropy"] = token_entropy

    return res


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Args:
        tensor: torch.Tensor, shape = (batch_size, sequence_length)
        mask: torch.Tensor, shape = (batch_size, sequence_length)
        normalize_constant: float, constant to avoid division by zero
        dim: int | None, dimension to normalize over
    Returns:
        torch.Tensor, normalized tensor
    """

    assert tensor.shape == mask.shape, "Tensor and mask must have the same shape"

    masked_tensor = tensor * mask

    if dim is not None:
        masked_sum = torch.sum(masked_tensor, dim=dim)
    else:
        masked_sum = torch.sum(masked_tensor)

    return masked_sum / normalize_constant


@torch.no_grad()
def log_generation(
    prompts: list[str],
    true_answers: list[str],
    reward_fn: Callable,
    model,
    tokenizer,
    vllm,
    sampling_params,
):
    device = next(model.parameters()).device
    responses = generate_responses(
        vllm,
        prompts,
        sampling_params,
    )

    reward_dicts = [reward_fn(resp, gt) for resp, gt in zip(responses, true_answers)]

    total_rewards = torch.tensor([float(d["reward"]) for d in reward_dicts])
    fmt_rewards = torch.tensor([float(d["format_reward"]) for d in reward_dicts])
    ans_rewards = torch.tensor([float(d["answer_reward"]) for d in reward_dicts])
    correct = total_rewards == 1.0

    tok = tokenize_prompt_and_output(
        prompts,
        responses,
        tokenizer,
    )
    input_ids, labels, response_mask = (
        tok["input_ids"],
        tok["labels"],
        tok["response_mask"],
    )

    model.eval()
    out = get_response_log_probs(
        model,
        input_ids=input_ids.to(device),
        labels=labels.to(device),
        return_token_entropy=True,
    )
    ent = out["token_entropy"].cpu()

    res_len = response_mask.sum(dim=1).type_as(
        total_rewards
    )  # Number of response tokens per sample
    avg_ent = (ent * response_mask.type_as(ent)).sum(dim=1) / res_len

    rows = [
        {
            "prompt": p,
            "response": r,
            "true_answer": gt,
            "total_reward": float(tr.item()),
            "format_reward": float(fr.item()),
            "answer_reward": float(ar.item()),
            "is_correct": bool(c.item()),
            "response_length": int(rl.item()),
            "avg_token_entropy": float(ae.item()),
        }
        for p, r, gt, tr, fr, ar, c, rl, ae in zip(
            prompts,
            responses,
            true_answers,
            total_rewards,
            fmt_rewards,
            ans_rewards,
            correct,
            res_len,
            avg_ent,
        )
    ]

    summary = {
        "avg_reward": float(total_rewards.float().mean().item()),
        "avg_token_entropy": float(avg_ent.mean().detach().cpu().item()),
        "avg_resp_len": float(res_len.float().mean().item()),
        "avg_len_correct": float(res_len[correct].float().mean().item())
        if correct.any()
        else 0.0,
        "avg_len_wrong": float(res_len[~correct].float().mean().item())
        if (~correct).any()
        else 0.0,
        "n_examples": len(prompts),
    }

    model.train()
    return {"summary": summary, "rows": rows}
