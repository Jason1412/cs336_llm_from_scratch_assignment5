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


_VOCAB_CHUNK = 4096  # Process vocab in slices to avoid large (B, T, V) intermediates


def _chunked_log_probs_and_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    return_entropy: bool,
    chunk_size: int = _VOCAB_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute per-token log-probs (and optionally entropy) by chunking over the
    vocab dimension.

    Peak extra memory ≈ O(B * T * chunk_size) instead of O(B * T * V), which
    avoids the ~1.56 GiB float32 upcast that torch.logsumexp / F.softmax perform
    internally on bfloat16 tensors (vocab_size=152k for Qwen2.5).

    Math (numerically stable via max-subtraction):
        max_i   = max_v logit_{i,v}
        sum_exp = sum_v exp(logit_{i,v} - max_i)
        lse_i   = max_i + log(sum_exp_i)
        log p_i = logit_{i,label_i} - lse_i
        H_i     = log(sum_exp_i) - weighted_shifted_i / sum_exp_i
            where weighted_shifted_i = sum_v exp(logit-max) * (logit-max)
    """
    V = logits.size(-1)
    batch_shape = logits.shape[:-1]  # (B, T)

    target_logits = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    max_logits = logits.max(dim=-1).values  # (B, T)

    sum_exp = torch.zeros(batch_shape, dtype=logits.dtype, device=logits.device)
    weighted_shifted = (
        torch.zeros(batch_shape, dtype=logits.dtype, device=logits.device)
        if return_entropy
        else None
    )

    for start in range(0, V, chunk_size):
        chunk = logits[..., start : start + chunk_size]      # (B, T, C) — view, no copy
        shifted = chunk - max_logits.unsqueeze(-1)           # (B, T, C)
        exp_shifted = shifted.exp()                          # (B, T, C)
        sum_exp += exp_shifted.sum(dim=-1)                   # (B, T)
        if return_entropy:
            weighted_shifted += (exp_shifted * shifted).sum(dim=-1)

    lse = max_logits + sum_exp.log()     # (B, T)
    log_probs = target_logits - lse      # (B, T)

    entropy = None
    if return_entropy:
        # H = log(sum_exp) - weighted_shifted / sum_exp
        entropy = sum_exp.log() - weighted_shifted / sum_exp

    return log_probs, entropy


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits: torch.Tensor, shape = (..., vocab_size)
    Returns per-position entropy, chunked over vocab to avoid materialising the
    full (B, T, V) softmax tensor.
    """
    dummy_labels = torch.zeros(
        logits.shape[:-1], dtype=torch.long, device=logits.device
    )
    _, entropy = _chunked_log_probs_and_entropy(
        logits, dummy_labels, return_entropy=True
    )
    return entropy


class _FusedLMHeadLogProb(torch.autograd.Function):
    """Fused lm_head + per-token log-prob that never materialises (B*T, V).

    Forward  : two passes over vocab chunks → O(B*T*C) peak per chunk.
    Backward : recomputes logit chunks, accumulates grad_weight IN-PLACE into
               weight.grad, returns only grad_hidden (B*T × H, tiny).
               Peak extra memory per chunk ≈ 3 × (B*T × C) ≈ 36 MiB.
    """

    @staticmethod
    def forward(ctx, hidden, labels, weight, bias, chunk_size):
        # hidden : (B*T, H),  labels : (B*T,)
        # weight : (V, H),    bias   : (V,) or None
        B_T, H = hidden.shape
        V = weight.shape[0]
        dev, dt = hidden.device, hidden.dtype

        # Pass 1 – per-token max for numerical stability
        max_l = torch.full((B_T,), float("-inf"), device=dev, dtype=dt)
        for s in range(0, V, chunk_size):
            c = F.linear(hidden, weight[s : s + chunk_size],
                         None if bias is None else bias[s : s + chunk_size])
            max_l = torch.maximum(max_l, c.max(dim=-1).values)

        # Pass 2 – sum_exp and target logit
        sum_exp = torch.zeros(B_T, device=dev, dtype=dt)
        tgt = torch.zeros(B_T, device=dev, dtype=dt)
        for s in range(0, V, chunk_size):
            e = min(s + chunk_size, V)
            c = F.linear(hidden, weight[s:e],
                         None if bias is None else bias[s:e])
            sum_exp += (c - max_l.unsqueeze(-1)).exp().sum(-1)
            in_c = (labels >= s) & (labels < e)
            if in_c.any():
                loc = (labels - s).clamp(0, e - s - 1)
                tgt = torch.where(in_c, c.gather(1, loc.unsqueeze(1)).squeeze(1), tgt)

        lse = max_l + sum_exp.log()
        log_probs = tgt - lse

        ctx.save_for_backward(hidden, labels, lse)
        # Store weight/bias as Python attrs (already in GPU memory as params)
        ctx.weight = weight
        ctx.bias = bias
        ctx.chunk_size = chunk_size
        return log_probs

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out : (B*T,) — may be float32 if the GRPO loss mixes dtypes
        hidden, labels, lse = ctx.saved_tensors
        weight, bias = ctx.weight, ctx.bias
        chunk_size = ctx.chunk_size
        V = weight.shape[0]
        dt = hidden.dtype

        # Cast incoming gradient to the native dtype of hidden/weight (bfloat16).
        # The GRPO loss computes bfloat16 log_probs through float32 advantages,
        # which causes grad_out to arrive as float32.
        grad_out = grad_out.to(dt)

        grad_hidden = torch.zeros_like(hidden)  # (B*T, H) ← ~5 MiB

        # Initialise .grad buffers if this is the first microbatch
        with torch.no_grad():
            if weight.grad is None:
                weight.grad = torch.zeros_like(weight)
            if bias is not None and bias.grad is None:
                bias.grad = torch.zeros_like(bias)

        go = grad_out.unsqueeze(-1)  # (B*T, 1)
        for s in range(0, V, chunk_size):
            e = min(s + chunk_size, V)
            w_c = weight[s:e]                                        # (C, H) view
            b_c = None if bias is None else bias[s:e]
            logits_c = F.linear(hidden, w_c, b_c)                   # (B*T, C) recompute
            softmax_c = (logits_c - lse.unsqueeze(-1)).exp()        # (B*T, C)
            grad_c = softmax_c * go                                  # (B*T, C)

            in_c = (labels >= s) & (labels < e)
            if in_c.any():
                idx = in_c.nonzero(as_tuple=True)[0]
                grad_c[idx, labels[idx] - s] -= grad_out[idx]

            grad_hidden.addmm_(grad_c, w_c)   # (B*T,H) += (B*T,C)@(C,H)
            with torch.no_grad():
                weight.grad[s:e].addmm_(grad_c.t(), hidden)  # (C,H) in-place
                if bias is not None:
                    bias.grad[s:e] += grad_c.sum(0)

        # Return None for weight/bias grads – accumulated manually above
        return grad_hidden, None, None, None, None


def get_response_log_probs(
    model,  # PreTrainedModel (Qwen2ForCausalLM)
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Args:
        model: PreTrainedModel
        input_ids: (batch_size, sequence_length)
        labels:    (batch_size, sequence_length)
        return_token_entropy: bool

    Memory design
    -------------
    Qwen2 casts logits to float32 internally and cross_entropy backward requires
    a (B*T, V) softmax tensor simultaneously with the stored logits – peak ~892 MiB
    that does not fit after optimizer states are created.

    Instead we call model.model() to get hidden_states (B, T, H), then use
    _FusedLMHeadLogProb which chunks the lm_head projection and backward over the
    vocab dimension.  Peak extra memory per chunk ≈ 3 × (B*T × 4096) ≈ 36 MiB.
    """
    # Run transformer body (gradient checkpointing active inside)
    hidden = model.model(input_ids)[0]          # (B, T, H)
    B, T, H = hidden.shape

    weight = model.lm_head.weight               # (V, H) – model param
    bias = getattr(model.lm_head, "bias", None) # None for Qwen2

    log_probs = _FusedLMHeadLogProb.apply(
        hidden.view(B * T, H),
        labels.view(B * T),
        weight,
        bias,
        _VOCAB_CHUNK,
    ).view(B, T)

    res = {"log_probs": log_probs}

    if return_token_entropy:
        with torch.no_grad():
            _, token_entropy = _chunked_log_probs_and_entropy(
                # recompute logits chunked for entropy (no grad needed)
                F.linear(hidden.detach().view(B * T, H), weight).view(B, T, -1),
                labels,
                return_entropy=True,
            )
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
