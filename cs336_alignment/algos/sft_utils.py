import torch
import torch.nn.functional as F


def tokenizer_prompt_and_output(
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
