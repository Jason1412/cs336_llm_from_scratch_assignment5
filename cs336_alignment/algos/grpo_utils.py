import os
import random
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import wandb
from tqdm import trange
from transformers import AutoTokenizer, PreTrainedModel
from vllm import SamplingParams

from cs336_alignment.algos.sft_utils import (
    get_response_log_probs,
    log_generation,
    tokenize_prompt_and_output,
)

from cs336_alignment.base_config import BaseConfig
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.eval import evaluate_responses
from cs336_alignment.utils import (
    clear_memory,
    get_ctx,
    load_dataset,
    print_color,
    print_rich_dict,
    to_float,
)
from cs336_alignment.vllm_utils import (
    generate_responses,
    load_policy_into_vllm_instance,
)

REWARD_FN_MAP = {
    "r1_zero_reward_fn": r1_zero_reward_fn,
    "question_only_reward_fn": question_only_reward_fn,
}


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """
        Args:

        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
        the ground truths, producing a dict with keys "reward", "format_reward", and
        "answer_reward".

        rollout_responses: list[str], Rollouts from the policy. The length of this list is
        rollout_batch_size = n_prompts_per_rollout_batch * group_size.

        repeated_ground_truths: list[str] The ground truths for the examples. The length of this
        list is rollout_batch_size, because the ground truth for each example is repeated
        group_size times.

        group_size: int Number of responses per question (group).

        advantage_eps: float Small constant to avoid division by zero in normalization.

        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
        subtract only the group mean.

        Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
            advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
            response.

            raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout
    response.

    metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """

    rewards = []
    format_rewards = []
    answer_rewards = []

    for response, groud_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, groud_truth)
        rewards.append(reward_dict["reward"])
        format_rewards.append(reward_dict["format_reward"])
        answer_rewards.append(reward_dict["answer_reward"])

    advs = []
    for i in range(0, len(rewards), group_size):
        group_rewards = rewards[i : i + group_size]
        group_rewards_tensor = torch.tensor(group_rewards)
        group_mean_rewards = torch.mean(group_rewards_tensor)
        if normalize_by_std:
            group_std_rewards = torch.std(group_rewards_tensor)
            group_advs = (group_rewards_tensor - group_mean_rewards) / (
                group_std_rewards + advantage_eps
            )
        else:
            group_advs = group_rewards_tensor - group_mean_rewards
        advs.extend(group_advs.tolist())

    metadata = {
        "format_rewards": format_rewards,
        "answer_rewards": answer_rewards,
        "rewards": rewards,
    }
    return advs, rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.

    Args:

    raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
    reward/advantage for each rollout response.

    policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
    each token.

    Returns:

    torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
    be aggregated across the batch and sequence dimensions in the training loop).
    """

    return -(raw_rewards_or_advantages * policy_log_probs)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
    advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.

    policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
    probs from the policy being trained.

    old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
    from the old policy.

    cliprange: float Clip parameter ϵ (e.g. 0.2).

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss: torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
    loss.
        metadata: dict containing whatever you want to log. We suggest logging whether each
    token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
    the min was lower than the LHS.
    """

    ratio = torch.exp(policy_log_probs - old_log_probs)

    clamped_ratio = torch.clamp(ratio, min=1 - cliprange, max=1 + cliprange)

    was_clamped = (ratio != clamped_ratio).float()

    loss = -torch.minimum(advantages * ratio, advantages * clamped_ratio)

    tag_RHS_lower = (advantages * clamped_ratio < advantages * ratio).float()

    metadata = {
        "tag_RHS_lower": tag_RHS_lower,
        "was_clamped": was_clamped,
    }

    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.

    Args:
    policy_log_probs: (batch_size, sequence_length), per-token log-probabilities from the
    policy being trained.

    loss_type: One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".

    raw_rewards: Required if loss_type == "no_baseline"; shape (batch_size, 1).

    advantages: Required for "reinforce_with_baseline" and "grpo_clip"; shape
    (batch_size, 1).

    old_log_probs: Required for "grpo_clip"; shape (batch_size, sequence_length).

    cliprange: Required for "grpo_clip"; scalar ϵ used for clipping.

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss: (batch_size, sequence_length), per-token loss.
        metadata: dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    if loss_type == "no_baseline" and raw_rewards is None:
        raise ValueError("raw_rewards is required for no_baseline loss type")
    if loss_type == "reinforce_with_baseline" and advantages is None:
        raise ValueError("advantages is required for reinforce_with_baseline loss type")
    if loss_type == "grpo_clip" and (
        advantages is None or old_log_probs is None or cliprange is None
    ):
        raise ValueError(
            "advantages, old_log_probs, and cliprange are required for grpo_clip loss type"
        )

    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
        return loss, metadata
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
        return loss, metadata
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
        return loss, metadata
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where
    mask == 1.
    Args:
    tensor: torch.Tensor The data to be averaged.
    mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
    dim: int | None Dimension over which to average. If None, compute the mean over all
    masked elements.
    Returns:
    torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum() / mask.sum()
    else:
        return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:

    policy_log_probs: (batch_size, sequence_length), per-token log-probabilities from the
    policy being trained.

    response_mask: (batch_size, sequence_length), 1 for response tokens, 0 for
    prompt/padding.

    gradient_accumulation_steps: Number of microbatches per optimizer step.

    loss_type: One of "no_baseline", "reinforce_with_baseline", "grpo_clip".

    raw_rewards: Needed when loss_type == "no_baseline"; shape (batch_size, 1).

    advantages: Needed when loss_type != "no_baseline"; shape (batch_size, 1).

    old_log_probs: Required for GRPO-Clip; shape (batch_size, sequence_length).

    cliprange: Clip parameter ϵ for GRPO-Clip.

    Returns:

    tuple[torch.Tensor, dict[str, torch.Tensor]].

    loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
    this so we can log it.

    metadata: Dict with metadata from the underlying loss call, and any other statistics you
    might want to log.
    """

    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    masked_loss = masked_mean(
        tensor=loss,
        mask=response_mask,
        dim=-1,
    )

    avg_masked_loss = masked_loss.mean()
    scaled_loss = avg_masked_loss / gradient_accumulation_steps
    scaled_loss.backward()

    return scaled_loss, metadata


## ------- Dataset Utils ------ ##
def sample_batch_questions(
    prompts: list[str],
    answers: list[str],
    batch_size: int,
    group_size: int = 8,
) -> tuple[list[str], list[str]]:
    index = random.sample(range(len(prompts)), k=batch_size)
    sampled_prompts = [prompts[i] for i in index]
    sampled_answers = [answers[i] for i in index]

    batch_prompts = []
    batch_answers = []
    for p, a in zip(sampled_prompts, sampled_answers):
        batch_prompts.extend([p] * group_size)
        batch_answers.extend([a] * group_size)

    return batch_prompts, batch_answers


@dataclass
class GRPOTrainConfig(BaseConfig):
    n_grpo_cur_steps: int = 200
    rollout_batch_size: int = 256
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    group_size: int = 8

    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 128

    reward_fn: Literal["r1_zero_reward_fn"] = "r1_zero_reward_fn"
    cliprange: float = 0.2
    norm_by_std: bool = True

    # Optimizer hyperparameters
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = (
        "grpo_clip"
    )
    betas: tuple = field(default=(0.9, 0.95))
    weight_decay: float = 0.0
    max_lr: float = 5e-6
    max_grad_norm: float = 1.0

    # Sampling hyperparameters
    sampling_temperature: float = 1.0
    sampling_max_tokens: int = 1024
    sampling_min_tokens: int = 4
    sampling_top_p: float = 1.0
    sampling_stop_tokens: list[str] = field(default_factory=lambda: ["</answer"])

    # Others
    mixed_precision_training: bool = True
    eval_interval: int = 5
    checkpoint_interval: int = 50
    checkpoint_dir: str = "./checkpoints/grpo"
    seed: int = 42

    def __post_init__(self):
        self.run_name = f"grpo_dataset({self.dataset_name})_prompt({self.prompt_template_path.split('/')[-1]})_reward({self.reward_fn})_loss_type({self.loss_type})"

        assert self.rollout_batch_size % self.group_size == 0, (
            "rollout_batch_size must be divisible by group_size"
        )
        # For each micro_batch, calculate the gradient once
        self.micro_batch_size = (
            self.train_batch_size // self.gradient_accumulation_steps
        )
        # Number of prompts to process in each group
        self.n_prompts_per_rollout_batch = self.rollout_batch_size // self.group_size


class GRPOTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        train_config: GRPOTrainConfig,
        device: torch.device,
        dataset_dir_base: str = "./data/pre-processed",
    ):
        self.model = model
        self.train_config = train_config
        self.device = device
        self.dataset_dir_base = dataset_dir_base

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=train_config.model_name,
            use_fast=True,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            betas=train_config.betas,
            lr=self.train_config.max_lr,
            weight_decay=self.train_config.weight_decay,
            # fused=True executes the entire param update in a single fused CUDA kernel
            # without allocating intermediate tensors like exp_avg_sq.sqrt(). This is
            # critical because lm_head.weight is massive, and its intermediates
            # consume ~446 MiB which OOMs the step.
            fused=True,
        )

        self.ctx = get_ctx(
            use_mixed=self.train_config.mixed_precision_training,
            device=device,
        )

        dataset_dir = os.path.join(dataset_dir_base, train_config.dataset_name)
        train_prompts, train_cots, train_answers = load_dataset(
            os.path.join(dataset_dir, "train.jsonl"),
            prompt_template_path=self.train_config.prompt_template_path,
        )

        self.train_prompts = train_prompts
        self.train_answers = train_answers

        test_prompts, test_cots, test_answers = load_dataset(
            os.path.join(dataset_dir, "test.jsonl"),
            prompt_template_path=self.train_config.prompt_template_path,
        )

        self.test_prompts = test_prompts
        self.test_true_answers = test_answers

        self.checkpoint_path = os.path.join(
            train_config.checkpoint_dir,
            f"grpo_{train_config.model_name.split('/')[-1]}_{train_config.dataset_name}_{train_config.reward_fn}_loss({train_config.loss_type})",
        )
        os.makedirs(self.checkpoint_path, exist_ok=True)
        train_config.to_json(
            os.path.join(self.checkpoint_path, "train_config.json"),
        )

        self.sampling_params = SamplingParams(
            temperature=self.train_config.sampling_temperature,
            max_tokens=self.train_config.sampling_max_tokens,
            top_p=self.train_config.sampling_top_p,
            min_tokens=self.train_config.sampling_min_tokens,
            include_stop_str_in_output=True,
            stop=self.train_config.sampling_stop_tokens,
        )

        self.reward_fn = REWARD_FN_MAP[self.train_config.reward_fn]

        self.grpo_cur_step = 0

    def resume_from_latest_checkpoint(self):
        """Find the latest step_N checkpoint and restore model, optimizer, and step."""
        import glob
        step_dirs = glob.glob(os.path.join(self.checkpoint_path, "step_*"))
        if not step_dirs:
            print_color("No checkpoints found, starting from scratch.", color="yellow")
            return

        # Extract step numbers and find the max
        def _step_num(path):
            base = os.path.basename(path)
            try:
                return int(base.split("_")[1])
            except (IndexError, ValueError):
                return -1

        latest = max(step_dirs, key=_step_num)
        step = _step_num(latest)
        if step < 0:
            print_color("Could not parse checkpoint step, starting from scratch.", color="yellow")
            return

        print_color(f"Resuming from checkpoint: {latest} (step {step})", color="cyan")

        # Load model weights
        from transformers import AutoModelForCausalLM
        state_dict = AutoModelForCausalLM.from_pretrained(
            latest, torch_dtype=torch.bfloat16
        ).state_dict()
        self.model.load_state_dict(state_dict)
        del state_dict
        clear_memory()

        # Load optimizer state if saved
        opt_path = os.path.join(latest, "optimizer.pt")
        if os.path.exists(opt_path):
            opt_state = torch.load(opt_path, map_location=self.device, weights_only=True)
            self.optimizer.load_state_dict(opt_state)
            del opt_state
            clear_memory()
            print_color("Restored optimizer state.", color="cyan")

        self.grpo_cur_step = step
        print_color(f"Will resume training from step {step + 1}.", color="green")

    @torch.no_grad()
    def evaluate(self, vllm=None):
        print_color(
            f"Evaluating GRPO model on test dataset at step {self.grpo_cur_step}",
            color="magenta",
        )

        overview = evaluate_responses(
            vllm=vllm,
            prompts=self.test_prompts,
            answers=self.test_true_answers,
            sampling_params=self.sampling_params,
        )

        print_color("Evaluation Overview", color="magenta")
        print_rich_dict(overview)

        return overview

    @torch.no_grad()
    def sample_responses(
        self,
        vllm=None,
        num_samples: int = 5,
    ):
        print_color(f"Sampling {num_samples} responses for GPRO model...", color="cyan")

        index = random.sample(range(len(self.test_prompts)), k=num_samples)
        prompts = [self.test_prompts[i] for i in index]
        true_answers = [self.test_true_answers[i] for i in index]

        out = log_generation(
            prompts=prompts,
            true_answers=true_answers,
            reward_fn=self.reward_fn,
            model=self.model,
            tokenizer=self.tokenizer,
            vllm=vllm,
            sampling_params=self.sampling_params,
        )

        print_rich_dict(out)

    def grpo_train_step(
        self,
        vllm,
    ):
        print_color(
            f"Sampling batch of {self.train_config.rollout_batch_size} questions ...",
            color="green",
        )
        sample_prompts, sample_answers = sample_batch_questions(
            self.train_prompts,
            self.train_answers,
            self.train_config.n_prompts_per_rollout_batch,
            self.train_config.group_size,
        )

        print_color("Generating rollout responses...", color="green")
        rollout_responses = generate_responses(
            vllm, sample_prompts, self.sampling_params
        )

        tokenized = tokenize_prompt_and_output(
            sample_prompts,
            rollout_responses,
            self.tokenizer,
        )

        print_color("Computing rewards...", color="green")
        repeated_ground_truths = sample_answers
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            reward_fn=self.reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=self.train_config.group_size,
            advantage_eps=self.train_config.advantage_eps,
            normalize_by_std=self.train_config.norm_by_std,
        )

        print_color("Computing old log probabilities...", color="green")
        input_ids = tokenized["input_ids"].to(self.device, non_blocking=True)
        labels = tokenized["labels"].to(self.device, non_blocking=True)
        response_mask = tokenized["response_mask"].to(self.device, non_blocking=True)
        ave_length = response_mask.sum(dim=1).float().mean().item()

        old_log_probs = []
        self.model.eval()
        with torch.no_grad():
            for i in trange(0, input_ids.size(0), self.train_config.micro_batch_size):
                batch_input_ids = input_ids[i : i + self.train_config.micro_batch_size]
                batch_labels = labels[i : i + self.train_config.micro_batch_size]

                with self.ctx:
                    policy_outputs = get_response_log_probs(
                        self.model,
                        input_ids=batch_input_ids,
                        labels=batch_labels,
                        return_token_entropy=False,
                    )
                    batch_log_probs = policy_outputs["log_probs"]

                old_log_probs.append(batch_log_probs.cpu())
        old_log_probs = torch.cat(old_log_probs, dim=0)
        self.model.train()

        # The 256 no-grad forward passes above leave ~600-700 MiB of fragmented
        # reserved-but-free blocks in PyTorch's caching allocator pool.  Before the
        # training backward() runs and needs a contiguous 446 MiB block, return all
        # free cached blocks to CUDA so the driver can provide a single contiguous chunk.
        torch.cuda.empty_cache()

        n_train_steps = self.train_config.epochs_per_rollout_batch * (
            self.train_config.rollout_batch_size // self.train_config.train_batch_size
        )

        print_color(f"Performing {n_train_steps} training steps...", color="green")

        batch_loss = 0.0
        token_entropy_avg = 0.0
        n_grad_steps = (
            self.train_config.train_batch_size
            // self.train_config.gradient_accumulation_steps
        )

        for train_step in range(n_train_steps):
            for micro_step in trange(
                self.train_config.gradient_accumulation_steps,
                desc="Microbatches",
            ):
                start_index = micro_step * n_grad_steps
                end_index = start_index + n_grad_steps

                micro_input_ids = input_ids[start_index:end_index]
                micro_labels = labels[start_index:end_index]
                micro_response_mask = response_mask[start_index:end_index]
                micro_advantages = torch.tensor(advantages[start_index:end_index]).to(
                    self.device
                )
                micro_raw_rewards = torch.tensor(raw_rewards[start_index:end_index]).to(
                    self.device
                )
                micro_old_log_probs = old_log_probs[start_index:end_index].to(
                    self.device
                )

                with self.ctx:
                    policy_outputs = get_response_log_probs(
                        self.model,
                        input_ids=micro_input_ids,
                        labels=micro_labels,
                        return_token_entropy=True,
                    )
                    policy_log_probs = policy_outputs["log_probs"]
                    token_entropy = policy_outputs["token_entropy"]

                micro_loss, micro_metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=micro_response_mask,
                    gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
                    loss_type=self.train_config.loss_type,
                    raw_rewards=micro_raw_rewards,
                    advantages=micro_advantages,
                    old_log_probs=micro_old_log_probs,
                    cliprange=self.train_config.cliprange,
                )

                batch_loss += to_float(micro_loss)
                token_entropy_avg += (
                    to_float(token_entropy.mean())
                    / self.train_config.gradient_accumulation_steps
                )

            print_color(
                f"GRPO Step {self.grpo_cur_step} | Train Step {train_step + 1}/{n_train_steps} |"
                f"Batch loss: {batch_loss: .4f} | Avg Token Entropy: {token_entropy_avg:.4f}",
                color="green",
            )

            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_config.max_grad_norm,
            )

            # Free fragmented reserved memory before the optimizer step.
            # AdamW's _multi_tensor_adamw allocates temp buffers (e.g. for
            # foreach_sqrt) that need contiguous CUDA blocks.
            torch.cuda.empty_cache()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        del input_ids, labels, response_mask, old_log_probs
        clear_memory()

        return {
            "train/batch_loss": batch_loss,
            "train/token_entropy_avg": token_entropy_avg,
            "train/grad_norm": grad_norm,
            "train/avg_length": ave_length,
        }

    def train(self, vllm):
        remaining = self.train_config.n_grpo_cur_steps - self.grpo_cur_step
        for _ in range(remaining):
            self.grpo_cur_step += 1
            print_color(
                f"\n=== GRPO Training Step {self.grpo_cur_step} / {self.train_config.n_grpo_cur_steps} ===",
                color="magenta",
            )

            log_dict = self.grpo_train_step(vllm)

            print_color("Loading current policy into VLLM instace...", color="green")
            load_policy_into_vllm_instance(
                self.model,
                vllm,
            )
            # state_dict() inside load_policy_into_vllm_instance temporarily
            # allocates the full model weights on GPU (~3 GB) before copying to
            # CPU.  PyTorch keeps that memory in its reserved pool; flush it
            # now so the next grpo_train_step starts with a clean slate.
            clear_memory()

            if self.grpo_cur_step % self.train_config.eval_interval == 0:
                clear_memory()

                self.sample_responses(vllm=vllm, num_samples=3)
                out = self.evaluate(vllm)
                log_dict["eval/answer_accuracy"] = out["answer_accuracy"]
                log_dict["eval/answer_correct"] = out["answer_correct"]
                log_dict["eval/format_correct"] = out["format_correct"]
                log_dict["eval/formatted_but_answer_wrong"] = out[
                    "formatted_but_answer_wrong"
                ]
                log_dict["eval/reward_1"] = out["reward_1"]

            # Save checkpoint periodically
            if self.grpo_cur_step % self.train_config.checkpoint_interval == 0:
                ckpt_dir = os.path.join(
                    self.checkpoint_path, f"step_{self.grpo_cur_step}"
                )
                print_color(
                    f"Saving checkpoint at step {self.grpo_cur_step} to {ckpt_dir}",
                    color="cyan",
                )
                clear_memory()
                self.model.save_pretrained(ckpt_dir)
                torch.save(self.optimizer.state_dict(),
                           os.path.join(ckpt_dir, "optimizer.pt"))
                clear_memory()

                # Keep only the latest 3 checkpoints
                import glob
                import shutil
                step_dirs = glob.glob(os.path.join(self.checkpoint_path, "step_*"))
                if len(step_dirs) > 3:
                    def _step_num(path):
                        base = os.path.basename(path)
                        try:
                            return int(base.split("_")[1])
                        except (IndexError, ValueError):
                            return -1
                    
                    # Sort numerically
                    step_dirs.sort(key=_step_num)
                    
                    # Delete all but the latest 3
                    for d in step_dirs[:-3]:
                        print_color(f"Removing old checkpoint: {d}", color="yellow")
                        shutil.rmtree(d, ignore_errors=True)

            wandb.log(log_dict, step=self.grpo_cur_step)
