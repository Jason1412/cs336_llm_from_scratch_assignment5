# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CS 336 Spring 2025 Assignment 5 on LLM alignment. Students implement alignment algorithms (SFT, Expert Iteration, GRPO) by filling in stub functions. All implementations go in `tests/adapters.py`; the main package `cs336_alignment/` provides supporting utilities.

## Commands

```bash
# Install dependencies (flash-attn requires two-step install)
uv sync --no-install-package flash-attn
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_sft.py

# Run a single test by name
uv run pytest tests/test_grpo.py::test_compute_group_normalized_rewards

# Run tests with verbose output
uv run pytest -v

# Run with exact snapshot matching (stricter tolerances)
uv run pytest --snapshot-exact

# Create submission zip (runs tests first)
./test_and_make_submission.sh
```

## Architecture

### Implementation target: `tests/adapters.py`

All student work happens in `tests/adapters.py`. Every function raises `NotImplementedError`. The functions cover:

- **Tokenization**: `run_tokenize_prompt_and_output` — tokenizes prompt+output pairs and builds a `response_mask` marking which tokens belong to the response (not prompt/padding)
- **Reward computation**: `run_compute_group_normalized_rewards` — computes per-group normalized advantages for GRPO
- **Core primitives**: `run_compute_entropy`, `run_get_response_log_probs`, `run_masked_mean`, `run_masked_normalize`
- **Policy gradient losses**: `run_compute_naive_policy_gradient_loss`, `run_compute_grpo_clip_loss`, `run_compute_policy_gradient_loss` (dispatcher)
- **Train steps**: `run_sft_microbatch_train_step`, `run_grpo_microbatch_train_step`
- **Optional RLHF/safety** (separate supplement): `get_packed_sft_dataset`, `run_iterate_batches`, `run_parse_mmlu_response`, `run_parse_gsm8k_response`, `run_compute_per_instance_dpo_loss`

### Supporting package: `cs336_alignment/`

- `drgrpo_grader.py` — Math answer grading (symbolic verification, LaTeX parsing) used as the reward function for GRPO on GSM8K/MATH datasets
- `prompts/` — Prompt templates: `alpaca_sft.prompt`, `r1_zero.prompt`, `question_only.prompt`, `zero_shot_system_prompt.prompt`

### Test suite: `tests/`

- `conftest.py` — Fixtures: tiny GPT-2 model, tokenizer, synthetic tensors, reward functions. Also defines `NumpySnapshot` (`.npz` files) and `Snapshot` (`.pkl` files) for regression testing stored in `tests/_snapshots/`
- `test_sft.py` — Tests tokenization, entropy, log probs, masked ops, SFT train step
- `test_grpo.py` — Tests group reward normalization, policy gradient losses, GRPO-Clip
- `test_dpo.py` — Tests DPO loss (optional supplement)
- `test_data.py` — Tests packed SFT dataset, batch iteration (optional supplement)
- `test_metrics.py` — Tests MMLU/GSM8K response parsing (optional supplement)

### Snapshot testing

Tests compare outputs against pre-computed `.npz`/`.pkl` files in `tests/_snapshots/`. Default tolerance: `rtol=1e-4, atol=1e-2`. Pass `--snapshot-exact` for stricter matching. Snapshots are never regenerated automatically; they are ground-truth reference outputs.

### Data

`data/` contains benchmark datasets used in evaluation scripts (not tests): `gsm8k/`, `mmlu/`, `alpaca_eval/`, `simple_safety_tests/`. Test fixtures are in `tests/fixtures/` (tiny GPT-2, Llama tokenizer).
