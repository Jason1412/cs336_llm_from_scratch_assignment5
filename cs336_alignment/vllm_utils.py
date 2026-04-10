import traceback
from unittest.mock import patch

import torch
from vllm import LLM
from transformers import PreTrainedModel
from vllm.model_executor.utils import set_random_seed as vllm_set_random_seed


def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        try:
            return LLM(
                model=model_id,
                device=device,
                dtype=torch.float16,
                enable_prefix_caching=True,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=True,
            )
        except Exception as e:
            print(
                f"[init_vllm] Failed to initialize vLLM.\n"
                f"  model_id            : {model_id}\n"
                f"  device              : {device}\n"
                f"  gpu_memory_util     : {gpu_memory_utilization}\n"
                f"  Error ({type(e).__name__}): {e}\n"
                f"\nFull traceback:\n{traceback.format_exc()}"
            )
            raise


def generate_responses(vllm: LLM, prompts: list[str], sampling_params) -> list[str]:
    outputs = vllm.generate(
        prompts,
        sampling_params=sampling_params,
    )

    responses = [output.outputs[0].text for output in outputs]
    return responses


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    # Move to CPU to avoid cross-GPU OOM when loading into vLLM's GPU.
    # Avoids allocating a temporary GPU buffer for the device-to-device copy.
    cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(cpu_state_dict.items())
