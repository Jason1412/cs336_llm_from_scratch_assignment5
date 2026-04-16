import logging
import os

import dotenv
import fire
import torch
from transformers import AutoModelForCausalLM

from cs336_alignment.algos.grpo_utils import GRPOTrainConfig, GRPOTrainer
from cs336_alignment.utils import get_device, print_color, seed_everything
from cs336_alignment.vllm_utils import init_vllm


def main(
    train_config_path: str = "configs/grpo/train_r1_math.json",
    dataset_name: str = "math",
):
    logging.getLogger("vllm").setLevel(logging.WARNING)
    dotenv.load_dotenv()

    train_config = GRPOTrainConfig.from_json(train_config_path)
    train_config.dataset_name = dataset_name

    # ── vLLM initialisation (must happen before ANY CUDA call) ─────────────────
    # torch.cuda.manual_seed_all() (called inside seed_everything) initialises a
    # CUDA context on EVERY visible GPU, consuming ~1 GB per GPU.  By restricting
    # CUDA_VISIBLE_DEVICES to GPU 1 *before* any CUDA call we ensure vLLM gets the
    # full memory budget of that card.  We restore the env-var afterwards so the
    # training model can use GPU 0 normally.
    _orig_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    vllm = init_vllm(
        model_id=train_config.model_name,
        device="cuda:0",  # GPU 1 appears as cuda:0 inside the restricted env
        gpu_memory_utilization=0.85,
        seed=train_config.seed,
    )
    # Restore full GPU visibility before seeding / loading the training model.
    if _orig_cuda_visible is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _orig_cuda_visible
    vllm_device = get_device(rank=1, verbose=False)
    print_color(f"Initialized VLLM on {str(vllm_device)}", color="cyan")

    # Now safe to seed — CUDA context on GPU 1 is already established by vLLM
    seed_everything(train_config.seed)

    model_device = get_device(rank=0, verbose=False)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    model.to(model_device)
    print_color(f"Loaded model to {str(model_device)}", color="cyan")

    if train_config.wandb_logging:
        import wandb

        wandb_api = os.getenv("WANDB_API_KEY")
        if wandb_api is None:
            raise ValueError("WANDB_API_KEY not found in environment variables.")
        # wandb.login(key=wandb_api)
        wandb.init(
            project=train_config.project_name,
            name=train_config.run_name,
            config={
                "train_config": train_config.to_dict(),
            },
        )

    grpo_trainer = GRPOTrainer(
        model=model,
        train_config=train_config,
        device=model_device,
    )
    grpo_trainer.train(vllm=vllm)

    print_color("Training completed. Saving final model checkpoint...", color="green")
    # checkpoint_file = os.path.join(grpo_trainer.checkpoint_path, "checkpoint_final.pt")

    # Cleanup
    if train_config.wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
