from logging import config
import os
import warnings
import importlib
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from twoxplat.utils.init_utils import init_config
from twoxplat.utils.metric_utils import export_results, summarize_evaluation

warnings.filterwarnings('ignore', category=FutureWarning)

AMP_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}


# ---------------------------------------------------------------------------
# Dataset & dataloader
# ---------------------------------------------------------------------------

def build_dataloader(config: DictConfig) -> DataLoader:
    """Instantiate the inference dataset and wrap it in a DataLoader."""
    assert config.inference.batch_size_per_gpu == 1, \
        "Currently only batch size of 1 is supported for streaming inference."

    dataset_name = config.inference.get("dataset_name", "twoxplat.datasets.dataset.InferenceDataset")
    module_name, class_name = dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(module_name).__dict__[class_name]
    dataset = Dataset(config)

    return DataLoader(
        dataset,
        batch_size=config.inference.batch_size_per_gpu,
        shuffle=False,
        num_workers=config.inference.num_workers,
        prefetch_factor=config.inference.prefetch_factor,
        persistent_workers=True,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def load_model(config: DictConfig, device: torch.device) -> torch.nn.Module:
    """Instantiate the model and load checkpoint weights."""
    module_name, class_name = config.model.class_name.rsplit(".", 1)
    ModelClass = importlib.import_module(module_name).__dict__[class_name]
    model = ModelClass(config).to(device)
    print(model.load_ckpt(config.inference.ckpt_path))
    return model


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(
    config: DictConfig,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_context: torch.autocast,
) -> None:
    """Run inference over the full dataloader and export results."""
    print(f"Running inference; saving results to: {config.inference.out_dir}")

    model.eval()
    with torch.no_grad(), amp_context:
        for idx, batch in enumerate(dataloader):
            print(f"Processing batch {idx + 1}/{len(dataloader)}")
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            input_data = {
                k: v[:, :config.data.num_input_frames] if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            target_data = {
                k: v[:, config.data.num_input_frames:] if isinstance(v, torch.Tensor) else None
                for k, v in batch.items()
            }
            result = model(input_data, target_data)
            export_results(
                result,
                config.inference.out_dir,
                compute_metrics=config.inference.get("compute_metrics"),
                visualize=config.inference.get("visualize", False),
                uid=idx + 1,
            )
        torch.cuda.empty_cache()

    if config.inference.get("compute_metrics", False):
        summarize_evaluation(config.inference.out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Initialize config, model, and dataloader, then run inference."""
    config = init_config()

    os.environ["OMP_NUM_THREADS"] = str(config.inference.get("num_threads", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = config.inference.use_tf32
    torch.backends.cudnn.allow_tf32 = config.inference.use_tf32

    amp_context = torch.autocast(
        enabled=config.inference.use_amp,
        device_type="cuda",
        dtype=AMP_DTYPE[config.inference.amp_dtype],
    )

    dataloader = build_dataloader(config)
    model = load_model(config, device)

    run_inference(config, model, dataloader, device, amp_context)


if __name__ == "__main__":
    main()
