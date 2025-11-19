import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.pipelines import LongLivePipeline
from models_config import get_model_file_path, get_models_dir

# model_config will be auto-loaded from the pipeline's model.yaml
config = OmegaConf.create(
    {
        "model_dir": str(get_models_dir()),
        "generator_path": str(
            get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
        ),
        "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
        "text_encoder_path": str(
            get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
        ),
        "tokenizer_path": str(get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")),
        "height": 480,
        "width": 832,
    }
)

device = torch.device("cuda")
pipeline = LongLivePipeline(config, device=device, dtype=torch.bfloat16)

prompt_texts = [
    "A realistic video of a Texas Hold'em poker event at a casino. A male player in his late 30s with a medium build, short dark hair, light stubble, and a sharp jawline wears a fitted navy blazer over a charcoal crew-neck tee, dark jeans, and a stainless-steel watch. He sits at a well-lit poker table and tightly grips his hole cards, wearing a tense, serious expression. The table is filled with chips of various colors, the dealer is seen dealing cards, and several rows of slot machines glow in the background. The camera focuses on the player's strained concentration. Wide shot to medium close-up.",
]

outputs = []
latency_measures = []
fps_measures = []

for _, prompt_text in enumerate(prompt_texts):
    num_frames = 0
    max_output_frames = 81
    while num_frames < max_output_frames:
        start = time.time()

        prompts = [{"text": prompt_text, "weight": 100}]
        output = pipeline(prompts=prompts)

        num_output_frames, _, _, _ = output.shape
        latency = time.time() - start
        fps = num_output_frames / latency

        print(
            f"Pipeline generated {num_output_frames} frames latency={latency:2f}s fps={fps}"
        )

        latency_measures.append(latency)
        fps_measures.append(fps)
        num_frames += num_output_frames
        outputs.append(output.detach().cpu())

# Concatenate all of the THWC tensors
output_video = torch.concat(outputs)
print(output_video.shape)
output_video_np = output_video.contiguous().numpy()
export_to_video(output_video_np, str(Path(__file__).parent / "output.mp4"), fps=16)

# Print statistics
print("\n=== Performance Statistics ===")
print(
    f"Latency - Avg: {sum(latency_measures) / len(latency_measures):.2f}s, Max: {max(latency_measures):.2f}s, Min: {min(latency_measures):.2f}s"
)
print(
    f"FPS - Avg: {sum(fps_measures) / len(fps_measures):.2f}, Max: {max(fps_measures):.2f}, Min: {min(fps_measures):.2f}"
)
