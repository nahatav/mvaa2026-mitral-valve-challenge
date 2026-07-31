FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV MVAA_INPUT_DIR=/input
ENV MVAA_OUTPUT_DIR=/output
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV TORCH_CUDA_ARCH_LIST="7.0"

WORKDIR /workspace

# --no-deps on monai/smp/timm is critical: they would otherwise pull in a
# newer torch/torchvision as a transitive dependency, silently replacing the
# base image's known V100(sm_70)-compatible torch 2.5.1+cu124 build with one
# that may drop support for that older GPU architecture. Their own
# non-torch sub-dependencies (huggingface-hub, safetensors, etc.) are safe to
# resolve normally since none of them depend on torch.
RUN pip install --no-cache-dir --no-deps monai==1.6.0 segmentation-models-pytorch==0.5.0 timm \
 && pip install --no-cache-dir huggingface-hub safetensors tqdm pyyaml numpy scipy nibabel pillow \
 && python -c "import torch; assert torch.__version__.startswith('2.5.1'), torch.__version__; print('torch OK', torch.__version__)" \
 && python -c "import segmentation_models_pytorch, monai, timm; print('imports OK')"

COPY src/ /workspace/src/
COPY weights/ /workspace/weights/
COPY run.sh /workspace/run.sh
RUN chmod +x /workspace/run.sh

ENTRYPOINT ["/workspace/run.sh"]
