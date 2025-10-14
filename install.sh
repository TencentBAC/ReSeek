# pip install -r requirements-build.txt
# VLLM_TARGET_DEVICE=empty pip install -e .

cd vllm-ascend
export COMPILE_CUSTOM_KERNELS=1
sh ascend_fix.sh

cd ReSeek
pip install -r requirements-npu.txt
pip install -e .

pip install torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu


