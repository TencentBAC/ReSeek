# cd /group/40077/shyuli/gits/vllm
# pip install -r requirements-build.txt
# VLLM_TARGET_DEVICE=empty pip install -e .

cd /group/40077/shyuli/gits/vllm-ascend
export COMPILE_CUSTOM_KERNELS=1
sh ascend_fix.sh

cd /group/40077/shyuli/gits/verl
pip install -r requirements-npu.txt
pip install -e .

set_proxy
pip install torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu


