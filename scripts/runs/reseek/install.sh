pip install transformers datasets pyserini

export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com,127.0.0.1 no_proxy=localhost,.woa.com,.oa.com,.tencent.com,127.0.0.1 
export {HTTP,HTTPS}_PROXY=$ENV_VENUS_PROXY {http,https}_proxy=$ENV_VENUS_PROXY
## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi sentence_transformers==3.3.1