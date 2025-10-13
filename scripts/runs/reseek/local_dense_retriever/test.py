# import numpy as np
# d = 64                           # dimension
# nb = 100000                      # database size
# nq = 10000                       # nb of queries
# np.random.seed(1234)             # make reproducible
# xb = np.random.random((nb, d)).astype('float32')
# xb[:, 0] += np.arange(nb) / 1000.
# xq = np.random.random((nq, d)).astype('float32')
# xq[:, 0] += np.arange(nq) / 1000.
# import faiss                   # make faiss available
# index = faiss.IndexFlatL2(d)   # build the index
# print(index.is_trained)
# index.add(xb)                  # add vectors to the index
# print(index.ntotal)

# k = 4                          # we want to see 4 nearest neighbors
# D, I = index.search(xb[:5], k) # sanity check
# print(I)
# print(D)
# D, I = index.search(xq, k)     # actual search
# print(I[:5])                   # neighbors of the 5 first queries
# print(I[-5:])                  # neighbors of the 5 last queries
import requests
import json

# 服务器地址
url = "http://localhost:8000/retrieve"

# 请求数据
data = {
    "queries": ["什么是Python?", "告诉我关于神经网络"],
    "topk": 3,
    "return_scores": True
}

# 发送POST请求
response = requests.post(url, json=data)

# 打印结果
if response.status_code == 200:
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print(f"请求失败: {response.status_code}")