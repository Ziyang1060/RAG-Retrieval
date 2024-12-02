import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from rag_retrieval import Reranker

#如果自动下载对应的模型失败，请先从huggface下载对应的模型到本地，然后这里输入本地的路径。

ranker = Reranker('/data_train/search/zengziyang/RAG-Retrieval/output/t2ranking_100_example_llm_decoder/runs/checkpoints/checkpoint_0',dtype='bf16',verbose=1)
# ranker = Reranker('/data_train/search/zengziyang/models/Qwen/Qwen2.5-7B-Instruct',dtype='bf16',verbose=1)


pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]


scores = ranker.compute_score(pairs, normalize=False)

print(scores)