import torch
from FlagEmbedding import FlagReranker

devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
print(f"Available GPUs: {devices}")

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, cache_dir='bge-reranker-v2-m3', devices=devices)

def compute_scores(sentence_pairs, normalize=False):
    return reranker.compute_score(sentence_pairs=sentence_pairs, normalize=normalize)
