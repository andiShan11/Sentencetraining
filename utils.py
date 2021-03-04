import os
import pickle
from sentence_transformers import SentenceTransformer, util

path = "/home/yf/Documents/zs/民事案例_关键句子"
root_dir = os.path.abspath(os.path.dirname(__file__))
# print(root_dir)
dataset_name = "关键句子-400"


def calLen(path):
    count = len(os.listdir(path))
    ll = 0
    for file in os.listdir(path):
        with open(os.path.join(path, file), "r", encoding="utf-8") as fin:
            ll += len(fin.read())
    return ll / count


def readSentence(path):
    sentences = []
    for (index, file) in enumerate(os.listdir(path)):
        with open(os.path.join(path, file), "r", encoding="utf-8") as fin:
            sentences.append(fin.read())

        if index > 200:
            break
    return sentences


sentences = readSentence(path)
# model = SentenceTransformer('paraphrase-distilroberta-base-v1')
# embeddings = model.encode(sentences, convert_to_tensor=True)
# cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
#
# pairs = []
# for i in range(len(cosine_scores)-1):
#     for j in range(i+1, len(cosine_scores)):
#         pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
#
# pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)


with open(os.path.join(root_dir, 'data', f'{dataset_name}.pkl'), 'wb') as fout:
    pickle.dump(sentences, fout)









