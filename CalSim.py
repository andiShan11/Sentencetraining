# from sentence_transformers import SentenceTransformer, util
# import pickle
# import os
#
#
# # sentences = []
# path = ""
# with open(os.path.join("/home/amax/Documents/ZS/PycharmProject/sentence-transformers-train/data", "民事案例.pkl"), "rb")\
#         as fout:
#     sentences = pickle.load(fout)
#
# print(len(sentences))
#
# # for (index, item) in enumerate(sentences):
# #     print(index, item)
# #     if index == 10:
# #         break
# #     # break
#
# model = SentenceTransformer('paraphrase-distilroberta-base-v1')
# embeddings = model.encode(sentences, convert_to_tensor=True)
# cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
#
# pairs = []
# for i in range(len(cosine_scores)-1):
#     for j in range(i+1, len(cosine_scores)):
#         pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})