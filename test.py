import os
import pickle

# with open(os.path.join(
#             "/home/yf/Documents/zs/pycharmProject/sentence-transformer-training/data",
#             "相似案例.pkl"), "rb") as fout:
#     if fout:
#         pairs = pickle.load(fout)
#         print("成功读取")
#
# print(type(pairs))
# print(len(pairs))

# print(os.path.dirname(__file__))
# import torch
# print(torch.cuda.is_available())

# from sentence_transformers import SentenceTransformer, models, InputExample, losses, util
# import torch.nn as nn
#
#
# word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=512)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
#                            out_features=256,
#                            activation_function=nn.Tanh())
#
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
#
# sentence1 = "谢晓凤在三台县人民医院处因剖宫产行持续硬膜外麻醉后，发生右下肢功能障碍系神经源性损害，出现运动、感觉功能障碍，" \
#             "不排除麻醉穿刺过程中的机械损伤，其损害后果与三台县人民医院的医疗行为之间具有一定的因果关系，" \
#             "医疗损害的参与度酌定为50%"
#
# sentence2 = "黑龙江新讼司法鉴定中心司法鉴定意见书复印件一份（与原件核对无异），鉴定意见证明谢涛涛的人工流产与医院的医疗行为存在" \
#             "因果关系，医院在未查清原告怀孕情况下用药有过错，责任参与度为70%，医疗终结期及误工费为一个月"
# sent_embed1 = model.encode(sentence1, convert_to_tensor=True)
# sent_embed2 = model.encode(sentence2, convert_to_tensor=True)
# score = util.pytorch_cos_sim(sent_embed1, sent_embed2)
# print(score)

import torch
# from sentence_transformers import SentenceTransformer, InputExample
# from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
# from datetime import datetime
# import random
#
# def data_spilt(examples, ratio1, ratio2, shuffle=False):
#     ll = len(examples)
#     offset1 = ll*ratio1
#     # if offset1 == 0 or offset1 < 1:
#     #     return [], examples
#     offset2 = ll*ratio2
#
#     if shuffle:
#         random.shuffle(examples)
#     train_examples = examples[:int(offset1)]
#     test_examples = examples[int(offset1):int(offset1+offset2)]
#     dev_examples = examples[int(offset1+offset2):]
#     return train_examples, test_examples, dev_examples
# with open(os.path.join(
#             "/home/yf/Documents/zs/pycharmProject/sentence-transformer-training/data",
#             "关键句子-100.pkl"), "rb") as fout:
#     sentences = pickle.load(fout)
#
# with open(os.path.join(
#             "/home/yf/Documents/zs/pycharmProject/sentence-transformer-training/data",
#             "相似案例.pkl"), "rb") as fout:
#     inputs = pickle.load(fout)
#
# examples = []
#
# for input in inputs:
#     examples.append(InputExample(texts=[sentences[input['index'][0]],
#                                         sentences[input['index'][1]]],
#                                  label=input['score']))
#
# train_examples, test_examples, dev_examples = data_spilt(examples, ratio1=0.6, ratio2=0.2, shuffle=True)
# model_name = "distilbert-base-uncased"
# model_save_path = "/home/yf/Documents/zs/pycharmProject/sentence-transformer-training/result/" \
#                   "train-distilbert-base-uncased-2021-03-01_21-49-08.pth"
# model = SentenceTransformer(model_save_path)
# test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name='sts-test')
# test_evaluator(model, output_path=model_save_path)
#
#
# # from sentence_transformers import util
# # modelPath = "/home/yf/Documents/zs/pycharmProject/sentence-transformer-training/result/model.pth"
# # model = torch.load(modelPath)
# #
# # sentence1 = "谢晓凤在三台县人民医院处因剖宫产行持续硬膜外麻醉后，发生右下肢功能障碍系神经源性损害，出现运动、感觉功能障碍，" \
# #             "不排除麻醉穿刺过程中的机械损伤，其损害后果与三台县人民医院的医疗行为之间具有一定的因果关系，" \
# #             "医疗损害的参与度酌定为50%"
# #
# # sentence2 = "黑龙江新讼司法鉴定中心司法鉴定意见书复印件一份（与原件核对无异），鉴定意见证明谢涛涛的人工流产与医院的医疗行为存在" \
# #             "因果关系，医院在未查清原告怀孕情况下用药有过错，责任参与度为70%，医疗终结期及误工费为一个月"
# # sent_embed1 = model.encode(sentence1, convert_to_tensor=True)
# # sent_embed2 = model.encode(sentence2, convert_to_tensor=True)
# # score = util.pytorch_cos_sim(sent_embed1, sent_embed2)
# # print(score)

#
# def gen_data(Path):
#     sentences, labels = [], []
#     for (index, file) in enumerate(os.listdir(Path)):
#         with open(os.path.join(Path, file), "r", encoding="utf-8") as fout:
#             lines = fout.readlines()
#             for line in lines:
#                 content = line.split("     ")
#                 if len(content) < 2:
#                     continue
#                 sentences.append(content[1])
#                 labels.append(content[0])
#     return sentences, labels

# gen_data()


# Path = "/home/yf/Documents/zs/关键句子_训练集"
# sentences, labels = gen_data(Path)
# print(len(sentences), len(labels))
# print(sentences[1], labels[1])

# text = '0     2015年6月19日，受害人去了医院，但没找到主治医生谢鹏，中午时医生回电告知去体检\n'
# text = text.split("\t\t\t")
# # print(len(text.split("     ")))
# print(text[0], text[1])


# import torch
# # n_gpu = torch.cuda.device_count()
# # print(n_gpu)

# from pathlib import Path
#
# BERT_PRETRAINED_PATH = Path('chinese_L-12_H-768_A-12/')
# for file in os.listdir(BERT_PRETRAINED_PATH):
#     print(file)

for