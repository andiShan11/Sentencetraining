from sentence_transformers import SentenceTransformer, models, InputExample, losses, util, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import torch
from torch import nn
from torch.utils.data import DataLoader
import pickle
import os
import random
import numpy as np
import logging
from datetime import datetime
import math


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def data_spilt(examples, ratio1, ratio2, shuffle=False):
    ll = len(examples)
    offset1 = ll*ratio1
    # if offset1 == 0 or offset1 < 1:
    #     return [], examples
    offset2 = ll*ratio2

    if shuffle:
        random.shuffle(examples)
    train_examples = examples[:int(offset1)]
    test_examples = examples[int(offset1):int(offset1+offset2)]
    dev_examples = examples[int(offset1+offset2):]
    return train_examples, test_examples, dev_examples


setup_seed(54123)

model_name = "distilbert-base-uncased"
train_batch_size = 10
num_epochs = 10
model_save_path = "/home/yf/Documents/zs/pycharmProject/sentence-transformer-training/result/train-"+model_name\
                  + '-' +datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Define the model

word_embedding_model = models.Transformer(model_name, max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                           out_features=256,
                           activation_function=nn.Tanh())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

# Define your train dataset, the dataloader and the train loss
logging.info("Read STSbenchmark train dataset")
with open(os.path.join(
            "/home/yf/Documents/zs/pycharmProject/sentence-transformer-training/data",
            "关键句子-100.pkl"), "rb") as fout:
    sentences = pickle.load(fout)

with open(os.path.join(
            "/home/yf/Documents/zs/pycharmProject/sentence-transformer-training/data",
            "相似案例.pkl"), "rb") as fout:
    inputs = pickle.load(fout)

examples = []

for input in inputs:
    examples.append(InputExample(texts=[sentences[input['index'][0]],
                                        sentences[input['index'][1]]],
                                 label=input['score']))

train_examples, test_examples, dev_examples = data_spilt(examples, ratio1=0.6, ratio2=0.2, shuffle=True)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model)

# evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_examples, name='sts-dev')

# Configure the training. We skip evaluation in this example
warm_steps = math.ceil(len(train_dataloader)*num_epochs*0.1)  # 10% of train data for warm-up
logging.info("Warm-steps:{}".format(warm_steps))

'''
- train_objectives Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
- evaluator An evaluator(sentence_transformers.evaluation) evaluates the model performance during training on held-out 
    dev data. It is used to determine the best model that is saved to disc.
- epochs Number of epoches for training
- steps_per_epoch Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size
    from train_objectives.
- evaluation_steps – If > 0, evaluate the model using evaluator after each number of training steps
- output_path – Storage path for the model and evaluation files
- save_best_model – If true, the best model (according to evaluator) is stored at output_path
- max_grad_norm – Used for gradient normalization.
'''
# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warm_steps,
          evaluator=evaluator, evaluation_steps=400, output_path=model_save_path, save_best_model=True)

################################################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
################################################################################################
model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name='sts-test')
test_evaluator(model, output_path=model_save_path)