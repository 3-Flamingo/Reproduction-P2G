import torch
import pdb
import traceback
import pickle
import torch.nn as nn
from sklearn import preprocessing
from torch.nn import Parameter,Module
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import pprint,copy,os,random,math,sys,pickle,time
import pdb
import argparse, random, os, time
import numpy as np
# os.environ[ "CUDA_VISIBLE_DEVICES" ] = "1,2,3, 4 "
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(7)if use_cuda else "cpu")
def trans_to_cuda(variable):
    if use_cuda:
        return variable.to(device)
    else:
        return variable

class Data_data(object):
    def __init__(self, data, batch_size, debug, percent):
        super(Data_data, self).__init__()

        self.inputs = [item[0] for item in data]
        self.targets = [item[1] for item in data]
        self.masks = [item[2] for item in data]
        if debug:
            self.inputs, self.targets, self.masks = self.inputs[:100], self.targets[:100], self.masks[:100]

        self.epoch_flag = True
        self.corpus_length=int(len(self.targets) * percent)
        self.start=0
        self.len_data = int(self.corpus_length // batch_size)
        self.batch_size = batch_size
        print("total: ", self.corpus_length, self.len_data, self.batch_size)

    def next_batch(self, batch_size):
        start = self.start
        end = self.start + batch_size if self.start + batch_size < self.corpus_length else self.corpus_length
        self.start = self.start + batch_size
        if self.start<self.corpus_length:
            self.epoch_flag = True
        else:
            self.start=self.start%self.corpus_length
            self.epoch_flag = False
        # return [trans_to_cuda(torch.LongTensor(self.inputs[start:end]).item()),
        #         trans_to_cuda(torch.LongTensor(self.targets[start:end]).item()),
        #         trans_to_cuda(torch.FloatTensor(self.masks[start:end]).item())]
        return [trans_to_cuda(torch.LongTensor(self.inputs[start:end])),
                trans_to_cuda(torch.LongTensor(self.targets[start:end])),
                trans_to_cuda(torch.FloatTensor(self.masks[start:end]))]
        # return [trans_to_cuda(torch.LongTensor([item.cpu().detach().numpy() for item in self.inputs[start:end]])),
        #         trans_to_cuda(torch.LongTensor([item.cpu().detach().numpy() for item in self.targets[start:end]])),
        #         trans_to_cuda(torch.FloatTensor([item.cpu().detach().numpy() for item in self.masks[start:end]]))]

    def all_data(self, index=None):
        if self.masks is None:
            return [trans_to_cuda(torch.LongTensor(self.inputs)), trans_to_cuda(torch.LongTensor(self.targets))]
        else:
            return [trans_to_cuda(torch.LongTensor(self.inputs)), trans_to_cuda(torch.LongTensor(self.targets)), trans_to_cuda(torch.LongTensor(self.masks))]



def load(batch_size, model_type, data_name, debug, percent):
    if model_type == 'BERT':
        # dev_data = Data_data(pickle.load(open('/home/chenzhen/work/P2G/data/{}_basic_dev_bertdata.pickle'.format(data_name), 'rb')), batch_size=batch_size, debug=debug, percent=1.0)
        # test_data = Data_data(pickle.load(open('/home/chenzhen/work/P2G/data/{}_basic_test_bertdata.pickle'.format(data_name), 'rb')), batch_size=batch_size, debug=debug, percent=1.0)
        # train_data = Data_data(pickle.load(open('/home/chenzhen/work/P2G/data/{}_basic_train_bertdata.pickle'.format(data_name), 'rb')), batch_size=batch_size, debug=debug, percent=percent)
        dev_data = Data_data(
            pickle.load(open('/home/chenzhen/work/P2G/data/original_basic_dev_bertdata.pickle', 'rb')),
            batch_size=batch_size, debug=debug, percent=1.0)
        test_data = Data_data(
            pickle.load(open('/home/chenzhen/work/P2G/data/original_basic_test_bertdata.pickle', 'rb')),
            batch_size=batch_size, debug=debug, percent=1.0)
        train_data = Data_data(
            pickle.load(open('/home/chenzhen/work/P2G/data/original_basic_train_bertdata.pickle', 'rb')),
            batch_size=batch_size, debug=debug, percent=percent)

        print("Load {} Data ...".format(data_name))
    elif model_type == 'Roberta':
        dev_data = Data_data(pickle.load(open('/home/chenzhen/work/P2G/data/data/{}_basic_dev_bertdata.pickle'.format(data_name), 'rb')), batch_size=batch_size, debug=debug, percent=1.0)
        test_data = Data_data(pickle.load(open('/home/chenzhen/work/P2G/data/data/{}_basic_test_bertdata.pickle'.format(data_name), 'rb')), batch_size=batch_size, debug=debug, percent=1.0)
        train_data = Data_data(pickle.load(open('/home/chenzhen/work/P2G/data/data/{}_basic_train_bertdata.pickle'.format(data_name), 'rb')), batch_size=batch_size, debug=debug, percent=percent)
    print("{} data loaded ... ".format(model_type))


    return train_data, test_data, dev_data 

def write_results(task_name, result, tokenizer):
    with open('../res/{}'.format(task_name), "w") as f:
        for item in result:
            # print(item['chain'].cpu().detach().tolist())
            input_chain = tokenizer.convert_ids_to_tokens(item['chain'].cpu().detach().tolist())
            pred = tokenizer.convert_ids_to_tokens(item['pred'].cpu().detach().tolist())
            gold = tokenizer.convert_ids_to_tokens(item['gold'].cpu().detach().tolist())
            candidates = tokenizer.convert_ids_to_tokens(item['candidates'].cpu().detach().tolist())
            # res.append({"input_chain", input_chain, "pred": pred, "gold":gold})
            f.write(",".join(input_chain) + '\n')
            f.write("candidates: " + ",".join(candidates) + '\n')
            f.write("pred: " + ",".join(pred) + '\n')
            f.write("gold: " + ",".join(gold) + '\n')
            f.write("\n")
    f.close()
