import gc
import argparse
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
# import tensorflow as tf
import transformers
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AdamW
from set_rand import torch_set_random_seed
from mmd import MMD_loss
gc.collect()
import math
import os


# =================================================================================================================== #
# set parameters
parser = argparse.ArgumentParser()
# data parameters
parser.add_argument('--data_path', type=str, default="../data/semantic_data/")
parser.add_argument('--source_domain', type=str, default="book")
parser.add_argument('--target_domain', type=str, default="dvd")
parser.add_argument('--MAXLEN', type=int, default='512')

# model parameters
parser.add_argument('--pertrain_model_path', type=str, default='../pertrain_model/roberta-base/')
parser.add_argument('--cache_path', type=str, default='./cache')
parser.add_argument('--model_path', type=str, default='./cache/model')

# train parameters
parser.add_argument('--SEED', type=int, default=1023) # [2, 3, 5, 7, 9, 11, 13, 17, 19]
parser.add_argument('--EPOCH', type=int, default=10)
parser.add_argument('--BATCHSIZE', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-5) # TODO: fix lr

args = parser.parse_args()


# =================================================================================================================== #
# set random seed
torch_set_random_seed(args.SEED)


# =================================================================================================================== #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =================================================================================================================== #
# init pertrain model
ROBERTA_PATH = "../pertrain_model/roberta-base/"
TOKENIZER_PATH = "../pertrain_model/roberta-base/"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


# =================================================================================================================== #
# load data
def load_data(root_path, domain):
    positive_data_file = root_path + domain + "_positive_1000.txt"
    negative_data_file = root_path + domain + "_negative_1000.txt"
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8', errors='ignore').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8', errors='ignore').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    text = positive_examples + negative_examples
    label = [0] * len(positive_examples) + [1] * len(negative_examples)
    data_df = pd.DataFrame({'text':text, 'label':label})
    return data_df


# =================================================================================================================== #
# class Dataset
class LitRobertaBaseDataset(Dataset):
    def __init__(self, df, inference_only=False):
        super().__init__()

        self.df = df
        self.inference_only = inference_only
        self.text = df.text.tolist()

        if not self.inference_only:
            self.label = torch.tensor(df.label.values, dtype=torch.int64)

        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding='max_length',
            max_length=args.MAXLEN,
            truncation=True,
            return_attention_mask=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        attention_mask = torch.tensor(self.encoded['attention_mask'][index])

        if self.inference_only:
            return input_ids, attention_mask
        else:
            label = self.label[index]
            return input_ids, attention_mask, label


# =================================================================================================================== #
# class model
class RobertaBaseModel(nn.Module):
    def __init__(self):
        super(RobertaBaseModel, self).__init__()

        config = AutoConfig.from_pretrained(ROBERTA_PATH)
        config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": 0.2,
                       "layer_norm_eps": 1e-7})

        self.roberta = AutoModel.from_pretrained("../pertrain_model/roberta-base", config=config)

    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)
        return output


class TLTCPModel(nn.Module):
    def __init__(self, num_classes):
        super(TLTCPModel, self).__init__()
        self.roberta = RobertaBaseModel()
        self.attention = nn.Sequential(
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self.cls_fc_1 = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512,1),
            nn.ReLU()
        )
        self.cls_fc_2 = nn.Sequential(
            nn.Linear(args.MAXLEN, num_classes)
        )
        self.mmd_loss = MMD_loss()

    def forward(self, source_input_ids, source_attention_mask, source_label, target_input_ids, target_attention_mask):
        mmd_loss = 0
        cond_loss = 0
        source_roberta_output = self.roberta(source_input_ids, source_attention_mask)
        source = source_roberta_output.hidden_states[-1]
        source = self.cls_fc_1(source).squeeze()
        if source.sum() == 0:
            print(1)
        if self.training:
            target_roberta_output = self.roberta(target_input_ids, target_attention_mask)
            target = target_roberta_output.hidden_states[-1]
            target = self.cls_fc_1(target).squeeze()
            target_label = torch.nn.functional.softmax(target, dim=1)
            mmd_loss += self.mmd_loss.marginal(source, target)
            cond_loss += self.mmd_loss.conditional(source,target,source_label,target_label)
        source = self.cls_fc_2(source)
        return source, mmd_loss, cond_loss


# =================================================================================================================== #
# optimizer
def init_optimizer(model):
    roberta_parameters = list(model.roberta.parameters())

    freezen_parameters = roberta_parameters[:197]
    fine_tuning_parameters = roberta_parameters[197::]

    freezen_group = [params for params in freezen_parameters]
    fine_tuning_group = [params for params in fine_tuning_parameters]

    parameters = []
    # TODO: remove all Regularization
    parameters.append({"params": freezen_group,
                       "weight_decay": 1e-4,
                       "lr": 5e-6})
    parameters.append({'params': fine_tuning_group,
                       'weight_decay': 1e-4,
                       'lr': 5e-6})
    optimizer = AdamW(parameters)
    # optimizer = torch.optim.SGD(parameters)
    return optimizer


# =================================================================================================================== #
# train
def train(model, optimizer, source_data_loader, target_data_loader):
    gc.collect()

    source_iter = iter(source_data_loader)
    target_iter = iter(target_data_loader)

    cond_loss_line = []
    mmd_loss_line = []
    cls_loss_line = []
    loss_line = []
    test_acc = []

    for epoch in range(args.EPOCH):
        for batch_num, ((source_input_ids, source_attention_mask, source_label),\
                        (target_input_ids, target_attention_mask, target_label) )\
                in enumerate(zip(source_data_loader,target_data_loader)):
            source_input_ids = source_input_ids.to(DEVICE)
            source_attention_mask = source_attention_mask.to(DEVICE)
            source_label = source_label.to(DEVICE)
            target_input_ids = target_input_ids.to(DEVICE)
            target_attention_mask = target_attention_mask.to(DEVICE)

            optimizer.zero_grad()
            model.train()
            source_predictions, mmd_loss, cond_loss = \
                model(source_input_ids, source_attention_mask, source_label, target_input_ids, target_attention_mask)
            cls_loss = F.nll_loss(F.log_softmax(source_predictions, dim=1), source_label)
            lambd = 1 / (1 + math.exp(-10 * (batch_num+epoch*len(target_data_loader)) /
                                            args.EPOCH*len(target_data_loader))) -1
            lambd2 = 1 / (1 + math.exp(-10 * (batch_num + epoch * len(target_data_loader)) /
                                      args.EPOCH * len(target_data_loader))) - 1
            loss = 0.3*cls_loss + 0.3*mmd_loss*lambd + 0.3*cond_loss[0]*lambd2
            # loss = mmd_loss *0.2
            # loss = cls_loss
            loss.backward()
            optimizer.step()
            # ===================================== #
            # record
            cond_loss_line.append(cond_loss.tolist()[0])
            mmd_loss_line.append(mmd_loss.tolist())
            cls_loss_line.append(cls_loss.tolist())
            loss_line.append(loss.tolist())
            # ===================================== #
            # display
            print(f"epoch{epoch}:{format(100*float(batch_num)/float(len(source_data_loader)), '.2f')}%\t"
                  f"mmd_loss: {format(mmd_loss_line[-1], '.5f')}\t"
                  f"cond_loss: {format(cond_loss_line[-1], '.5f')}\t"
                  f"cls_loss: {format(cls_loss_line[-1], '.5f')}\t"
                  f"loss: {format(loss_line[-1], '.5f')}")
            # ===================================== #
            # eval
            if batch_num != 0 and batch_num % 49 == 0:
                test_acc.append(eval(model, target_data_loader))
            # if batch_num == 20: break
    gc.collect()
    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path)

        # save loss as file
    loss_lines = {"loss":loss_line, "mmd_loss_line":mmd_loss_line, "cls_loss_line":cls_loss_line, "cond_loss_line":cond_loss_line}
    pd_loss_lines = pd.DataFrame(loss_lines)
    pd_loss_lines.to_csv(f"{args.cache_path}/loss_lines.csv")

    acc_lines = {"acc":test_acc}
    pd_acc_lines = pd.DataFrame(acc_lines)
    pd_acc_lines.to_csv(f"{args.cache_path}/acc_line.csv")

    plt_loss(loss_line, title ='loss')
    plt_loss(mmd_loss_line, title = 'mmd_loss')
    plt_loss(cond_loss_line, title='cond_loss_line')
    plt_loss(cls_loss_line, title = 'cls_loss')
    plt_loss(test_acc, title = "test_acc")


# =================================================================================================================== #
# test
def eval(model, target_data_loader):
    model.eval()

    loss = 0
    acc= 0
    with torch.no_grad():
        for (target_input_ids, target_attention_mask, target_label) in target_data_loader:
            target_input_ids = target_input_ids.to(DEVICE)
            target_attention_mask = target_attention_mask.to(DEVICE)
            target_label = target_label.to(DEVICE)

            predictions, mmd_loss, cond_loss = model(target_input_ids, target_attention_mask, target_label, target_input_ids, target_attention_mask)
            loss += F.nll_loss(F.log_softmax(predictions, dim=1), target_label, reduction='sum').item()
            predictions = predictions.data.max(1)[1]
            acc += predictions.eq(target_label.data.view_as(predictions)).cpu().sum()
    loss /= len(target_data_loader.dataset)
    acc = acc.tolist()/len(target_data_loader.dataset)
    print(f"{args.target_domain}: Average loss = {format(loss, '.5f')}\t"
          f"Accuracy = {format(acc*100, '.3f')}%")
    return acc


# =================================================================================================================== #
# =================================================================================================================== #
# =================================================================================================================== #
# =================================================================================================================== #
# =================================================================================================================== #
def plt_loss(data_line, title ="loss", smooth = False, width = 2):
    # print(loss_line)
    if smooth:
        new_line = []
        for i in range(len(data_line) - width):
            new_line.append(np.mean(data_line[i:i + width]))
        data_line = new_line

    t = np.arange(1, len(data_line) + 1, 1)
    plt.plot(t, data_line, 'r')
    label = [title]
    plt.legend(label, loc='upper left')
    plt.savefig(f'{args.cache_path}/{title}.jpg')
    # plt.show()


# =================================================================================================================== #
def main():
    print(f"DEVICE: {DEVICE}")
    source_data_df = load_data(args.data_path, args.source_domain)
    target_data_df = load_data(args.data_path, args.target_domain)
    # source_data_df = source_data_df[:200]
    # target_data_df = target_data_df[:200]
    source_data = LitRobertaBaseDataset(source_data_df)
    target_data = LitRobertaBaseDataset(target_data_df)
    source_data_loader = DataLoader(source_data, batch_size=args.BATCHSIZE, drop_last=True, shuffle=True)
    target_data_loader = DataLoader(target_data, batch_size=args.BATCHSIZE, drop_last=True, shuffle=True)

    model = TLTCPModel(num_classes=2).to(DEVICE)
    optimizer = init_optimizer(model)
    train(model, optimizer, source_data_loader, target_data_loader)

    print()
    pass


# =================================================================================================================== #
if __name__ == "__main__":
    main()