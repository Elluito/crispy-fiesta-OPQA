import json

import numpy as np
import torch
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertModel

from main import crear_batch
from reading_datasets import read_dataset, unify_token, convert2string

PATH_TO_MODEL = "local_model/bert_tiny.bin"
PATH_TO_TINY = "local_model/uncased_L-2_H-128_A-2"
URL_TO_TINY = "google/bert_uncased_L-2_H-128_A-2"
def leer_config(path_to_config):
    archivo = open(path_to_config,"r")
    archivo_config = json.load(archivo)
    archivo.close()
    return archivo_config

def negLogSum(y_pred,y_true):
    y_true = torch.cuda.FloatTensor(y_true)
    y_pred = - torch.log(y_pred)
    real = torch.mul(y_pred,y_true)
    suma = real.sum()

    return suma

def build_model(max_seq_length):
    # path =PATH_TO_TINY+"/bert_config.json"
    # weights = torch.load(PATH_TO_MODEL)
    #
    # config = BertConfig(*leer_config(path))
    # model = BertModel(config)
    # model.load_state_dict(weights)

    #

    # model = AutoModel.from_pretrained(URL_TO_TINY)
    # model.eval()
    model = BertModel.from_pretrained(URL_TO_TINY)
    model.train()

    class CustomModel(torch.nn.Module):
        def __init__(self, D_out):
            """
            In the constructor we instantiate two nn.Linear modules and assign them as
            member variables.
            """
            super(CustomModel, self).__init__()
            self.BERT = model

            self.linear2 = torch.nn.Linear(self.BERT.pooler.dense.out_features, D_out)
            self.linear1 = torch.nn.Linear(self.BERT.pooler.dense.out_features, D_out)
            self.final = torch.nn.Softmax()
        def set_mode(self,mode="train"):
            if mode == "train":
                self.BERT.train()
            elif mode == "eval":
                self.BERT.eval()
            else:
                raise  Exception(f"No existe dicho modo {mode}")

        def forward(self, ids,masks):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            # ids,mask = list(zip(*x))
            # ids = torch.tensor(list(ids))
            # mask = torch.tensor(list(mask))
            h_relu = self.BERT(ids, token_type_ids=masks)[0]
            y_start = self.linear1(h_relu)
            y_end = self.linear2(h_relu)
            y_start = y_start.view(-1,max_seq_length)
            y_end = y_end.view(-1,max_seq_length)

            y_start = self.final(y_start)
            y_end = self.final(y_end)
            return y_start,y_end


    return CustomModel(1)
def adjust_x(x):
    ids, mask = x
    # new_ids = np.array(ids).reshape((1,-1))
    #
    # new_mask = np.array(mask).reshape((1,-1))

    tokens_tensor = torch.LongTensor(ids)
    segments_tensors = torch.LongTensor(mask)
    return tokens_tensor,segments_tensors
def reshape_x(x,mode=1):
    ids, mask = x
    if mode==1:
        return ids.view(-1),mask.view(-1)
    else:
        return np.array(ids).reshape(-1), np.array(mask).reshape(-1)
class SquadDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, ids,mask,start,end, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.IDS = ids
        self.Mask = mask
        self.y_start = np.array(start)
        self.y_end = np.array(end)
        self.transform = transform

    def __len__(self):
        return len(self.IDS)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ids = self.IDS[idx]
        segment = self.Mask[idx]
        start = self.y_start[idx]
        end = self.y_end[idx]
        sample = {'ids': ids, 'segment': segment,"start":start,"end":end}

        if self.transform:
            sample = self.transform(sample)

        return sample
def  pytroch_metric(ids,segments,y_pred_s,y_pred_e,y_start,y_end,tokenizer,log_name):

    start_acuraccy=[]
    end_acuraccy = []
    f = open(log_name, "w")
    for i,tokens in enumerate(ids):
        start_pred = np.argmax(y_pred_s[i])
        end_pred = np.argmax(y_pred_e[i])
        real_start = np.argmax(y_start[i])
        real_end = np.argmax(y_end[i])
        if start_pred == real_start:
            start_acuraccy.append(1)
        elif start_pred != real_start:
            start_acuraccy.append(0)
        if end_pred == real_end:
            end_acuraccy.append(1)
        elif start_pred != real_start:
            end_acuraccy.append(0)
        real_tokens = tokenizer.convert_ids_to_tokens(tokens)
        real_tokens = unify_token(real_tokens)
        question_index = real_tokens.index("[SEP]")
        QUESTION_TOKENS = real_tokens[question_index:]
        s = ""

        true_answer = convert2string(tokens[real_start:real_end])
        predicted_answer = convert2string(tokens[start_pred:end_pred])
        for tok in QUESTION_TOKENS:
            if tok != "[PAD]" and tok != "[SEP]" and tok != "[CLS]":
                s += tok + " "
        f.write("Question:{} True answer: {}   \n  Predicted_answer: {}      \n".format(s,true_answer,predicted_answer ))
    f.write(f"\n Start index acuraccy: {np.mean(start_acuraccy)}- Start end acuraccy: {np.mean(end_acuraccy)}")
    print(f"\n Start index acuraccy: {np.mean(start_acuraccy)}- Start end acuraccy: {np.mean(end_acuraccy)}")
    f.close()





if __name__ == '__main__':
    #
    tokenizer = AutoTokenizer.from_pretrained(URL_TO_TINY)
    device = torch.cuda.current_device()


    max_seq_length = 350
    model = build_model(max_seq_length)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00005, betas=(0.9, 0.98),
                                             eps=1e-9)
    model.cuda()

    criterion = negLogSum

    # Train data
    path = read_dataset(mode="test", dataset="squad", tokenizer=tokenizer, max_seq_length=max_seq_length,
                        fragmented=False, framework="torch")
    x,y = crear_batch(path,fragmented=False)


    thing = list(map(list, zip(*x)))
    ids,mask = np.squeeze(np.array(thing[0])),np.squeeze(np.array(thing[1]))


    train_dataset = SquadDataset(ids, mask, y[:, 0],y[:,1] )
    # Test data
    path = read_dataset(mode="train", dataset="squad", tokenizer=tokenizer, max_seq_length=max_seq_length,
                        fragmented=False, framework="torch")
    x_test, y_test = crear_batch(path, fragmented=False)
    thing = list(map(list, zip(*x_test)))
    ids, mask = np.squeeze(np.array(thing[0])), np.squeeze(np.array(thing[1]))


    test_dataset = SquadDataset(ids, mask, y[:, 0], y[:, 1])


    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        ids, segment,y_start,y_end =  batch["ids"].reshape(1,-11), batch["segment"].reshape(1,-1), batch["start"],\
                                      batch["end"]

        y1,y2 = model(torch.cuda.LongTensor(ids),torch.cuda.LongTensor(segment))
        loss = criterion(y1, y_start)+criterion(y2,y_end)
        loss.backward()
        optimizer.step()
        s = trainer.state
        item = loss.item()
        print(
            "{}/{} : {} - {:.3f}".format(s.epoch, s.max_epochs, s.iteration, item)
        )
        return item


    trainer = Engine(train_step)


    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            ids, segment, y_start, y_end = batch["ids"].reshape(1, -11), batch["segment"].reshape(1, -1), batch[
                "start"], \
                                           batch["end"]
            y1, y2 = model(torch.cuda.LongTensor(ids), torch.cuda.LongTensor(segment))
            y1_temp = torch.zeros(max_seq_length)
            y2_temp = torch.zeros(max_seq_length)
            y1_temp[torch.argmax(y1)] = 1
            y2_temp[torch.argmax(y2)] = 1
            y_start = torch.cuda.FloatTensor(y_start)
            y_end = torch.cuda.FloatTensor(y_end)

            return  torch.stack((y1_temp,y2_temp)),torch.stack((y_start,y_end))


    evaluator = Engine(validation_step)

    Accuracy().attach(evaluator, "val_acc")


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_dataset)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f}".format(trainer.state.epoch,
                                                                                           metrics["val_acc"]))


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(test_dataset)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f}"
              .format(trainer.state.epoch, metrics["val_acc"]))


    # trainer.run(train_dataset,epoch_length=100,max_epochs=3)
    # evaluator.run(test_dataset)
    model.eval()
    thing = list(map(list, zip(*x_test)))
    ids, mask = np.squeeze(np.array(thing[0])), np.squeeze(np.array(thing[1]))
    y_pred_start , y_pred_end = model( torch.cuda.LongTensor(ids), torch.cuda.LongTensor(mask))
    y_pred_start = y_pred_start.numpy()
    y_pred_end = y_pred_end.numpy()
    #
    #
    # print(model(torch.LongTensor(ids[:2]),torch.LongTensor(mask[:2])))

    pytroch_metric(ids,mask,y_pred_start,y_pred_end,y_test[:,0],y_test[:,1],log_name="prueba_predic_and_metric")
