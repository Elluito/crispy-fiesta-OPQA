import numpy as np
import tensorflow as tf
from tensorflow.data import TFRecordDataset
import pandas as pd
import json
import pickle

import os





PATH_TO_SQUAD="datasets/Squad/"
PATH_TO_NARRATIVEQA_SHORT="datasets/NARRATIVEQA/"
PATH_TO_NARRATIVEQA_FULL="datasets/NARRATIVEQA/"
PATH_TO_NATURAL_QUESTIONS="D:\datsets_tesis\Kaggle_competition\Tensorflow_Q_and_A_competition/"

def convert_sentence_to_features(sentence, tokenizer, max_seq_len):
    tokens = ['[CLS]']
    tokens.extend(tokenizer.tokenize(sentence))
    if len(tokens) > max_seq_len - 1:
        tokens = tokens[:max_seq_len - 1]
    tokens.append('[SEP]')

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len - len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)

    return np.array(input_ids).reshape(1,max_seq_len),np.array(input_mask).reshape(1,max_seq_len), np.array(segment_ids).reshape(1,max_seq_len)


def read_dataset(dataset="squad",mode="test",tokenizer=None,max_seq_length=512):



    if dataset=="squad":
        if mode=="test":
            if not   os.path.exists(PATH_TO_SQUAD+"X_test") and not os.path.exists(PATH_TO_SQUAD+"Y_test") and  not os.path.exists(PATH_TO_SQUAD+"ids_test"):


                ids=[]
                X=[]
                y=[]
                f=open(PATH_TO_SQUAD+"dev-v1.1.json","r",encoding="utf8")
                for line in f:
                    temas=json.loads(line)["data"]
                    for temp in temas:
                        for cosa in temp["paragraphs"]:
                            text = cosa["context"]
                            C_id, C_mask, C_segment = convert_sentence_to_features(text, tokenizer, max_seq_length)
                            text_tokens = tokenizer.tokenize(text)
                            for question in cosa["qas"]:
                                unique_id = question["id"]
                                ids.append(unique_id)
                                for ans in question["answers"]:
                                    try:
                                        text_answer_list = ans["text"].split()
                                        first_word = tokenizer.tokenize(text_answer_list[0])[0]
                                        last_word = tokenizer.tokenize(text_answer_list[-1])[0]
                                        first_index =text_tokens.index(first_word)
                                        last_index=text_tokens.index(last_word)
                                    except:
                                        continue

                                Q_id, Q_mask, Q_segment = convert_sentence_to_features(question["question"], tokenizer,max_seq_length)
                                temp_y_start=np.zeros(max_seq_length)
                                temp_y_start[first_index]=1
                                temp_y_end= np.zeros(max_seq_length)
                                temp_y_end[last_index]=1
                                # dictionary={"questions_id":Q_id,"question_input_mask":Q_mask,"question_segment_id":Q_segment,"context_id":C_id,"context_input_mask":C_mask,"context_segment_id":C_segment}
                                X.append([C_id,C_mask,C_segment,Q_id,Q_mask,Q_mask])
                                y.append([temp_y_start,temp_y_start])
                # X=np.array(X)
                # y=np.array(y)
                # dictionary = {"questions_id": X[:,3], "question_input_mask": X[:,4], "question_segment_id": X[:,5],"context_id": X[:,0], "context_input_mask": X[:,1], "context_segment_id": X[:,2]}
                x_writer=open(PATH_TO_SQUAD+"X_test","w+b")
                y_writer=open(PATH_TO_SQUAD+"Y_test","w+b")
                ids_writer = open(PATH_TO_SQUAD + "ids_test", "w+b")

                pickle.dump(X,x_writer)
                pickle.dump(y,y_writer)
                pickle.dump(ids, ids_writer)
                x_writer.close()
                y_writer.close()
                ids_writer.close()
                return X,y,ids
            else:

                x_reader = open(PATH_TO_SQUAD + "X_test", "r+b")
                y_reader = open(PATH_TO_SQUAD + "Y_test", "r+b")
                ids_reader = open(PATH_TO_SQUAD + "ids_test", "r+b")

                X=pickle.load(x_reader)
                y=pickle.load(y_reader)
                ids=pickle.load(ids_reader)
                x_reader.close()
                y_reader.close()
                ids_reader.close()
                return X, y, ids








    elif dataset == "narrativeShort":
        pass

    elif dataset == "narrative_full":

        pass



    elif dataset == "naturalq":
        pass
