import json
import os
import pickle
import re

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from official.nlp.bert.tokenization import FullTokenizer

PATH_TO_SQUAD ="datasets/Squad/"
PATH_TO_NARRATIVEQA_SHORT ="datasets/NARRATIVEQA/"
PATH_TO_NARRATIVEQA_FULL ="datasets/NARRATIVEQA/"
# PATH_TO_NATURAL_QUESTIONS ="D:\datsets_tesis\Kaggle_competition\Tensorflow_Q_and_A_competition/"
PATH_TO_NATURAL_QUESTIONS ="datasets/NaturalQ_dataset/Tensorflow_Q_and_A_competition/"
import string
feature_description = {
    'context_id': tf.io.FixedLenFeature([],tf.int64, default_value=0),
    'context_input_mask': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'context_segment_id': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'question_id': tf.io.FixedLenFeature([],tf.int64, default_value=0),
    'question_input_mask':tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'question_segment_id': tf.io.FixedLenFeature([], tf.int64, default_value=0),
}

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
def convert_tokens_to_features(sentence, tokenizer, max_seq_len):
    tokens = ['[CLS]']
    tokens.extend(sentence)
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


def read_dataset(dataset="squad",mode="test",version="simplified",fragmented=True,tokenizer=None,max_seq_length=512):



    if dataset == "squad":
        if mode == "test":
            if fragmented:
                i=0
                if not os.listdir(PATH_TO_SQUAD+"test"):


                    ids = []
                    X = []
                    y = []
                    f = open(PATH_TO_SQUAD+"dev-v1.1.json","r",encoding="utf8")
                    for line in f:
                        temas = json.loads(line)["data"]
                        for temp in temas:
                            for cosa in temp["paragraphs"]:
                                text = cosa["context"]
                                C_id, C_mask, C_segment = convert_sentence_to_features(text, tokenizer, max_seq_length)
                                text_tokens = tokenizer.tokenize(text)
                                if len(text_tokens)>max_seq_length:
                                    continue
                                for question in cosa["qas"]:
                                    unique_id = question["id"]
                                    ids.append(unique_id)
                                    for ans in question["answers"]:
                                        try:
                                            if ans["text"] in text:
                                                text_answer_list = tokenizer.tokenize(ans["text"])
                                                indices = []
                                                for token in text_answer_list:
                                                    indices.append(list(text_tokens).index(token))
                                                first_index = indices[0]
                                                last_index = indices[-1]
                                                break
                                        except:
                                            continue

                                    Q_id, Q_mask, Q_segment = convert_sentence_to_features(question["question"], tokenizer,max_seq_length)
                                    temp_y_start=np.zeros(max_seq_length)
                                    temp_y_start[first_index]=1
                                    temp_y_end= np.zeros(max_seq_length)
                                    temp_y_end[last_index]=1
                                    # dictionary={"questions_id":Q_id,"question_input_mask":Q_mask,"question_segment_id":Q_segment,"context_id":C_id,"context_input_mask":C_mask,"context_segment_id":C_segment}
                                    X.append([C_id,C_mask,C_segment,Q_id,Q_mask,Q_mask])
                                    y.append([temp_y_start,temp_y_end])

                                    if np.array(X).itemsize*np.array(X).size>1000000:
                                        with open(PATH_TO_SQUAD+"test/"+"X_{}".format(i),"w+b")as f:
                                            pickle.dump(X,f)
                                        with open(PATH_TO_SQUAD+"test/"+"Y_{}".format(i),"w+b")as f:
                                            pickle.dump(y,f)
                                        with open(PATH_TO_SQUAD+"test/"+"ids_{}".format(i),"w+b")as f:
                                            pickle.dump(ids,f)
                                        i+=1
                                        X=[]
                                        y=[]
                                        ids=[]
                    # X=np.array(X)
                    # y=np.array(y)
                    # dictionary = {"questions_id": X[:,3], "question_input_mask": X[:,4], "question_segment_id": X[:,5],"context_id": X[:,0], "context_input_mask": X[:,1], "context_segment_id": X[:,2]}

                    return PATH_TO_SQUAD+"test/"
                else:

                    # x_reader = open(PATH_TO_SQUAD + "X_test", "r+b")
                    # y_reader = open(PATH_TO_SQUAD + "Y_test", "r+b")
                    # ids_reader = open(PATH_TO_SQUAD + "ids_test", "r+b")
                    #
                    # X=pickle.load(x_reader)
                    # y=pickle.load(y_reader)
                    # ids=pickle.load(ids_reader)
                    # x_reader.close()
                    # y_reader.close()
                    # ids_reader.close()
                    return PATH_TO_SQUAD+"test/"
            else:
                if not os.listdir(PATH_TO_SQUAD + "test"):
                    ids = []
                    X = []
                    y = []
                    f = open(PATH_TO_SQUAD + "dev-v1.1.json", "r", encoding="utf8")
                    for line in f:
                        temas = json.loads(line)["data"]
                        for temp in temas:
                            for cosa in temp["paragraphs"]:
                                text = cosa["context"]
                                C_id, C_mask, C_segment = convert_sentence_to_features(text, tokenizer, max_seq_length)
                                text_tokens = tokenizer.tokenize(text)
                                if len(text_tokens) > max_seq_length:
                                    continue
                                for question in cosa["qas"]:
                                    unique_id = question["id"]
                                    ids.append(unique_id)
                                    for ans in question["answers"]:
                                        try:
                                            if ans["text"] in text:
                                                text_answer_list = tokenizer.tokenize(ans["text"])
                                                indices = []
                                                for token in text_answer_list:
                                                    indices.append(list(text_tokens).index(token))
                                                first_index = indices[0]
                                                last_index = indices[-1]
                                                break
                                        except:
                                            continue

                                    Q_id, Q_mask, Q_segment = convert_sentence_to_features(question["question"], tokenizer,
                                                                                           max_seq_length)
                                    temp_y_start = np.zeros(max_seq_length)
                                    temp_y_start[first_index] = 1
                                    temp_y_end = np.zeros(max_seq_length)
                                    temp_y_end[last_index] = 1

                                    X.append([C_id, C_mask, C_segment, Q_id, Q_mask, Q_mask])
                                    y.append([temp_y_start, temp_y_end])

                                    # if np.array(X).itemsize * np.array(X).size > 1000000:
                                    #     with open(PATH_TO_SQUAD + "test/" + "X_{}".format(i), "w+b")as f:
                                    #         pickle.dump(X, f)
                                    #     with open(PATH_TO_SQUAD + "test/" + "Y_{}".format(i), "w+b")as f:
                                    #         pickle.dump(y, f)
                                    #     with open(PATH_TO_SQUAD + "test/" + "ids_{}".format(i), "w+b")as f:
                                    #         pickle.dump(ids, f)
                                    #     i += 1
                                    #     X = []
                                    #     y = []
                                    #     ids = []
                    # X=np.array(X)
                    # y=np.array(y)
                    # dictionary = {"questions_id": X[:,3], "question_input_mask": X[:,4], "question_segment_id": X[:,5],"context_id": X[:,0], "context_input_mask": X[:,1], "context_segment_id": X[:,2]}
                        with open(PATH_TO_SQUAD + "test/" + "X", "w+b")as f:
                            pickle.dump(X, f)
                        with open(PATH_TO_SQUAD + "test/" + "Y", "w+b")as f:
                            pickle.dump(y, f)
                        with open(PATH_TO_SQUAD + "test/" + "ids", "w+b")as f:
                            pickle.dump(ids, f)

                    return PATH_TO_SQUAD + "test/"
                else:


                    return PATH_TO_SQUAD + "test/"



        elif mode == "train":
            if fragmented:
                i = 0
                if not os.listdir(PATH_TO_SQUAD + "train"):

                    ids = []
                    X = []
                    y = []
                    f = open(PATH_TO_SQUAD + "train-v1.1.json", "r", encoding="utf8")
                    for line in f:
                        temas = json.loads(line)["data"]
                        for temp in temas:
                            for cosa in temp["paragraphs"]:
                                text = cosa["context"]
                                C_id, C_mask, C_segment = convert_sentence_to_features(text, tokenizer, max_seq_length)
                                text_tokens = tokenizer.tokenize(text)
                                if len(text_tokens) > max_seq_length:
                                    continue
                                for question in cosa["qas"]:
                                    unique_id = question["id"]
                                    ids.append(unique_id)
                                    for ans in question["answers"]:
                                        try:
                                            if ans["text"] in text:
                                                text_answer_list = tokenizer.tokenize(ans["text"])
                                                indices = []
                                                for token in text_answer_list:
                                                    indices.append(list(text_tokens).index(token))
                                                first_index = indices[0]
                                                last_index = indices[-1]
                                                break
                                        except:
                                            continue

                                    Q_id, Q_mask, Q_segment = convert_sentence_to_features(question["question"],
                                                                                           tokenizer, max_seq_length)
                                    temp_y_start = np.zeros(max_seq_length)
                                    temp_y_start[first_index] = 1
                                    temp_y_end = np.zeros(max_seq_length)
                                    temp_y_end[last_index] = 1
                                    # dictionary={"questions_id":Q_id,"question_input_mask":Q_mask,"question_segment_id":Q_segment,"context_id":C_id,"context_input_mask":C_mask,"context_segment_id":C_segment}
                                    X.append([C_id, C_mask, C_segment, Q_id, Q_mask, Q_segment])
                                    y.append([temp_y_start, temp_y_end])

                                    if np.array(X).itemsize * np.array(X).size > 1000000:
                                        with open(PATH_TO_SQUAD + "train/" + "X_{}".format(i), "w+b")as f:
                                            pickle.dump(X, f)
                                        with open(PATH_TO_SQUAD + "train/" + "Y_{}".format(i), "w+b")as f:
                                            pickle.dump(y, f)
                                        with open(PATH_TO_SQUAD + "train/" + "ids_{}".format(i), "w+b")as f:
                                            pickle.dump(ids, f)
                                        i += 1
                                        X = []
                                        y = []
                                        ids = []

                    return PATH_TO_SQUAD + "train/"
                else:

                    # x_reader = open(PATH_TO_SQUAD + "X_test", "r+b")
                    # y_reader = open(PATH_TO_SQUAD + "Y_test", "r+b")
                    # ids_reader = open(PATH_TO_SQUAD + "ids_test", "r+b")
                    #
                    # X=pickle.load(x_reader)
                    # y=pickle.load(y_reader)
                    # ids=pickle.load(ids_reader)
                    # x_reader.close()
                    # y_reader.close()
                    # ids_reader.close()
                    return PATH_TO_SQUAD + "test/"

            else:
                if not   os.path.exists(PATH_TO_SQUAD+"train/"+"X") and not os.path.exists(PATH_TO_SQUAD+"train/"+"Y") and  not os.path.exists(PATH_TO_SQUAD+"train/"+"ids"):


                    ids=[]
                    X=[]
                    y=[]
                    f=open(PATH_TO_SQUAD+"train-v1.1.json","r",encoding="utf8")
                    for line in f:
                        temas=json.loads(line)["data"]
                        for temp in temas:
                            for cosa in temp["paragraphs"]:
                                text = cosa["context"]
                                C_id, C_mask, C_segment = convert_sentence_to_features(text, tokenizer, max_seq_length)
                                text_tokens = tokenizer.tokenize(text)
                                if len(text_tokens)>max_seq_length:
                                    continue
                                for question in cosa["qas"]:
                                    unique_id = question["id"]
                                    ids.append(unique_id)
                                    for ans in question["answers"]:
                                        try:
                                            if ans["text"] in text:
                                                text_answer_list = tokenizer.tokenize(ans["text"] )
                                                indices=[]
                                                for token in text_answer_list:
                                                    indices.append(list(text_tokens).index(token))
                                                first_index =indices[0]
                                                last_index = indices[-1]
                                            break
                                        except:
                                            continue

                                    Q_id, Q_mask, Q_segment = convert_sentence_to_features(question["question"], tokenizer,max_seq_length)
                                    temp_y_start=np.zeros(max_seq_length)
                                    temp_y_start[first_index]=1
                                    temp_y_end= np.zeros(max_seq_length)
                                    temp_y_end[last_index]=1
                                    # dictionary={"questions_id":Q_id,"question_input_mask":Q_mask,"question_segment_id":Q_segment,"context_id":C_id,"context_input_mask":C_mask,"context_segment_id":C_segment}
                                    X.append([C_id,C_mask,C_segment,Q_id,Q_mask,Q_mask])
                                    y.append([temp_y_start,temp_y_end])
                    # X=np.array(X)
                    # y=np.array(y)
                    # dictionary = {"questions_id": X[:,3], "question_input_mask": X[:,4], "question_segment_id": X[:,5],"context_id": X[:,0], "context_input_mask": X[:,1], "context_segment_id": X[:,2]}
                    x_writer=open(PATH_TO_SQUAD+"train/"+"X","w+b")
                    y_writer=open(PATH_TO_SQUAD+"train/"+"Y","w+b")
                    ids_writer = open(PATH_TO_SQUAD+"train/" + "ids", "w+b")

                    pickle.dump(X,x_writer)
                    pickle.dump(y,y_writer)
                    pickle.dump(ids, ids_writer)
                    x_writer.close()
                    y_writer.close()
                    ids_writer.close()
                    return PATH_TO_SQUAD+"train/"
                else:

                    # x_reader = open(PATH_TO_SQUAD +"train/"+ "X", "r+b")
                    # y_reader = open(PATH_TO_SQUAD +"train/"+ "Y", "r+b")
                    # ids_reader = open(PATH_TO_SQUAD +"train/"+ "ids", "r+b")
                    #
                    # X=pickle.load(x_reader)
                    # y=pickle.load(y_reader)
                    # ids=pickle.load(ids_reader)
                    # x_reader.close()
                    # y_reader.close()
                    # ids_reader.close()
                    return PATH_TO_SQUAD+"train/"








    elif dataset == "narrativeShort":
        pass

    elif dataset == "narrative_full":

        pass



    elif dataset == "naturalq":
        if mode == "test":
            if fragmented:
                i = 0
                if not os.listdir(PATH_TO_NATURAL_QUESTIONS + "test"):

                    ids = []
                    X = []
                    y = []
                    f = open(PATH_TO_NATURAL_QUESTIONS + "simplified-nq-test.jsonl", "r", encoding="utf8")
                    for line in f:
                        temas = json.loads(line)["data"]
                        for temp in temas:
                            for cosa in temp["paragraphs"]:
                                text = cosa["context"]
                                C_id, C_mask, C_segment = convert_sentence_to_features(text, tokenizer, max_seq_length)
                                text_tokens = tokenizer.tokenize(text)
                                if len(text_tokens) > max_seq_length:
                                    continue
                                for question in cosa["qas"]:
                                    unique_id = question["id"]
                                    ids.append(unique_id)
                                    for ans in question["answers"]:
                                        try:
                                            if ans["text"] in text:
                                                text_answer_list = tokenizer.tokenize(ans["text"])
                                                indices = []
                                                for token in text_answer_list:
                                                    indices.append(list(text_tokens).index(token))
                                                first_index = indices[0]
                                                last_index = indices[-1]
                                                break
                                        except:
                                            continue

                                    Q_id, Q_mask, Q_segment = convert_sentence_to_features(question["question"],
                                                                                           tokenizer, max_seq_length)
                                    temp_y_start = np.zeros(max_seq_length)
                                    temp_y_start[first_index] = 1
                                    temp_y_end = np.zeros(max_seq_length)
                                    temp_y_end[last_index] = 1
                                    # dictionary={"questions_id":Q_id,"question_input_mask":Q_mask,"question_segment_id":Q_segment,"context_id":C_id,"context_input_mask":C_mask,"context_segment_id":C_segment}
                                    X.append([C_id, C_mask, C_segment, Q_id, Q_mask, Q_mask])
                                    y.append([temp_y_start, temp_y_end])

                                    if np.array(X).itemsize * np.array(X).size > 1000000:
                                        with open(PATH_TO_NATURAL_QUESTIONS + "test/" + "X_{}".format(i), "w+b")as f:
                                            pickle.dump(X, f)
                                        with open(PATH_TO_NATURAL_QUESTIONS + "test/" + "Y_{}".format(i), "w+b")as f:
                                            pickle.dump(y, f)
                                        with open(PATH_TO_NATURAL_QUESTIONS + "test/" + "ids_{}".format(i), "w+b")as f:
                                            pickle.dump(ids, f)
                                        i += 1
                                        X = []
                                        y = []
                                        ids = []
                    # X=np.array(X)
                    # y=np.array(y)
                    # dictionary = {"questions_id": X[:,3], "question_input_mask": X[:,4], "question_segment_id": X[:,5],"context_id": X[:,0], "context_input_mask": X[:,1], "context_segment_id": X[:,2]}

                    return PATH_TO_NATURAL_QUESTIONS + "test/"
                else:

                    # x_reader = open(PATH_TO_NATURAL_QUESTIONS + "X_test", "r+b")
                    # y_reader = open(PATH_TO_NATURAL_QUESTIONS + "Y_test", "r+b")
                    # ids_reader = open(PATH_TO_NATURAL_QUESTIONS + "ids_test", "r+b")
                    #
                    # X=pickle.load(x_reader)
                    # y=pickle.load(y_reader)
                    # ids=pickle.load(ids_reader)
                    # x_reader.close()
                    # y_reader.close()
                    # ids_reader.close()
                    return PATH_TO_NATURAL_QUESTIONS + "test/"
            else:
                if not os.listdir(PATH_TO_NATURAL_QUESTIONS + "test"):
                    ids = []
                    X = []
                    html_text = []
                    y = []
                    if version == "simplified":
                        f = open(PATH_TO_NATURAL_QUESTIONS + "simplified-nq-test.jsonl", "r", encoding="utf8")
                    else:
                        f = open(PATH_TO_NATURAL_QUESTIONS + "v1.0-simplified_nq-dev-all.jsonl", "r", encoding="utf8")
                    number_ignored = 0
                    no_answer = 0
                    i= 0
                    for line in f:
                        if i==0:
                            print(line)
                        temas = json.loads(line)

                        question = temas["question_text"]
                        text = temas["document_text"].lower()
                        clean_text = cleanhtml(text)

                        annotations = temas["long_answer_candidates"]
                        html_text.append(text)
                        indice =[]
                        for elem in annotations:
                            byte_start_index = elem["start_token"]
                            byte_end_index = elem["end_token"]
                            indice.append((byte_start_index,byte_end_index))

                        long_answer = text[indice[0][0]:indice[0][1]-1]
                        ## SOn tokens lo que representan los indices
                        long_answer = text.split()[byte_start_index: byte_end_index]
                        # long_answer = convert2string(long_answer)
                        # tokenized_answer = tokenizer.tokenize(long_answer)
                        clean_text = convert2string(clean_text.split())
                        # Este comando devuelve todas las parejas de indices que contienen dicho substring por eso necesitamos el primer elemento de la última pareja [-1][0]
                        final_index = find_substring(clean_text, "see also")[-1][0]

                        clean_text = clean_text[:final_index]

                        tokenized_text = tokenizer.tokenize(clean_text)
                        initial_index, final_index = find_answer_index(clean_text, long_answer, mode=2)

                        tokenized_answer = tokenizer.tokenize(clean_text[initial_index:final_index])

                        #  ESTO ES PARA ENCONTRAR LOS  INDICE CONSECUTIVOS DE LA RESPUESTA EN EL TEXTO LIMPIO TOKENIZADO
                        answer_indexes = [(i, i + len(tokenized_answer)) for i in range(len(tokenized_text)) if
                                          tokenized_text[i:i + len(tokenized_answer)] == tokenized_answer]
                        final_text = []
                        final_text.extend(tokenized_text[answer_indexes[0][0]:answer_indexes[0][1]])
                        i = 0
                        indice_atras = 0
                        indice_adelante = 0
                        correcto = False
                        while len(final_text) < max_seq_length or not correcto:

                            if i == 0:
                                final_text.append(tokenized_text[answer_indexes[0][1] + indice_adelante])
                                indice_adelante += 1
                                i = 1
                            if i == 1:
                                final_text.insert(0, tokenized_text[answer_indexes[0][0] - indice_atras])
                                indice_atras += 1
                                i = 0

                        ids.append(temas["example_id"])
                        # html_text.append(text)

                        C_id, C_mask, C_segment = convert_tokens_to_features(final_text, tokenizer, max_seq_length)

                        Q_id, Q_mask, Q_segment = convert_sentence_to_features(question,
                                                                               tokenizer, max_seq_length)


                        # if annotations["yes_no_answer"] == "YES":
                        #     temp_y_start[-2] = 1
                        #     temp_y_end[-2] = 1
                        #
                        # elif annotations["yes_no_answer"] == "NO":
                        #     temp_y_start[-1] = 1
                        #     temp_y_end[-1] = 1
                        #
                        # if annotations["yes_no_answer"] == "NONE":
                        #     temp_y_start[answer_indexes[0]] = 1
                        #     temp_y_end[answer_indexes[-1]] = 1

                        X.append([C_id, C_mask, C_segment, Q_id, Q_mask, Q_segment])

                        with open(PATH_TO_NATURAL_QUESTIONS + "test/" + "X", "w+b")as f:
                            pickle.dump(X, f)
                        with open(PATH_TO_NATURAL_QUESTIONS + "test/" + "html_text", "w+b")as f:
                            pickle.dump(html_text, f)
                        with open(PATH_TO_NATURAL_QUESTIONS + "test/" + "ids", "w+b")as f:
                            pickle.dump(ids, f)

                    return PATH_TO_NATURAL_QUESTIONS + "test/"
                else:

                    return PATH_TO_NATURAL_QUESTIONS + "test/"



        elif mode == "train":
            if fragmented:
                i = 0
                if not os.listdir(PATH_TO_NATURAL_QUESTIONS + "train"):

                    ids = []
                    X = []
                    y = []
                    f = open(PATH_TO_NATURAL_QUESTIONS + "simplified-nq-train.jsonl", "r", encoding="utf8")
                    number_ignored = 0
                    no_answer = 0
                    i=0
                    archivo_pequeño=[]
                    for line in f:
                        temas = json.loads(line)
                        print(temas)
                        if i <500:
                            archivo_pequeño.append(temas)


                        question = temas["question_text"]
                        text = temas["document_text"].lower()
                        clean_text = cleanhtml(text)
                        annotations = temas["annotations"]

                        byte_start_index = annotations["long_answer"]["start_byte"]
                        byte_end_index = annotations["long_answer"]["end_byte"]
                        if byte_end_index == -1 and byte_start_index == -1 and annotations["yes_no_answer"] == "NONE":
                            number_ignored += 1
                            no_answer += 1
                            continue
                        long_answer = text[byte_start_index, byte_end_index]
                        tokenized_answer = tokenizer.tokenize(long_answer)
                        clean_text = ' '.join(clean_text.split())
                        # Este comando devuelve todas las parejas de indices que contienen dicho substring por eso necesitamos el primer elemento de la última pareja [-1][0]
                        final_index = find_substring(clean_text, "see also")[-1][0]

                        clean_text = clean_text[:final_index]

                        tokenized_text = tokenizer.tokenize(clean_text)
                        initial_index, final_index = find_answer_index(clean_text, tokenized_answer, mode=2)
                        tokenized_answer = tokenizer.tokenize(clean_text[initial_index:final_index])
                        #  ESTO ES PARA ENCONTRAR LOS  INDICE CONSECUTIVOS DE LA RESPUESTA EN EL TEXTO LIMPIO TOKENIZADO
                        answer_indexes = [(i, i + len(tokenized_answer)) for i in range(len(tokenized_text)) if
                                          tokenized_text[i:i + len(tokenized_answer)] == tokenized_answer]



                        if answer_indexes[-1] > 350 and annotations["yes_no_answer"] == "NONE":
                            # I skip the text that do not have the answer on the first 350 token
                            number_ignored += 1
                            continue

                        ids.append(temas["example_id"])
                        # html_text.append(text)

                        C_id, C_mask, C_segment = convert_sentence_to_features(clean_text, tokenizer, max_seq_length)

                        Q_id, Q_mask, Q_segment = convert_sentence_to_features(question,
                                                                               tokenizer, max_seq_length)
                        temp_y_start = np.zeros(max_seq_length + 2)
                        temp_y_end = np.zeros(max_seq_length + 2)

                        if annotations["yes_no_answer"] == "YES":
                            temp_y_start[-2] = 1
                            temp_y_end[-2] = 1

                        elif annotations["yes_no_answer"] == "NO":
                            temp_y_start[-1] = 1
                            temp_y_end[-1] = 1

                        if annotations["yes_no_answer"] == "NONE":
                            temp_y_start[answer_indexes[0]] = 1
                            temp_y_end[answer_indexes[-1]] = 1

                        X.append([C_id, C_mask, C_segment, Q_id, Q_mask, Q_segment])
                        y.append([temp_y_start, temp_y_end])

                        if np.array(X).itemsize * np.array(X).size > 1000000:
                            with open(PATH_TO_NATURAL_QUESTIONS + "train/" + "X_{}".format(i), "w+b")as f:
                                pickle.dump(X, f)
                            with open(PATH_TO_NATURAL_QUESTIONS + "train/" + "Y_{}".format(i), "w+b")as f:
                                pickle.dump(y, f)
                            with open(PATH_TO_NATURAL_QUESTIONS + "train/" + "ids_{}".format(i), "w+b")as f:
                                pickle.dump(ids, f)
                            i += 1
                            X = []
                            y = []
                            ids = []

                    return PATH_TO_NATURAL_QUESTIONS + "train/"
                else:

                    # x_reader = open(PATH_TO_NATURAL_QUESTIONS + "X_test", "r+b")
                    # y_reader = open(PATH_TO_NATURAL_QUESTIONS + "Y_test", "r+b")
                    # ids_reader = open(PATH_TO_NATURAL_QUESTIONS + "ids_test", "r+b")
                    #
                    # X=pickle.load(x_reader)
                    # y=pickle.load(y_reader)
                    # ids=pickle.load(ids_reader)
                    # x_reader.close()
                    # y_reader.close()
                    # ids_reader.close()
                    return PATH_TO_NATURAL_QUESTIONS + "test/"

            else:
                if not os.path.exists(PATH_TO_NATURAL_QUESTIONS + "train/" + "X") and not os.path.exists(
                        PATH_TO_NATURAL_QUESTIONS + "train/" + "Y") and not os.path.exists(PATH_TO_NATURAL_QUESTIONS + "train/" + "ids"):

                    ids = []
                    X = []
                    y = []
                    # html_text = []
                    f = open(PATH_TO_NATURAL_QUESTIONS + "simplified-nq-train.jsonl", "r", encoding="utf8")
                    number_ignored = 0
                    no_answer =0
                    i = 0
                    archivo_pequeno = []
                    counter=0
                    for line in f:
                        counter+=1
                        temas = json.loads(line)

                        if i < 500:
                            archivo_pequeno.append(temas)
                            i+=1
                        question = temas["question_text"]
                        text = temas["document_text"].lower()
                        clean_text = cleanhtml(text)
                        annotations = temas["annotations"]

                        byte_start_index = annotations[0]["long_answer"]["start_token"]
                        byte_end_index = annotations[0]["long_answer"]["end_token"]

                        if byte_end_index ==-1 and byte_start_index==-1 and annotations[0]["yes_no_answer"] =="NONE":
                            number_ignored += 1
                            no_answer += 1
                            continue

                        ## SOn tokens lo que representan los indices
                        long_answer = text.split()[byte_start_index: byte_end_index]
                        # long_answer = convert2string(long_answer)
                        # tokenized_answer = tokenizer.tokenize(long_answer)
                        clean_text = convert2string(clean_text.split())
                        # Este comando devuelve todas las parejas de indices que contienen dicho substring por eso necesitamos el primer elemento de la última pareja [-1][0]
                        final_index = find_substring(clean_text, "see also")[-1][0]

                        clean_text = clean_text[:final_index]

                        tokenized_text = tokenizer.tokenize(clean_text)
                        initial_index, final_index = find_answer_index(clean_text, long_answer, mode=2)

                        tokenized_answer = tokenizer.tokenize(clean_text[initial_index:final_index])

                        #  ESTO ES PARA ENCONTRAR LOS  INDICE CONSECUTIVOS DE LA RESPUESTA EN EL TEXTO LIMPIO TOKENIZADO

                        answer_indexes = [(i, i + len(tokenized_answer)) for i in range(len(tokenized_text)) if
                                          tokenized_text[i:i + len(tokenized_answer)] == tokenized_answer]
                        print(answer_indexes)
                        print("length tokenized answer")
                        print(len(tokenized_text))
                        final_text = []
                        final_text.extend(tokenized_text[answer_indexes[0][0]:answer_indexes[0][1]])
                        i = 0
                        indice_atras = 0
                        indice_adelante = 0
                        while len(final_text) < max_seq_length:
                            if i == 0:
                                final_text.append(tokenized_text[answer_indexes[0][1] + indice_adelante])
                                indice_adelante += 1
                                i = 1
                            if i == 1:
                                final_text.insert(0, tokenized_text[answer_indexes[0][0] - indice_atras])
                                indice_atras += 1
                                i = 0


                        ids.append(temas["example_id"])
                        # html_text.append(text)





                        C_id, C_mask, C_segment = convert_tokens_to_features(final_text, tokenizer, max_seq_length)


                        Q_id, Q_mask, Q_segment = convert_sentence_to_features(question,
                                                                                           tokenizer, max_seq_length)
                        temp_y_start = np.zeros(max_seq_length+2)
                        temp_y_end = np.zeros(max_seq_length+2)
                        answer_indexes = [(i, i + len(tokenized_answer)) for i in range(len(final_text)) if
                                          final_text[i:i + len(tokenized_answer)] == tokenized_answer]

                        if annotations[0]["yes_no_answer"] == "YES":
                            temp_y_start[-2] = 1
                            temp_y_end[-2] = 1

                        elif annotations[0]["yes_no_answer"] == "NO":
                            temp_y_start[-1] = 1
                            temp_y_end[-1] = 1

                        if annotations[0]["yes_no_answer"] == "NONE":
                            temp_y_start[answer_indexes[0][0]] = 1
                            temp_y_end[answer_indexes[0][1]] = 1

                        X.append([C_id, C_mask, C_segment, Q_id, Q_mask,  Q_segment])
                        y.append([temp_y_start, temp_y_end])


                    x_writer = open(PATH_TO_NATURAL_QUESTIONS + "train/" + "X", "w+b")
                    y_writer = open(PATH_TO_NATURAL_QUESTIONS + "train/" + "Y", "w+b")
                    ids_writer = open(PATH_TO_NATURAL_QUESTIONS + "train/" + "ids", "w+b")
                    # original_texts_writer = open(PATH_TO_NATURAL_QUESTIONS + "train/" + "original_text", "w+b")
                    with  open("muestra_entrenamiento", "w+b")as f:
                        pickle.dump(archivo_pequeno, f)

                    pickle.dump(X, x_writer)
                    pickle.dump(y, y_writer)
                    pickle.dump(ids, ids_writer)
                    x_writer.close()
                    y_writer.close()
                    ids_writer.close()

                    return PATH_TO_NATURAL_QUESTIONS + "train/"
                else:

                    # x_reader = open(PATH_TO_NATURAL_QUESTIONS +"train/"+ "X", "r+b")
                    # y_reader = open(PATH_TO_NATURAL_QUESTIONS +"train/"+ "Y", "r+b")
                    # ids_reader = open(PATH_TO_NATURAL_QUESTIONS +"train/"+ "ids", "r+b")
                    #
                    # X=pickle.load(x_reader)
                    # y=pickle.load(y_reader)
                    # ids=pickle.load(ids_reader)
                    # x_reader.close()
                    # y_reader.close()
                    # ids_reader.close()
                    return PATH_TO_NATURAL_QUESTIONS + "train/"




def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext
def find_substring(string,substring):
    indexes = []
    for m in re.finditer(substring, string):
             indexes.append((m.start(), m.end()))
    return indexes
def unify_token(tokens):
    newlist = []
    i = 0
    for elem in tokens:
        if "##" in elem and tokens.index(elem)!=0:
            newlist[-1] = newlist[-1].strip() + elem.replace("##","")

        else:
            newlist.append(elem)
    # Elimino la puntuación del final de la lista como del pricipio de la lista para poder buscar mejor la respuesta en el
    for i in newlist[::1]:
        if i in string.punctuation:
            newlist.pop(newlist.index(i))
        else:
            break
    for i in newlist:
        if i in string.punctuation:
            newlist.pop(newlist.index(i))
        else:
            break
    return newlist


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )

    return (matrix[size_x - 1, size_y - 1])


# Storing the sets of punctuation in variable result

def convert2string(tokens):
    clean_tokens = unify_token(tokens)
    salida = ""

    for elem in clean_tokens:
        if elem in ".":
            salida += elem
        else:
            salida += " " + elem
    salida = " ".join(salida.split())

    return salida


def find_answer_index(html_text, answer_tokens, mode=1):
    if mode==1:
        """
        Este modo es suponiendo que la respuesta NO TIENE HTML TOKENS y el texto si.
        """
        answer_string = convert2string(answer_tokens)
        clean_tokens = unify_token(answer_tokens)
        first_word = clean_tokens[0]
        last_word = clean_tokens[-1]
        # cleanr = re.compile('{}.*?{}'.format(first_word,last_word))
        matches = re.findall('{}.*?{}'.format(first_word,last_word),html_text)
        best_match = " "
        minmun_distance = float("Inf")
        for match in matches:
            distance = levenshtein(answer_string,match)
            if distance < minmun_distance:
                minmun_distance = distance
                best_match = match

        start_index = html_text.index(best_match)
        return start_index,start_index+len(best_match)
    elif mode==2:
        """
                Este modo es suponiendo que la respuesta TIENE HTML TOKENS y el texto NO.
                """
        answer_string = convert2string(answer_tokens)
        # clean_tokens = unify_token(answer_tokens)
        # first_word = clean_tokens[0]
        # last_word = clean_tokens[-1]
        #
        # matches = re.findall('{}.*?{}'.format(first_word, last_word), html_text)
        # best_match = " "
        # minmun_distance = float("Inf")
        # for match in matches:
        #     distance = levenshtein(answer_string, match)
        #     if distance < minmun_distance:
        #         minmun_distance = distance
        #         best_match = match
        best_match = cleanhtml(answer_string)
        best_match = " ".join(best_match.split())

        start_index = html_text.index(best_match)
        return start_index, start_index + len(best_match)-1
    else:
        raise Exception("Este modo no es adimitido")

def prueba():
    url_uncased = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1"
    bert_layer = hub.KerasLayer(url_uncased, trainable=False)
    #
    # vocab_file = bert_layer.resolved_object.sp_model_file.asset_path.numpy()
    # tokenizer = FullSentencePieceTokenizer(vocab_file)
    # print(tokenizer.convert_tokens_to_ids([102,1205,367]))
    #
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file)
    del bert_layer

    max_seq_length=350
    f = open("prueba.json","r",encoding="utf8")
    temas = json.load(f)



    question = temas["question_text"]
    text = temas["document_text"].lower()
    clean_text = cleanhtml(text)
    annotations = temas["annotations"]

    byte_start_index = annotations[0]["long_answer"]["start_token"]
    byte_end_index = annotations[0]["long_answer"]["end_token"]

    ## SOn tokens lo que representan los indices
    long_answer = text.split()[byte_start_index: byte_end_index]
    # long_answer = convert2string(long_answer)
    # tokenized_answer = tokenizer.tokenize(long_answer)
    clean_text = convert2string(clean_text.split())
    # Este comando devuelve todas las parejas de indices que contienen dicho substring por eso necesitamos el primer elemento de la última pareja [-1][0]
    final_index = find_substring(clean_text, "see also")[-1][0]

    clean_text = clean_text[:final_index]

    tokenized_text = tokenizer.tokenize(clean_text)
    initial_index, final_index = find_answer_index(clean_text, long_answer, mode=2)

    tokenized_answer = tokenizer.tokenize(clean_text[initial_index:final_index])

    #  ESTO ES PARA ENCONTRAR LOS  INDICE CONSECUTIVOS DE LA RESPUESTA EN EL TEXTO LIMPIO TOKENIZADO
    answer_indexes = [(i, i + len(tokenized_answer)) for i in range(len(tokenized_text)) if
                      tokenized_text[i:i + len(tokenized_answer)] == tokenized_answer]
    final_text = []
    final_text.extend(tokenized_text[answer_indexes[0][0]:answer_indexes[0][1]])
    i = 0
    indice_atras = 0
    indice_adelante = 0
    correcto = False
    while len(final_text) < max_seq_length or not correcto:

        if i == 0:
            final_text.append(tokenized_text[answer_indexes[0][1]+indice_adelante])
            indice_adelante += 1
            i = 1
        if i == 1:
            final_text.insert(0,tokenized_text[answer_indexes[0][0] - indice_atras])
            indice_atras += 1
            i = 0


    if answer_indexes[0][-1] > 350 and annotations[0]["yes_no_answer"] == "NONE":
        # I skip the text that do not have the answer on the first 350 token
        print("Malo")

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""

  return tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(value,[-1])))
  # if isinstance(value,np.ndarray):
  #   return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))
  # else:
  #     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def serialize_example_features(feature0, feature1, feature2, feature3,feature4,feature5):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.

  feature = {
      'context_id': _int64_feature(feature0),
      'context_input_mask': _int64_feature(feature1),
      'context_segmen_id': _int64_feature(feature2),
      'question_id': _int64_feature(feature3),
      'question_input_mask': _int64_feature(feature4),
      'question_segment_id': _int64_feature(feature5),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()
def tf_serialize_example_features(lista_de_features):
  tf_string = tf.py_function(serialize_example_features,(lista_de_features[0],lista_de_features[1],lista_de_features[2],lista_de_features[3],lista_de_features[4],lista_de_features[5]),tf.string)      # the return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The result is a scalar

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)