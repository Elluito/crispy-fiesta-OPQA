import json
import os
import pickle

import numpy as np
import tensorflow as tf

PATH_TO_SQUAD="datasets/Squad/"
PATH_TO_NARRATIVEQA_SHORT="datasets/NARRATIVEQA/"
PATH_TO_NARRATIVEQA_FULL="datasets/NARRATIVEQA/"
PATH_TO_NATURAL_QUESTIONS="D:\datsets_tesis\Kaggle_competition\Tensorflow_Q_and_A_competition/"

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


def read_dataset(dataset="squad",mode="test",fragmented=True,tokenizer=None,max_seq_length=512):



    if dataset=="squad":
        if mode=="test":
            if fragmented:
                i=0
                if not os.listdir(PATH_TO_SQUAD+"test"):


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
                                if len(text_tokens)>max_seq_length:
                                    continue
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
                                            text_answer_list = ans["text"].split()
                                            first_word = tokenizer.tokenize(text_answer_list[0])[0]
                                            last_word = tokenizer.tokenize(text_answer_list[-1])[0]
                                            first_index = text_tokens.index(first_word)
                                            last_index = text_tokens.index(last_word)
                                        except:
                                            continue

                                    Q_id, Q_mask, Q_segment = convert_sentence_to_features(question["question"], tokenizer,
                                                                                           max_seq_length)
                                    temp_y_start = np.zeros(max_seq_length)
                                    temp_y_start[first_index] = 1
                                    temp_y_end = np.zeros(max_seq_length)
                                    temp_y_end[last_index] = 1
                                    # dictionary={"questions_id":Q_id,"question_input_mask":Q_mask,"question_segment_id":Q_segment,"context_id":C_id,"context_input_mask":C_mask,"context_segment_id":C_segment}
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
                    #
                    # x_reader = open(PATH_TO_SQUAD + "test/" + "X", "r+b")
                    # y_reader = open(PATH_TO_SQUAD + "test/" +"Y", "r+b")
                    # ids_reader = open(PATH_TO_SQUAD + "test/" +"ids", "r+b")

                    # X=pickle.load(x_reader)
                    # y=pickle.load(y_reader)
                    # ids=pickle.load(ids_reader)
                    # x_reader.close()
                    # y_reader.close()
                    # ids_reader.close()
                    return PATH_TO_SQUAD + "test/"



        elif mode=="train":
            if fragmented:
                i = 0
                if not os.listdir(PATH_TO_SQUAD + "test"):

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
                                            text_answer_list = ans["text"].split()
                                            first_word = tokenizer.tokenize(text_answer_list[0])[0]
                                            last_word = tokenizer.tokenize(text_answer_list[-1])[0]
                                            first_index = text_tokens.index(first_word)
                                            last_index = text_tokens.index(last_word)
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
                    # X=np.array(X)
                    # y=np.array(y)
                    # dictionary = {"questions_id": X[:,3], "question_input_mask": X[:,4], "question_segment_id": X[:,5],"context_id": X[:,0], "context_input_mask": X[:,1], "context_segment_id": X[:,2]}

                    return PATH_TO_SQUAD + "test/"
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
                                            text_answer_list = ans["text"].split()
                                            first_word = tokenizer.tokenize(text_answer_list[0])[0]
                                            last_word = tokenizer.tokenize(text_answer_list[-1])[0]
                                            first_index = text_tokens.index(first_word)
                                            last_index = text_tokens.index(last_word)
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
        pass
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