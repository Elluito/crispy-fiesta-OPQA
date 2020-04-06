import tensorflow_hub as hub

import tensorflow as tf
import  tensorflow.keras as keras
from models import MyDenseLayer
from tensorflow.keras.layers import LSTM
from official.nlp.bert.tokenization import FullTokenizer
from official.nlp.bert.bert_models import *
from reading_datasets import read_dataset
import numpy as np
import os
from  sklearn.model_selection import train_test_split
# tf.compat.v1.enable_eager_execution()
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
url_uncased= "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
url="https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(url_uncased,
                            trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)
del bert_layer
# del vocab_file
# del do_lower_case
#
# vocab_file=b'C:\\Users\\LUISAL~1\\AppData\\Local\\Temp\\tfhub_modules\\88ac13afec2955fd14396e4582c251841b67429a\\assets\\vocab.txt'
# tokenizer = FullTokenizer(vocab_file)
def metric_(X,y_true,y_pred):
    promedio_desempeno=0
    i=0
    for features,true_index,pred_index in zip(X,y_true,y_pred):
        questions_ids=features[3]
        questions_tokens=tokenizer.convert_ids_to_tokens(questions_ids)
        context_tokens=tokenizer.convert_ids_to_tokens(features[0])
        true_ini=np.argmax(true_index[0])
        true_end=np.argmax(true_index[1])
        pred_ini=np.argmax(pred_index[0])
        pred_end=np.argmax(pred_index[1])
        A=set(range(true_ini,true_end))
        B=set(range(pred_ini,pred_end))
        jaccard_index=len(A.intersection(B))/len(A.union(B))
        promedio_desempeno+= (jaccard_index-promedio_desempeno)/i
        i+=1
        s=""
        for tok in questions_tokens:
            s+ tok+" "
        print("Question:{} True answer: {}     Predicted_answer: {}       Jaccard: {}".format(s,context_tokens[true_ini:true_end],context_tokens[true_ini:true_end],jaccard_index))


    print("Performance promedio {}".format(promedio_desempeno))



def init_weights(nin,nout):  # Glorot normal
  W=np.random.rand(nin,nout)

  sd = np.sqrt(2.0 / (nin + nout))
  for i in range(nin):
    for j in range(nout):
      x = np.float32(np.random.uniform(-sd, sd))
      W[i,j] = x
  return W
def convert_two_sentences_to_features(sentence1,sentence2, tokenizer, max_seq_len):
    tokens = ['[CLS]']
    tokens.extend(tokenizer.tokenize(sentence1))
    if len(tokens) > max_seq_len - 1:
        tokens = tokens[:max_seq_len - 1]
    tokens.append('[SEP]')
    tokens.extend(tokenizer.tokenize(sentence2))
    if len(tokens) > max_seq_len - 1:
        tokens = tokens[:max_seq_len - 1]
    segment_ids=[0]*len(tokenizer.tokenize(sentence1))
    segment_ids.extend([1] * len(tokenizer.tokenize(sentence2)))
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # # Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len - len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)

    return np.array(input_ids),np.array(input_mask), np.array(segment_ids)
def log_loss_function(y_true,y_pred):
    suma=0

    for i in range(y_true.shape[0]):
         init=y_true[i,0]
         end=y_true[i,1]

         suma+=y_pred[i][0][init]+y_pred[i][1][end]

    return suma
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

    return np.array(input_ids).reshape(1,max_seq_length),np.array(input_mask).reshape(1,max_seq_length), np.array(segment_ids).reshape(1,max_seq_length)


def convert_sentences_to_features(sentences, tokenizer, max_seq_len=20):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)

    return all_input_ids, all_input_mask, all_segment_ids
def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  y1,y2 = model(list(map(list,x)),training=True)
  print(y1)
  print(y2)
  loss=loss_object(y_true=y[:,0], y_pred=y1)+loss_object(y_true=y[:,1], y_pred=y2)

  return loss

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train_model(model,X,Y,batch_size=32,step_per_epoch=10,epochs=10):
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.000121)
    train_dataset =tf.data.Dataset.from_tensor_slices((X,Y)).batch(batch_size=batch_size).repeat().shuffle(1000)
    train_loss_results=[]
    train_accuracy_results=[]

    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        # Training loop - using batches of 32
        i=0
        for x, y in train_dataset:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy(y, model(x, training=True))
            i+=1
            if i>step_per_epoch:
                break


        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        # if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,epoch_loss_avg.result(),epoch_accuracy.result()))


def build_model(max_seq_length = 512 ):
    question_input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="questions_id")
    question_input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="question_input_mask")
    question_segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="question_segment_id")

    context_input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="context_id")
    context_input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="context_input_mask")
    context_segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="context_segment_id")



    bert_layer = hub.KerasLayer(url_uncased, trainable=True, name="Bert_model")

    question_pooled_output, question_sequence_output = bert_layer([question_input_word_ids, question_input_mask, question_segment_ids])
    # print(tf.shape(question_sequence_output))
    context_pooled_output, context_sequence_output = bert_layer([context_input_word_ids, context_input_mask, context_segment_ids])
    # print(tf.shape(context_sequence_output))
    activation = keras.activations.elu
    substring=[i for i in url_uncased.split("_") if "H-" in i][0]
    dim=[int(s) for s in substring.split("-") if s.isdigit()][0]

    similarity_matrix = 1 / ( dim** (1 / 2)) * tf.matmul(activation(question_sequence_output),activation(context_sequence_output),transpose_b=True,name="Attention_matmul")
    temp = tf.math.reduce_max(similarity_matrix, axis=1,keepdims=True,name="Reduction_of_similarity_function")

    temp = tf.math.softmax(temp)
    new_representation =tf.math.multiply(context_sequence_output, tf.transpose(temp,[0,2,1]))

    layer_encoder = keras.layers.Bidirectional(LSTM(128, return_sequences=True, input_shape=(max_seq_length,dim)),merge_mode='ave')

    layer_decoder = keras.layers.Bidirectional(LSTM(128, return_sequences=True, input_shape=(max_seq_length, 128)), merge_mode='ave')

    output_for_start = layer_encoder(new_representation)
    # encoder_state_c = ecoder_state_c_forth + ecoder_state_c_back
    # encoder_state_h = ecoder_state_h_forth + ecoder_state_h_back
    # encoder_state = [encoder_state_h, encoder_state_c]
    output_for_end = layer_decoder(output_for_start)

    output_start=tf.reshape(output_for_start,[-1,max_seq_length,128])
    # _,out=tf.shape(output_start).numpy()

    W1 = tf.keras.backend.variable(init_weights(128,1),dtype=tf.float32,name="weights_for_start")
    output_end=tf.reshape(output_for_end,[-1,max_seq_length,128])
    # _,out=tf.shape(output_end).numpy()
    # W2 = tf.keras.layers.Dense(max_seq_length,name="weights_for_end",activation="softmax")
    W2=tf.keras.backend.variable(init_weights(128,1),dtype=tf.float32,name="weights_for_end")
    temp_start = tf.reshape(tf.matmul(output_start,W1),[-1,max_seq_length])
    temp_end = tf.reshape(tf.matmul(output_end,W2),[-1,max_seq_length])
    soft_max_start = tf.nn.softmax(temp_start)
    soft_max_end = tf.nn.softmax(temp_end)

    soft_max_start=tf.reshape(soft_max_start,[-1,max_seq_length],name="start_output")
    soft_max_end=tf.reshape(soft_max_end,[-1,max_seq_length],name="end_output")


    logits_for_start = tf.math.log(soft_max_start,name="log_start")
    logits_for_end = tf.math.log(soft_max_end,name="log_end")
    model = keras.Model(inputs=[question_input_word_ids, question_input_mask, question_segment_ids, context_input_word_ids,context_input_mask, context_segment_ids], outputs=[logits_for_end, logits_for_start],name="Luis_net")
    model.build(input_shape=[None,6,1,max_seq_length])
    # model.compile(optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.0001),loss=[tf.keras.losses.CategoricalCrossentropy(),tf.keras.losses.CategoricalCrossentropy()])
    model.summary()
    return model
    # optim=keras.optimizers.Adam(lr=0.0001)
    # model.compile(optimizer=optim,loss=log_loss_function)

# natural_questions_dataset_path ="D:\datsets_tesis\Kaggle_competition\Tensorflow_Q_and_A_competition/"

#


# tf.enable_eager_execution()
# init_op = tf.initialize_all_variables()

# Later, when launching the model


max_seq_length = 512 # Your choice here.
mex_text_length=512*4
model=build_model(max_seq_length)
# print(model.weights)
# with tf.Session() as sess:
#   # Run the init operation.
#   sess.run(init_op)
X,y,ids= read_dataset(tokenizer=tokenizer,max_seq_length=max_seq_length)


X_train,X_test,y_train,y_test,ids_train,ids_test=train_test_split(X,y,ids,test_size=0.1)

N=len(X_train)
X_train=np.array(X_train)
# entrada=[X_train[0,0,0][:512].reshape(1,512),X_train[0,1,0][:512].reshape(1,512),X_train[0,2,0][:512].reshape(1,512),X_train[0,3,0][:512].reshape(1,512),X_train[0,4,0][:512].reshape(1,512),X_train[0,5,0][:512].reshape(1,512)]
prob_start,prob_end=model(X[:3],training=False)
print(prob_end)
print(prob_start)
train_model(model,X_train,y_train)
# X_test_= np.array(X_test)
# X_train = {"questions_id": X_train[:,3].reshape(-1,max_seq_length), "question_input_mask": X_train[:,4].reshape(-1,max_seq_length), "question_segment_id": X_train[:,5].reshape(-1,max_seq_length),"context_id": X_train[:,0].reshape(-1,max_seq_length), "context_input_mask": X_train[:,1].reshape(-1,max_seq_length), "context_segment_id": X_train[:,2].reshape(-1,max_seq_length)}
# X_test_pre ={"questions_id": X_test_[:,3].reshape(-1,max_seq_length), "question_input_mask": X_test_[:,4].reshape(-1,max_seq_length), "question_segment_id": X_test_[:,5].reshape(-1,max_seq_length),"context_id": X_test_[:,0].reshape(-1,max_seq_length), "context_input_mask": X_test_[:,1].reshape(-1,max_seq_length), "context_segment_id": X_test_[:,2].reshape(-1,max_seq_length)}
# y_train_array=np.array(y_train)
# y_train={"tf_op_layer_start_output":y_train_array[:,0].reshape(-1,1,max_seq_length),"tf_op_layer_end_output":y_train_array[:,1].reshape(-1,1,max_seq_length)}
# model.fit(X_train,y_train,batch_size=35,epochs=10)
# model.save("modelo_prueba.h5")
# y_pred=model.predict(X_test_pre)
# cosas =""
# metric_(X_test,y_test,y_pred)
#

