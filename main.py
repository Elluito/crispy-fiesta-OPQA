import glob
import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
from official.nlp.bert.tokenization import FullTokenizer

# from official.nlp.bert.bert_models import *
from reading_datasets import read_dataset

print(tf.__version__)
BATCH_SIZE=10
tf.compat.v1.enable_eager_execution()
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# capacity=3000
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            print("\n\nENTRE EN LAS GPUS IN PUSE SET MEMORY GROWTH TRUE")
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=capacity*0.8)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# url_uncased="https://tfhub.dev/google/albert_base/3"
# url_uncased= "https://tfhub.dev/tensorflow/albert_en_base/1"
url_uncased="https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(url_uncased,trainable=False)
#
# vocab_file = bert_layer.resolved_object.sp_model_file.asset_path.numpy()
# tokenizer = FullSentencePieceTokenizer(vocab_file)
# print(tokenizer.convert_tokens_to_ids([102,1205,367]))
#
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file)



del bert_layer

# del vocab_file
# del do_lower_case
#
# vocab_file=b'C:\\Users\\LUISAL~1\\AppData\\Local\\Temp\\tfhub_modules\\88ac13afec2955fd14396e4582c251841b67429a\\assets\\vocab.txt'
# tokenizer = FullTokenizer(vocab_file)
def metric_(X,y_true,y_start,y_end):
    promedio_desempeno=0
    i=0
    N = X.shape[0]
    y_true=np.array(y_true)
    f = open("Salida_modelo.txt","w")
    for index in range(N):
        features = X[index,:]
        true_index = y_true[i]
        questions_ids = features[3][0]

        questions_tokens = tokenizer.convert_ids_to_tokens(questions_ids)
        context_tokens = tokenizer.convert_ids_to_tokens(list(features[0][0]))
        context_tokens.pop(0)
        true_ini = np.argmax(true_index[0])
        true_end = np.argmax(true_index[1])
        pred_ini = np.argmax(y_start[i,:])
        pred_end = np.argmax(y_end[i,:])
        A = set(range(true_ini,true_end))
        B = set(range(pred_ini,pred_end))
        denominator = len(A.union(B))
        if denominator==0:
            jaccard_index = 0
        else:
            jaccard_index = len(A.intersection(B))/len(A.union(B))

        promedio_desempeno += (jaccard_index-promedio_desempeno)/(i+1)
        i+=1
        s=""
        for tok in questions_tokens:
            if tok!="[PAD]" and tok!="[SEP]" and tok!="[CLS]":
                s+=tok+" "
        f.write("Question:{} True answer: {}   \n  Predicted_answer: {}       Jaccard: {}  \n".format(s,context_tokens[true_ini:true_end+1],context_tokens[pred_ini:pred_end+1],jaccard_index))


    f.write("\nPerformance promedio {}".format(promedio_desempeno))
    f.close()



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
    loss_object1 = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_object2 = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # entrada_total={"questions_id": elem[3], "question_input_mask":elem[4], "question_segment_id": elem[5],"context_id": elem[0], "context_input_mask": elem[1], "context_segment_id":elem[2]}
    # y_1=[]
    # y_2=[]
    # for elem in list(x):
    #     entrada={"questions_id": elem[3], "question_input_mask":elem[4], "question_segment_id": elem[5],"context_id": elem[0], "context_input_mask": elem[1], "context_segment_id":elem[2]}
    #     y1,y2 = model(entrada,training=True)
    #     y_1.append(np.squeeze(y1))
    #     y_2.append(np.squeeze(y2))


    entrada = {"questions_id": np.squeeze(x[:, 3]), "question_input_mask": np.squeeze(x[:, 4]),
           "question_segment_id": np.squeeze(x[:, 5]), "context_id": np.squeeze(x[:, 0]),
           "context_input_mask": np.squeeze(x[:, 1]), "context_segment_id": np.squeeze(x[:, 2])}


    y1, y2 = model(entrada)
    loss1=loss_object1(y_true=np.squeeze(y[:,0]), y_pred=y1)
    loss2 = loss_object2(y_true=np.squeeze(y[:,1]), y_pred=y2)
    return  loss1,loss2,y1,y2

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value1,loss_value2,y1,y2 = loss(model, inputs, targets, training=True)
  return loss_value1,loss_value2, tape.gradient([loss_value1,loss_value2],model.trainable_variables),y1,y2
def positional_embedding(pos, model_size):
    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return PE



def train_model(model,path_to_features,log_name,model_name,batch_size=32,step_per_epoch=10,epochs=10):
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.001)
    # train_dataset =tf.data.Dataset.from_tensor_slices((X,Y)).batch(batch_size=batch_size).repeat().shuffle(1000)
    print("Voy a empezar el entrenamiento")
    for epoch in range(epochs):

        epoch_loss_avg1 = tf.keras.metrics.Mean()
        epoch_loss_avg2 = tf.keras.metrics.Mean()
        epoch_accuracy_start = tf.keras.metrics.CategoricalAccuracy()
        epoch_accuracy_end = tf.keras.metrics.CategoricalAccuracy()
        # Training loop - using batches of 32

        for i in range(step_per_epoch):
            x,y=crear_batch(path_to_features,batch_size)
            # Optimize the model

            loss_value1,loss_value2, grads,y1,y2 = grad(model, x, y)

            # print(grads)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # optimizer.apply_gradients(zip(grads2, model.trainable_variables))

            # Track progress
            epoch_loss_avg1(loss_value1)  # Add current batch loss
            epoch_loss_avg2(loss_value2)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).

            # entrada = {"questions_id": np.squeeze(x[:, 3]), "question_input_mask": np.squeeze(x[:, 4]),
            #            "question_segment_id": np.squeeze(x[:, 5]), "context_id": np.squeeze(x[:, 0]),
            #            "context_input_mask": np.squeeze(x[:, 1]), "context_segment_id": np.squeeze(x[:, 2])}
            # y1,y2= model(entrada, training=True)

            epoch_accuracy_start(y[:,0],y1)
            epoch_accuracy_end(y[:, 1], y2)
            # print("Log_end: " + str(y2.numpy()))
        if epoch%10 == 0:
            model.save(model_name)



        # End epoch
        # train_loss_results.append(epoch_loss_avg.result())
        # train_accuracy_results_start.append(epoch_accuracy_start.result())
        # train_accuracy_results_end.append(epoch_accuracy_end.result())

        # if epoch % 50 == 0:
        f=open(log_name,"a")
        print("Epoch {:03d}: Loss start : {:.3f},Loss end :  {:0.3f}, Accuracy_for_start: {:.3%} ,  Accuracy_for_end: {:.3%}".format(epoch,epoch_loss_avg1.result(),epoch_loss_avg2.result(),epoch_accuracy_start.result(),epoch_accuracy_end.result()))

        f.write("Epoch {:03d}: Loss start : {:.3f},Loss end :  {:0.3f}, Accuracy_for_start: {:.3%} ,  Accuracy_for_end: {:.3%}".format(epoch,epoch_loss_avg1.result(),epoch_loss_avg2.result(),epoch_accuracy_start.result(),epoch_accuracy_end.result()))
        # f.write("Log_end: " + str(y2.numpy()))
        f.close()

def build_model(max_seq_length = 512 ):
    question_input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="questions_id")
    question_input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="question_input_mask")
    question_segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="question_segment_id")

    context_input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="context_id")
    context_input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="context_input_mask")
    context_segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="context_segment_id")

    # albert_inputs1 = dict(
    #     input_ids=question_input_word_ids,
    #     input_mask=question_input_mask,
    #     segment_ids=question_segment_ids)
    # albert_outputs = albert_module(albert_inputs1, signature="tokens", as_dict=True)
    # # question_pooled_output = albert_outputs["pooled_output"]
    # question_sequence_output = albert_outputs["sequence_output"]
    #
    # albert_inputs2 = dict(
    #     input_ids=context_input_word_ids,
    #     input_mask=context_input_mask,
    #     segment_ids=context_segment_ids)
    # albert_outputs = albert_module(albert_inputs2, signature="tokens", as_dict=True)
    # context_sequence_output = albert_outputs["sequence_output"]

    bert_layer = hub.KerasLayer(url_uncased,trainable=False, name="Bert_variant_model")

    question_pooled_output, question_sequence_output = bert_layer([question_input_word_ids, question_input_mask, question_segment_ids])

    context_pooled_output, context_sequence_output = bert_layer([context_input_word_ids, context_input_mask, context_segment_ids])

    activation = keras.activations.elu
    substring=[i for i in url_uncased.split("_") if "H-" in i]
    if substring==[]:
        dim=128
    else:
        substring=substring[0]
        dim=[int(s) for s in substring.split("-") if s.isdigit()][0]

    similarity_matrix = 1 / ( dim** (1 / 2)) * tf.matmul(activation(question_sequence_output),activation(context_sequence_output),transpose_b=True,name="Attention_matmul")
    temp = tf.math.reduce_max(similarity_matrix, axis=1,keepdims=True,name="Reduction_of_similarity_function")

    temp = tf.math.softmax(temp)
    attention_from_question_to_context = tf.math.multiply(context_sequence_output, tf.transpose(temp,[0,2,1]))
    self_attention_context = keras.layers.Attention(name="Self_attention_paragraph")([context_sequence_output,context_sequence_output])
    attention_from_context_to_question = keras.layers.Attention(name="Attention_from_context_to_question")([context_sequence_output,question_sequence_output])


    # new_representation = keras.layers.BatchNormalization()(new_representation)
    # layer_encoder_start = keras.layers.Bidirectional(LSTM(120,activation="tanh", return_sequences=True, input_shape=(max_seq_length,dim)),merge_mode='sum')
    #
    # layer_decoder_start= keras.layers.Bidirectional(LSTM(120, activation="tanh",return_sequences=True, input_shape=(max_seq_length, 120)), merge_mode='sum')
    #
    # layer_encoder_end = keras.layers.Bidirectional(
    #     LSTM(120, activation="tanh", return_sequences=True, input_shape=(max_seq_length, dim)), merge_mode='sum')
    #
    # layer_decoder_end = keras.layers.Bidirectional(
    #     LSTM(120, activation="tanh", return_sequences=True, input_shape=(max_seq_length, 120)), merge_mode='sum')

    # Hago el positional embedding
    pes = []
    for i in range(max_seq_length):
        pes.append(positional_embedding(i, dim))

    pes = np.concatenate(pes, axis=0)
    # pes = tf.constant(pes, dtype=tf.float32)
    new_pes =[]
    for i in range(BATCH_SIZE):
        new_pes .append(pes)

    pes = tf.constant(new_pes, dtype=tf.float32)

    #Sumo el positional embedding con cada una de las salidas

    attention_from_context_to_question += pes
    self_attention_context += pes
    attention_from_question_to_context += pes


    temp = attention_from_context_to_question+attention_from_question_to_context+self_attention_context
    temp1  = keras.layers.Dense(max_seq_length)(keras.layers.Flatten()(temp))
    temp2  = keras.layers.Dense(max_seq_length)(keras.layers.Flatten()(temp))
    # soft_max_salida_start =keras.layers.Dense(max_seq_length)(attention_from_question_to_context)+ keras.layers.Dense(max_seq_length)(attention_from_context_to_question)+keras.layers.Dense(max_seq_length)(self_attention_context)

    soft_max_salida_start = keras.layers.BatchNormalization()(temp1)
    soft_max_salida_start = keras.layers.Activation("softmax")(soft_max_salida_start )

    soft_max_salida_end = temp2
    soft_max_salida_end = keras.layers.BatchNormalization()(soft_max_salida_end)
    soft_max_salida_end = keras.layers.Activation("softmax")(soft_max_salida_end)


    # mid_start  = layer_encoder_start(new_representation)

    # output_for_start = layer_decoder_start(mid_start)
    # mid_end = layer_encoder_end(new_representation)
    # output_for_end = layer_decoder_end(mid_end)

    # output_for_start = tf.reshape(output_for_start,[-1,max_seq_length*120])
    # output_for_end = tf.reshape(output_for_end,[-1,max_seq_length*120])
    # soft_max_start =keras.layers.Dense(max_seq_length,activation="softmax",name="output_logits_for_start")(output_for_start)
    # soft_max_end = keras.layers.Dense(max_seq_length,activation="softmax",name="output_logits_for_end")(output_for_end)

    # _,out=tf.shape(output_start).numpy()

    # W1 = tf.keras.backend.variable(init_weights(128,1),dtype=tf.float32,name="weights_for_start")
    # W1 = init_weights(128,1)
    # output_end=tf.reshape(output_for_end,[-1,max_seq_length,128])
    # _,out=tf.shape(output_end).numpy()
    # W2 = tf.keras.layers.Dense(max_seq_length,name="weights_for_end",activation="softmax")
    # W2=tf.keras.backend.variable(init_weights(128,1),dtype=tf.float32,name="weights_for_end")
    # W2 =init_weights(128,1)


    #
    # temp_start = tf.reshape(tf.matmul(output_start,W1),[-1,max_seq_length])
    # temp_end = tf.reshape(tf.matmul(output_end,W2),[-1,max_seq_length])
    



    # soft_max_start=tf.reshape(soft_max_start,[-1,max_seq_length],name="start_output")
    # soft_max_end=tf.reshape(soft_max_end,[-1,max_seq_length],name="end_output")


    # logits_for_start = tf.math.log(soft_max_start,name="log_start")
    # logits_for_end = tf.math.log(soft_max_end,name="log_end")
    model = keras.Model(inputs=[question_input_word_ids, question_input_mask, question_segment_ids, context_input_word_ids,context_input_mask, context_segment_ids], outputs=[ soft_max_salida_start,soft_max_salida_end],name="Luis_net")

    # model.build(input_shape=[None,None])
    optim=keras.optimizers.Adam(lr=0.05,beta_2=0.98)
    model.compile(optimizer=optim,loss=[create_metric(max_seq_length),create_metric(max_seq_length)],
                                        metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.CategoricalAccuracy()])
    model.summary()



    return model

def macro_f1(y, y_hat, thresh = 0):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

def create_metric(number_clases,label_smoothing=0):

    @tf.function
    def custom_metric(y_true,y_pred):
        y_true =(1-label_smoothing)*y_true+ label_smoothing/number_clases
        multpli = tf.multiply(y_true,y_pred)
        result = -keras.backend.sum(multpli)
        return result
    return custom_metric

def crear_batch(path_to_features,fragmented=False,batchsize=32):
    if fragmented:
        elems=len(glob.glob(path_to_features+"X_*"))
        indice=int(np.random.randint(0,elems,1))
        with open(path_to_features+"X_{}".format(indice),"r+b") as f:
            X= np.array(pickle.load(f))
        with open(path_to_features+"Y_{}".format(indice),"r+b") as f:
            Y= np.array(pickle.load(f))
        indices=np.random.randint(0,len(X),batchsize)
        return X[indices,:],Y[indices,:]
    else:

        with open(path_to_features + "X", "r+b") as f:
            X = np.array(pickle.load(f))
        with open(path_to_features + "Y", "r+b") as f:
            Y = np.array(pickle.load(f))

        return X, Y




# natural_questions_dataset_path ="D:\datsets_tesis\Kaggle_competition\Tensorflow_Q_and_A_competition/"

#


# tf.enable_eager_execution()
# init_op = tf.initialize_all_variables()

# Later, when launching the model

#
max_seq_length = 350# Your choice here.

print("VOY A HACER EL MODELO")

# keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())
model = build_model(max_seq_length)

print("YA HICE EL MODELO")




# X_train,X_test,y_train,y_test,ids_train,ids_test=train_test_split(X,y,ids,test_size=0.1)

# N=len(X_train)
# X_train=np.array(X_train)
# string=serialize_example_features(X[0][0],X[0][1],X[0][2],X[0][3],X[0][4],X[0][5])
# example_proto = tf.train.Example.FromString(string)
#
#
# ########################  ASÍ SE RECUPERA EL FEATURE POR SI LO NECESITO MÁS TARDE##############################
# # list(example_proto.features.feature["question_id"].int64_list.value)
# ###############################################################################################################
# #
# print(string)

# print(features_dataset)
# serialized_features_dataset = features_dataset.map(tf_serialize_example_features)
# # print(serialized_features_dataset)
# #
# filename = 'x_test.tfrecord'
# writer = tf.data.experimental.TFRecordWriter(filename)
# writer.write(serialized_features_dataset)
# writer.close()
# filenames = [filename]
# train_dataset=tf.data.TFRecordDataset('x_test.tfrecord').batch(32)
# for serialized_example in train_dataset:
#     for elem in serialized_example
#         example = tf.train.Example()
#         example.ParseFromString(elem)
#         x_1 = np.array(example.features.feature['X'].float_list.value)
#         y_1 = np.array(example.features.feature['Y'].float_list.value)
#         break
#

path= read_dataset(mode="train",tokenizer=tokenizer,max_seq_length=max_seq_length,fragmented=False)
#
import time
t = time.time()
log_name = "Salida_modelo_{}.txt".format(t)
x,y = crear_batch(path,fragmented=False)
N = 5000 # len(x)
entrada = {"questions_id": np.squeeze(x[:N, 3].astype(np.int32)), "question_input_mask": np.squeeze(x[:N, 4].astype(np.int32)),
           "question_segment_id": np.squeeze(x[:N, 5].astype(np.int32)), "context_id": np.squeeze(x[:N, 0].astype(np.int32)),
           "context_input_mask": np.squeeze(x[:N, 1].astype(np.int32)), "context_segment_id": np.squeeze(x[:N, 2].astype(np.int32))}
salida=[y[:N,0],y[:N,1]]

# entrada = {"questions_id": np.squeeze(X_test[:2000, 3]), "question_input_mask": np.squeeze(X_test[:2000, 4]),
#            "question_segment_id": np.squeeze(X_test[:2000, 5]), "context_id": np.squeeze(X_test[:2000, 0]),
#            "context_input_mask": np.squeeze(X_test[:2000, 1]), "context_segment_id": np.squeeze(X_test[:2000, 2])}



model_callback=tf.keras.callbacks.ModelCheckpoint("local_model/model_e{epoch}-val_loss{val_loss:.4f}.hdf5",save_best_only=True)
# tensor_callback=keras.callbacks.TensorBoard("logs",batch_size=5)

early_callback_start=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, verbose=0, mode='auto', restore_best_weights=True
)
# model.load_weights("local_model/model_e2-val_loss7.0668.hdf5")
reduce_learning = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0)
model.fit(entrada,salida,batch_size=BATCH_SIZE,validation_split=0.1,epochs=20,callbacks=[model_callback,early_callback_start,reduce_learning],verbose=2)

# # train_model(model,path_to_features=path,model_name="model_{}.h5".format(t),batch_size=7,epochs=1,log_name=log_name)
#
# model.save_weights("modelo_prueba{}.hdf5".format(t))
path = read_dataset(mode="test",tokenizer=tokenizer,max_seq_length=max_seq_length,fragmented=False)
X_test,y_test = crear_batch(path,fragmented=False)
# X_test,y_test = X_test[:10,:],y_test[:10,:]
entrada = {"questions_id": np.squeeze(X_test[:, 3].astype(np.int32)), "question_input_mask": np.squeeze(X_test[:, 4].astype(np.int32)),
           "question_segment_id": np.squeeze(X_test[:, 5].astype(np.int32)), "context_id": np.squeeze(X_test[:, 0].astype(np.int32)),
           "context_input_mask": np.squeeze(X_test[:, 1].astype(np.int32)), "context_segment_id": np.squeeze(X_test[:, 2].astype(np.int32))}
# model.load_weights("local_model/model_e10-val_loss82.3117.hdf5")
y_start,y_end = model.predict(entrada)


with open("y_pred_end","w+b") as f :
    pickle.dump(y_end,f)
with open("y_pred_start","w+b") as f :
    pickle.dump(y_start,f)

#
# with open("y_pred_end","r+b") as f :
#     y_end = pickle.load(f)
# with open("y_pred_start","r+b") as f :
#     y_start = pickle.load(f)



#

metric_(X_test,y_test,y_start,y_end)
