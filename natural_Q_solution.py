import tensorflow_hub as hub
import tensorflow as tf
from reading_datasets import read_dataset
# from main import convert_two_sentences_to_features
from official.nlp.bert.tokenization import FullTokenizer
import numpy as np
tf.enable_eager_execution()
model = hub.load("https://tfhub.dev/prvi/tf2nq/1")
vocab_file=b'C:\\Users\\LUISAL~1\\AppData\\Local\\Temp\\tfhub_modules\\88ac13afec2955fd14396e4582c251841b67429a\\assets\\vocab.txt'
tokenizer = FullTokenizer(vocab_file, do_lower_case=True)
frase_de_kaggle=[101,2040,2003,1996,2148,3060,2152,5849,1999,2414, 102, 998
, 991, 987, 988,2152,3222,1997,2148,3088,1999,2414, 967, 966
, 987, 988,3295, 967, 986,19817,10354,2389,6843,2675,1010,2414
, 965, 966, 987, 988,4769, 967, 986,19817,10354,2389,6843,2675
,1010,2414,1010,15868,2475,2078,1019,18927, 965, 966, 987, 988
,12093, 967, 986,4868,1080,2382,1531,2382,1005,1005,1050,1014
,1080,5718,1531,4261,1005,1005,1059,1013,4868,1012,2753,2620
,2475,1080,1050,1014,1012,14010,2683,1080,1059,1013,4868,1012
,2753,2620,2475,1025,1011,1014,1012,14010,2683,12093,1024,4868
,1080,2382,1531,2382,1005,1005,1050,1014,1080,5718,1531,4261
,1005,1005,1059,1013,4868,1012,2753,2620,2475,1080,1050,1014
,1012,14010,2683,1080,1059,1013,4868,1012,2753,2620,2475,1025
,1011,1014,1012,14010,2683, 965, 966, 987, 988,2152,5849, 967
, 986,10030, 965, 966, 970,11673,1997,2148,3088,2160, 985,1996
,2152,3222,1997,2148,3088,1999,2414,2003,1996,8041,3260,2013
,2148,3088,2000,1996,2142,2983,1012,2009,2003,2284,2012,2148
,3088,2160,1010,1037,2311,2006,19817,10354,2389,6843,2675,1010
,2414,1012,2004,2092,2004,4820,1996,4822,1997,1996,2152,5849
,1010,1996,2311,2036,6184,1996,2148,3060,19972,1012,2009,2038
,2042,1037,3694,2462,1008,3205,2311,2144,3196,1012, 964, 982
,8417, 961, 994, 992,1015,2381, 971, 992,1016,2156,2036, 971
, 992,1017,7604, 971, 992,1018,6327,6971, 971, 973, 982,2381
,1006,10086,1007, 961, 985,2148,3088,2160,2001,2328,2011,7935
,1010,7658,10224,1004,21987,12474,2015,1999,1996,5687,2006,1996
,2609,1997,2054,2018,2042,20653,1005,1055,3309,2127,2009,2001
,7002,1999,4266,1012,1996,2311,2001,2881,2011,2909,7253,6243
,1010,2007,6549,6743,2011,24873,5339,26261,6038,4059,1998,2909
,2798,12819,1010,1998,2441,1999,4537,1012,1996,2311,2001,3734
,2011,1996,2231,1997,2148,3088,2004,2049,2364,8041,3739,1999
,1996,2866,1012,2076,2088,2162,2462,1010,3539,2704,5553,15488
,16446,2973,2045,2096,9283,2148,3088,1005,1055,2162,3488,1012
, 964, 985,1999,3777,1010,2148,3088,2150,1037,3072,1010,1998
,6780,2013,1996,5663,2349,2000,2049,3343,1997,5762,18771,1012
,11914,1010,1996,2311,2150,2019,8408,1010,2738,2084,1037,2152
,3222,1012,2076,1996,3865,1010,1996,2311,1010,2029,2001,2028
,1997,1996,2069,2148,3060,8041,6416,1999,1037,2270,2181,1010
,2001,9416,2011,13337,2013,2105,1996,2088,1012,2076,1996,2901
,8554,4171,12925,1010,1996,2311,2001,2275,4862,13900,2011,11421
,2545,1010,2348,2025,5667,5591,1012, 964, 985,1996,2034,3929
,2489,3537,3864,1999,2148,3088,2020,2218,2006,1996,2676,2258
,2807,1010,1998,1018,2420,2101,1010,1996,2406,14311,1996,5663
,1010,3943,2086,2000,1996,2154,2044,2009,6780,2588,3352,1037
,3072,1012,2247,2007,2406,1005, 977, 102]
print(tokenizer.convert_ids_to_tokens(frase_de_kaggle))

model.summary()

def output(unique_id ,model_output ,n_keep=10):
    pos_logits ,ans_logits ,long_mask ,short_mask ,cross = model_output
    long_span_logits =  pos_logits
    mask = tf.cast(tf.expand_dims(long_mask ,-1) ,long_span_logits.dtype)
    long_span_logits = long_span_logits - 10000 * mask
    long_p = tf.nn.softmax(long_span_logits ,axis=1)
    short_span_logits = pos_logits
    short_span_logits -= 10000 *tf.cast(tf.expand_dims(short_mask ,-1) ,short_span_logits.dtype)
    start_logits ,end_logits = short_span_logits[: ,: ,0] ,short_span_logits[: ,: ,1]
    batch_size ,seq_length = short_span_logits.shape[0] ,short_span_logits.shape[1]
    seq = tf.range(seq_length)
    i_leq_j_mask = tf.cast(tf.expand_dims(seq ,1 ) >tf.expand_dims(seq ,0) ,short_span_logits.dtype)
    i_leq_j_mask = tf.expand_dims(i_leq_j_mask ,0)
    logits  = tf.expand_dims(start_logits ,2 ) +tf.expand_dims(end_logits ,1 ) +cross
    logits -= 10000 *i_leq_j_mask
    logits  = tf.reshape(logits, [batch_size ,seq_length *seq_length])
    short_p = tf.nn.softmax(logits)
    indices = tf.argsort(short_p ,axis=1 ,direction='DESCENDING')[: ,:n_keep]
    short_p = tf.gather(short_p ,indices ,batch_dims=1)
    return dict(unique_id = unique_id,
                ans_logits= ans_logits,
                long_p    = long_p,
                short_p   = short_p,
                short_p_indices = indices)



test=read_dataset()
predictions={}

for tema in test:
    for cosa in tema["paragraphs"]:
        text=cosa["context"]
        for question in cosa["qas"]:
            unique_id=question["id"]
            text_tokens = tokenizer.tokenize(text)

            question_len=np.array([len(tokenizer.tokenize(question["question"]))])
            data_len = np.array([512])
            input_word_ids,input_mask,segment_id =convert_two_sentences_to_features(question["question"],text,tokenizer,512)


            c=[data_len.reshape(-1,),input_word_ids.reshape(1,-1), question_len.reshape(-1,)]
            out_dict = output(unique_id, model(c, training=False))
            for k, v in out_dict.items():
                if isinstance(v, tf.Tensor):
                    out_dict[k] = v.numpy()
            inicio,final=np.argmax(out_dict["long_p"][0][:, 0]), np.argmax(out_dict["long_p"][0][:, 1])

            print("Question: {}".format(question["question"]))
            print("Answer from system: {}".format(text_tokens[inicio:final]))

            pass

