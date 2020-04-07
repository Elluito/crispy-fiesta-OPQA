import tensorflow_hub as hub

import tensorflow as tf
import  tensorflow.keras as keras
from official.nlp.bert.tokenization import FullTokenizer
# from official.nlp.bert.bert_models import *
import numpy as np

class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs,input_shape_with_batch,name="kernel"):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs
    self.shape=input_shape_with_batch
    self.name=name
  def build(self, input_shape):
    self.kernel = self.add_weight(self.name,
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])

  def call(self, input):
    return tf.matmul(input, self.kernel)