import tensorflow as tf

import torch
if __name__ == '__main__':

  # Create some tensors
  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  c = tf.matmul(a, b)

  print(c.device)

  # Place tensors on the CPU
  with tf.device('/CPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

  c = tf.matmul(a, b)
  print(c.device)


  print(torch.cuda.current_device())


  print(torch.cuda.device(0))


  print(torch.cuda.device_count())


  print(torch.cuda.get_device_name(0))

  print(torch.cuda.is_available())
