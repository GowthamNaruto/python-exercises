import tensorflow as tf


# TensorFlow operates on multidimensional arrays or tensors represented as tf.Tensor objects
x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

print(x + x)
print(x.shape)
print(x.dtype)
