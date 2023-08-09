import tensorflow as tf

# Create a Tensor
hello = tf.constant("hello world")
print(hello)

# ztf.Tensor(hello world, shape=(), dtype=string)
# To access a Tensor value, call numpy().
print(hello.numpy())
