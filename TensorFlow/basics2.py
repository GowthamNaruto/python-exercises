from __future__ import print_function
import tensorflow as tf

# Define tensor constants
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(5)

# Various tensor operatios.
# Note: Tensor also support python operators(+, *, -, ...)
add = tf.add(a, b)
sub = tf.subtract(a, b)
mul = tf.multiply(a, b)
div = tf.divide(a, b)

# Access tensor value
print(f"add = {add.numpy()}")
print(f"sub = {sub.numpy()}")
print(f"mul = {mul.numpy()}")
print(f"div = {div.numpy()}")

# Some more operations
mean = tf.reduce_mean([a, b, c])
sum = tf.reduce_sum([a, b, c])

# Access tensor value.
print(f"mean = {mean.numpy()}")
print(f"sum = {sum.numpy()}")

# Matrix multiplication
matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[5., 6.], [7., 8.]])

product = tf.matmul(matrix1, matrix2)

# Display Tensor
print(f"Tensor = {product}")


# Convert Tensor to Numpy
print(f"toNumpy: {product.numpy()}")
