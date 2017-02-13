import tensorflow as tf
import numpy as np

tf.reset_default_graph()

tf.set_random_seed(10)
np.random.seed(10)

# Batch size
B = 4
# (Maximum) number of time steps in this batch
T = 8
RNN_DIM = 128
NUM_CLASSES = 10

# The *acutal* length of the examples
example_len = [1, 2, 3, 8]

# The classes of the examples at each step (between 1 and 9, 0 means padding)
y = np.random.randint(1, 10, [B, T])
for i, length in enumerate(example_len):
    y[i, length:] = 0   
    
# The RNN outputs
rnn_outputs = tf.convert_to_tensor(np.random.randn(B, T, RNN_DIM), dtype=tf.float32)

# Output layer weights
W = tf.get_variable(
    name="W",
    initializer=tf.random_normal_initializer(),
    shape=[RNN_DIM, NUM_CLASSES])

# Calculate logits and probs
# Reshape so we can calculate them all at once
rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, RNN_DIM])
logits_flat = tf.batch_matmul(rnn_outputs_flat, W)
probs_flat = tf.nn.softmax(logits_flat)

# Calculate the losses 
y_flat =  tf.reshape(y, [-1])
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_flat, y_flat)

# Mask the losses
mask = tf.sign(tf.to_float(y_flat))
masked_losses = mask * losses

# Bring back to [B, T] shape
masked_losses = tf.reshape(masked_losses,  tf.shape(y))

# Calculate mean loss
mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / example_len
mean_loss = tf.reduce_mean(mean_loss_by_example)

result = tf.contrib.learn.run_n(
    {
        "masked_losses": masked_losses,
        "mean_loss_by_example": mean_loss_by_example,
        "mean_loss": mean_loss
    },
    n=1,
    feed_dict=None)

print(result[0]["masked_losses"])
print(result[0]["mean_loss_by_example"])
print(result[0]["mean_loss"])