#-------------Input Queue Train--------------------------------------

input_queue_train = tf.RandomShuffleQueue(
	capacity = 100000,
	min_after_dequeue = 10000,
	dtypes=[tf.int32])

input_enqueue_op = input_queue_train.enqueue(data_train)
qr_input = tf.train.QueueRunner(input_queue_train, [input_enqueue_op] * 20)
tf.train.add_queue_runner(qr_input)
non_paddled_input = input_queue_train.dequeue()

padding_queue_train = tf.PaddingFIFOQueue(
    capacity=padding_queue_cap,
    dtypes=[tf.int32],
    shapes=[[None, 1]])

padding_enqueue_op = padding_queue_train.enqueue(non_paddled_input)
qr_padding_train = tf.train.QueueRunner(padding_queue_train, [padding_enqueue_op] * 10)
tf.train.add_queue_runner(qr_padding_train)
inputs_test = padding_queue_train.dequeue_many(batch_size)

#----------------------Input Queue Test-------------------------------


input_queue_test = tf.RandomShuffleQueue(
    capacity = 100000,
    min_after_dequeue = 10000,
    dtypes=[tf.int32])

input_enqueue_op_test = input_queue_test.enqueue(data_test)
qr_input_test = tf.train.QueueRunner(input_queue_test, [input_enqueue_op_test] * 20)
tf.train.add_queue_runner(qr_input)
non_paddled_input = input_queue_test.dequeue()

padding_queue_test = tf.PaddingFIFOQueue(
    capacity=padding_queue_cap,
    dtypes=[tf.int32],
    shapes=[[None, 1]])

padding_enqueue_op = padding_queue_test.enqueue(non_paddled_input)
qr_padding_test = tf.train.QueueRunner(padding_queue_test, [padding_enqueue_op] * 10)
tf.train.add_queue_runner(qr_padding_test)
inputs_train = padding_queue_test.dequeue_many(batch_size)
