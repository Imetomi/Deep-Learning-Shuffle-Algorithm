import tensorflow as tf
import numpy as np



@tf.function
def calc_loss(x_train, y_train, model, loss_function):
    with tf.GradientTape() as tape:
        logits = model(x_train, training=False)
        loss_value = loss_function(y_train, logits)
    return loss_value


length = 10
def bad_batch_selector(data, idx, model, loss_function ):
    largest_loss = 0
    largest_loss_idx = idx

    if idx < len(data) - length:
        for i in range(idx, idx+length):
            x_batch_train = data[i][0]
            y_batch_train = data[i][1]
            loss = calc_loss(x_batch_train, y_batch_train, model, loss_function)
            if loss > largest_loss:
                largest_loss = loss
                largest_loss_idx = i
        return largest_loss_idx
    else:
        loss = calc_loss(data[idx][0], data[idx][1], model, loss_function)
        return idx