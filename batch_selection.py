import tensorflow as tf
import numpy as np


# Calculate the loss of one batch.
@tf.function
def calc_loss(x_train, y_train, model, loss_function):
    with tf.GradientTape() as tape:
        # Run the forward pass.
        logits = model(x_train, training=False)
        # Compute loss value.
        loss_value = loss_function(y_train, logits)
    return loss_value


length = 10
def windowed_batch_selector(data, idx, model, loss_function ):
    largest_loss = 0
    largest_loss_idx = idx

    if idx < len(data) - length:
        # Look at next length batches and select the one with the biggest loss.
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


losses = []
def sorting_batch_selector(data, idx, model, loss_function):
    global losses
    # Index is 0, then we are the beginning of an epoch.
    if idx == 0:
        # Calculate losses.
        for i in range(len(data)):
            x_batch_train = data[i][0]
            y_batch_train = data[i][1]
            losses.append([i, float(calc_loss(x_batch_train, y_batch_train, model, loss_function))])
        # Sort batches by loss.
        losses = sorted(losses, key=lambda x:x[1], reverse=True)


    return_idx = losses[idx][0]
    if idx == len(data)-1:
        losses.clear()
    
    return return_idx