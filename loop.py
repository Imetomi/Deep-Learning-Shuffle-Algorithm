import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.autonotebook import tqdm, trange
from sklearn.model_selection import train_test_split

# This class manages the training of a model with a custom training loop.
# It is needed so we can replace the batch selection part of the whole training algorithm.
class TrainingLoop:
    def __init__(self, model, X, y, loss_function, optimizer, train_metrics=None, val_metrics=None, validation_split=0, shuffle=True, batch_selection=None, batch_size=1):
        np.random.seed(42)
        tf.random.set_seed(42)
        self.Model = model 
        self.LossFunction = loss_function
        self.TrainMetrics = train_metrics
        self.Optimizer = optimizer
        self.ValMetrics = val_metrics
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, shuffle=shuffle, test_size=validation_split, random_state=42)
        self.batch_size = batch_size
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train)).shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)
        self.validation_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val)).batch(batch_size, drop_remainder=True)
        self.BatchSelector = batch_selection



    # The training step of the network for one batch.
    @tf.function
    def train_step(self, x_train, y_train):
        with tf.GradientTape() as tape:
            # Run the forward pass on this batch.
            logits = self.Model(x_train, training=True)
            # Compute loss value for this batch.
            loss_value = self.LossFunction(y_train, logits)
        
        # Retreive the gradients of the trainable weigths for the current forward pass.
        grads = tape.gradient(loss_value, self.Model.trainable_weights)
        # Run the gradient descent with the calculated gradients.
        self.Optimizer.apply_gradients(zip(grads, self.Model.trainable_weights))
        # Update training metric.
        if self.TrainMetrics != None:
            self.TrainMetrics.update_state(y_train, logits)

        return loss_value



    # The validation step of the network, this is run for every batch, at the end of every epoch.
    @tf.function
    def validation_step(self, x_val, y_val):
        # Run the forward pass on this batch.
        val_logits = self.Model(x_val, training=False)
        # Update the validation metrics.
        if self.ValMetrics != None:
            self.ValMetrics.update_state(y_val, val_logits)



    def train(self, epochs):
        train_data = list(self.train_dataset)

        # Go through every epoch.
        for epoch in range(epochs):    
            steps = trange(len(train_data), bar_format="{desc}\t{percentage:3.0f}% {r_bar}")

            # Go through the batches of the dataset.
            for i in steps:
                step = i
                # By default we don't use batch selection, just simply go through the shuffled dataset.
                x_batch_train = train_data[i][0]
                y_batch_train = train_data[i][1]
                
                # If we provide a BatchSelector, then use that to select the current batch.
                if self.BatchSelector != None:
                    x_batch_train, y_batch_train = self.BatchSelector(train_data, i)

                # Train the network on the selected batch.
                loss_value = self.train_step(x_batch_train, y_batch_train)

                if self.TrainMetrics != None:
                    steps.set_description("Epoch " + str(epoch+1) + '/' + str(epochs) + "\tLoss: " + str(float(loss_value))[:6]
                                    + "\tMetrics: " + str(float(self.TrainMetrics.result()))[:6])
                else:
                    steps.set_description("Epoch " + str(epoch+1) + '/' + str(epochs) + "\tLoss: " + str(float(loss_value))[:6])
                                
                if i == len(train_data)-1 and self.ValMetrics != None:
                    # Run validation loop, which means calling the validation step for every batch in the validation dataset.
                    for x_batch_val, y_batch_val in self.validation_dataset:
                        self.validation_step(x_batch_val, y_batch_val)
                    steps.set_description(steps.desc + "\tValidation metrics: " + str(float(self.ValMetrics.result()))[:6])
                    self.ValMetrics.reset_states()
            
            if self.TrainMetrics != None:
                self.TrainMetrics.reset_states()
