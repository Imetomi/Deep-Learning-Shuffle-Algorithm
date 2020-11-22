import tensorflow as tf
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.autonotebook import tqdm, trange
from sklearn.model_selection import train_test_split

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



    @tf.function
    def train_step(self, x_train, y_train):
        with tf.GradientTape() as tape:
            logits = self.Model(x_train, training=True)
            loss_value = self.LossFunction(y_train, logits)
        grads = tape.gradient(loss_value, self.Model.trainable_weights)
        self.Optimizer.apply_gradients(zip(grads, self.Model.trainable_weights))
        if self.TrainMetrics != None:
            self.TrainMetrics.update_state(y_train, logits)
        return loss_value



    @tf.function
    def validation_step(self, x_val, y_val):
        val_logits = self.Model(x_val, training=False)
        self.ValMetrics.update_state(y_val, val_logits)



    def train(self, epochs):
        train_data = list(self.train_dataset)

        for epoch in range(epochs):    
            steps = trange(len(train_data), bar_format="{desc}\t{percentage:3.0f}% {r_bar}")
            for i in steps:
                step = i
                x_batch_train = train_data[i][0]
                y_batch_train = train_data[i][1]
                
                if self.BatchSelector != None:
                    x_batch_train, y_batch_train = self.BatchSelector(train_data, i)


                loss_value = self.train_step(x_batch_train, y_batch_train)
                steps.set_description("Epoch " + str(epoch+1) + '/' + str(epochs) + "\tLoss: " + str(float(loss_value))[:6]
                                    + "\tAccuracy: " + str(float(self.TrainMetrics.result()))[:6])
                

                if i == len(train_data)-1:
                    for x_batch_val, y_batch_val in self.validation_dataset:
                        self.validation_step(x_batch_val, y_batch_val)
                    steps.set_description(steps.desc + "\tValidation accuracy: " + str(float(self.ValMetrics.result()))[:6])

            self.TrainMetrics.reset_states()
            self.ValMetrics.reset_states()
