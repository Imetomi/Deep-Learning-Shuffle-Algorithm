# Deep Learning Batch Selection Algorithm
The goal of our project is to create a batch selection algorithm that is able to make the learning process faster by carefully selecting corresponding instances in every batch. Our hope is to outperform random shuffling that is commonly used in deep learning problems.  

## Datasets
To measure the success of our ideas we’re going to use a variety of datasets and neural network architectures in order to get relevant benchmarks. Some of these are well-known preprocessed and toy-datasets while we also create some of our own data for a more advanced comparison. The reason for using some toy-datasets as well is to have a better mathematical understanding about how our algorithm affects the learning process.  
These are the following:  
Dataset       | Architecture  | Source 
------------- | ------------- | -------------
Fashion MNIST  | CNN | [GitHub](https://github.com/zalandoresearch/fashion-mnist)
MNIST  | CNN | Loaded from Keras
Sentiment140 | LSTM | [Kaggle](https://www.kaggle.com/kazanova/sentiment140)
Music Genre | FFW | [GitHub](https://github.com/kumargauravsingh14/music-genre-classification/blob/master/data.csv)
Wine Dataset | FFW | Loaded from SciKit-Learn
Boston Dataset | FFW | Loaded from SciKit-Learn
Iris Dataset | FFW | [Kaggle](https://www.kaggle.com/uciml/iris)
Archimedean Spirals | FFW | Custom Generator Code

A few of these are quite large and are not uploaded in this repository.  
With the exception of the NLP problem these datasets don't need advanced preprocessing and feature engineering. We used normalization and standardization where needed and made some plots just to see what are we going to tackle.  

## Metrics
We measure the success rate of our custom algorithm based on how much **time** is needed for the neural network to achieve a given accuracy. When working on different computers we also check the number of **epochs** needed for a given loss. These results will be compared to random shuffling with a predefined random state.

## Models
Using various neural network architectures we aim to test our algorithm in different conditions. Most of these are small models to get an insight about how our batch selection program performs.

## Custom Training Loop
In order to define custom batch selection algorithms we need to access the training loop. To solve this problem we created a custom training class which can be found in the `loop.py` file. The program iterates through the epochs and executes the training steps with a given optimizer, loss function, and dataset. The code also utilizes the Tensorflow framework's acceleration so training time is roughly the same compared to the original.

## Batch Selection Program
We prepared our framework so the only thing left is to create the algorithm that fulfils this part. This function is going to be given as a parameter to the training class.

## Participants
Czibor Dóra  
Kacsó Péter Gábor
Imets Tamás

## Team
Johnny Depp Learning
