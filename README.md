# Deep Learning Batch Selection Algorithm
The goal of our project is to create a batch selection algorithm that is able to make the learning process faster by carefully selecting corresponding instances in every batch. Our hope is to outperform random shuffling that is commonly used in deep learning problems.  

## Datasets
To measure the success of our ideas we’re going to use a variety of datasets and neural network architectures in order to get relevant benchmarks. Some of these are well-known preprocessed and toy-datasets while we also create some of our own data for a more advanced comparision.
These are the following:
Dataset       | Architecture  | Source
------------- | ------------- | -------------
Fahsion MNIST  | CNN | Loaded from Keras
MNIST  | CNN | Loaded from Keras
Sentiment140 | LSTM | [Kaggle](https://www.kaggle.com/kazanova/sentiment140)
Music Genre | FFW | [GitHub](https://github.com/kumargauravsingh14/music-genre-classification/blob/master/data.csv)
Wine Dataset | FFW | Loaded from SciKit-Learn
Boston Dataset | FFW | Loaded from SciKit-Learn
Iris Dataset | FFW | Loaded from SciKit-Learn
Archimedean Spirals | FFW | Custom Generator Code

## Metrics
We measure the success rate of our custom algorithm based on how much **time** is needed for the neural network to achieve a given accuracy. When working on different computers we also check the number of **epochs** needed for a given loss. These results will be compared to random shuffling with a predefined random state. 
