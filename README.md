# Deep Learning Models for Multiclass Classification
Deep learning models includes :
- Multilayer perceptrons (MLP)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)

In this sample, we will develope, evalue and make predictions with MLP for Iris flowers multiclass classsification
This problem involves predicting the species of iris flower given measures of the flower.</br>
![irisFlower](https://user-images.githubusercontent.com/73010204/135714519-7280d369-fd3d-4c2e-85fc-58b452feffce.jpg)

## Dataset:
- [Iris Dataset (csv).][data1]
- [Iris Dataset Description (csv).][data2]

There are 5 columns, and the number of rows is 150.
The features/variables are:
- sepal length in cm
- sepal width in cm
- petal length in cm
-  petal width in cm
- class:  Iris Setosa, Versicolor, or Virginica, used as the target.

We will explore the data by using libraries later.
## Import Libraries and Load Dataset
We need to import some libraries: pandas (loading dataset), numpy (matrix manipulation), matplotlib and seaborn (visualization), and sklearn (building classifiers) and keras for modeling.
```sh
from numpy import argmax
from pandas import read_csv
from pandas.core.algorithms import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from matplotlib import pyplot
from tensorflow.python.keras.engine import input_layer
```
Then just load the dataset by pandas
```sh
df = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv', header=None, names=['sepalLen', 'sepalWid', 'petalLen', 'petalWid', 'species'])
```
Because this csv has no headers, adding extra _names_ parameter should be the recommended way.
## Explore data
```sh
df.info()
```
![df_info](https://user-images.githubusercontent.com/73010204/135713898-d41d60a1-740a-4998-98b8-f3d6420aa6fd.PNG)

There are 150 examples and 4 features + the target variable (species ). All of the features are floats. There are no missing values. Note that neural networks work with numbers so we need to tranform target to numerical values.
```sh
print(df['species'].value_counts())
```
![value_count](https://user-images.githubusercontent.com/73010204/135713917-14de4ff8-7f4d-4515-bfba-cd9e03ae3414.PNG)

As we can see, the target is well-distributed. Each class has 50 examples. It is not skewed class, nice.
```sh
print(df.describe())
```
![describe](https://user-images.githubusercontent.com/73010204/135713980-aa57f58a-9b20-42bc-8747-bc10876b8463.PNG)
## Visualize data
So far so good, let's check the relationship between features by data visualization.
Create histogram for each input variable.
![histogram](https://user-images.githubusercontent.com/73010204/135714423-d297526c-84c1-4056-81d2-dd2e35452da5.PNG)

So as you can see, most of the distributions are scattered, except sepal width, it’s pretty normalized. We might consider normalizing them later on.
Next, we will plot pairwise relationships in a dataset.
![pairplot](https://user-images.githubusercontent.com/73010204/135714428-f8acf51f-7efd-4881-af02-cc51fc49fdd0.PNG)

From the above visualization:
- The features of an Iris-Setosa is distinctly different from the other two species.
- There is an overlap in the pairwise relationships of the Iris-Versicolor and Iris-Virginia.

## Split data
Now, we can split the dataset into a training set, a dev set (or validation set) and a test set.
Dev set is used to evaluate the performance of each classifier and fine-tune the model parameters in order to determine the best model. The test set is mainly used for reporting purposes.
We represent 80% (120), 10% (15), and 10% (15) of the original instances, respectively, and are randomly split.

```sh
def splitTrainDevTestSet(df, train_split=0.8, dev_split=0.1, test_split=0.1):
    assert (train_split + dev_split + test_split) == 1
    assert (train_split > 0) and (dev_split > 0) and (test_split > 0)

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=12)
    indices_or_sections = [int(train_split * len(df)), int((train_split + dev_split) * len(df))]
    train_ds, dev_ds, test_ds = np.split(df_sample, indices_or_sections=indices_or_sections)

    X_train = X_dev = X_test = y_train = y_dev = y_test= []    
    encoder = LabelEncoder()
    scaler = StandardScaler()

    # preprocessing train set 
    X_train, y_train = train_ds.values[:, :-1], train_ds.values[:, -1]
    X_train = scaler.fit_transform(X_train.astype('float32'))
    y_train = encoder.fit_transform(y_train)

    # preprocessing dev set 
    X_dev, y_dev = dev_ds.values[:, :-1], dev_ds.values[:, -1]
    X_dev = scaler.transform(X_dev.astype('float32'))
    y_dev = encoder.transform(y_dev)

    # preprocessing test set 
    X_test, y_test = test_ds.values[:, :-1], test_ds.values[:, -1]
    X_test = scaler.transform(X_test.astype('float32'))
    y_test = encoder.transform(y_test)

    data_splited = dict(); 
    data_splited['X_train'], data_splited['X_dev'], data_splited['X_test'] = X_train, X_dev, X_test
    data_splited['y_train'], data_splited['y_dev'], data_splited['y_test'] = y_train, y_dev, y_test

    return data_splited;
```
```sh
#Split and standardize data
data_splited = splitTrainDevTestSet(df, train_split= 0.8, dev_split= 0.1, test_split=0.1)
X_train, X_dev, X_test = data_splited['X_train'], data_splited['X_dev'], data_splited['X_test']
y_train, y_dev, y_test = data_splited['y_train'], data_splited['y_dev'], data_splited['y_test'] 
print('X_train.shape = %s, X_dev.shape = %s, X_test.shape = %s, y_train.shape = %s, y_dev.shape = %s, y_test.shape = %s' % (X_train.shape, X_dev.shape, np.array(X_test).shape, y_train.shape, y_dev.shape, np.array(y_test).shape ))
```
> X_train.shape = (120, 4), X_dev.shape = (15, 4), X_test.shape = (15, 4), y_train.shape = (120,), y_dev.shape = (15,), y_test.shape = (15,)

Input variables have different scales. And it may affect badly to the performance of the model.
In order to scale the input variables, we can do either normalization or standardization which can be achieved using the scikit-learn library.
- Normalization is a rescaling of the data so that all values are within the new range of 0 and 1. You can normalize your dataset using the scikit-learn object _MinMaxScaler_.
- Standardizing a dataset involves rescaling the distribution of values so that the mean of observed values is 0 and the standard deviation is 1. You can standardize your dataset using the scikit-learn object _StandardScaler_.

As you see, I chose _StandardScaler_ here. 
Note that it is not correct to perform standardization before splitting the data. 
In general, you should not fit any preprocessing algorithm (PCA, StandardScaler...) on the whole dataset, but only on the training set. Then use the fitted algorithm to transform the other sets. It's because the dev set is used to get an estimate of the performance of the model on unseen data. So you should behave as if you didn't have access to the test set while training the algorithm.
In other words, fitting a scaler is on the training set( _fit_transform_), and apply the same scaler on other sets (_transform_).</br>
Moreover, when predicting you should apply the fitted scaler to your real data also.</br>
You may try _MinMaxScaler_, or _MaxAbsScaler_( scaling to the range [-1, 1])... and see if it is higher performance.</br>
## Define the model
Models can be defined either with the Sequential API or the Functional API. In this sample, we use _Sequential API_
```sh
n_features = X_train.shape[1]
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax'))
```
## Compile the model
Choose loss function. Below are three most common loss functions, you can also define another one for your own.
- binary_crossentropy: for binary classification.
- sparse_categorical_crossentropy: for multi-class classification.
- mse (mean squared error): for regression.

In this sample, we select _sparse_categorical_crossentropy_ as the loss function that we want to optimize.
Next, we select _Adam_ as the algorithm to perform the optimization procedure. You may try another such as stochastic gradient descent and then compare the performance between them.
Then, select the metrics as _accuracy_.
```sh
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
## Fit the model
```sh
history = model.fit(x = X_train, y = y_train, batch_size=32, epochs=150, verbose=0, validation_data=(X_dev, y_dev))
```
## Evaluate the model.
```sh
loss_train, acc_train = model.evaluate(X_train, y_train, verbose = 0)
loss_dev, acc_dev = model.evaluate(X_dev, y_dev, verbose = 0)

print('loss_train = %.5f, acc_train = %.3f' %(loss_train, acc_train))
print('loss_dev = %.5f, acc_dev = %.3f' %(loss_dev, acc_dev))

# plot learning curves
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()
```
> loss_train = 0.15710, acc_train = 0.950
> loss_dev = 0.19406, acc_dev = 0.933

Note: Consider running the example a few times and calculate the average outcome.</br>
We plot model learning curves to show the neural network model performance over time.
![learningCurves](https://user-images.githubusercontent.com/73010204/135714361-834dd732-40f1-4057-95c3-9ed7690d98fe.PNG)

## Make predictions.
The model accuracy seems good, let 's use it to predict data, also apply on the test set to understand more deeply how accuracy it is.
```sh
row = [5.1,3.5,1.4,0.2]
yhat = model.predict([row])
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
# Report result on the test set
_, acc_test = model.evaluate(X_test, y_test, verbose=0)
print('acc_test_set = %.3f' %(acc_test))
```
> Predicted: [[4.5509735e-04 1.5763266e-01 8.4191221e-01]] (class=2)
> acc_test_set = 0.933
## Improvement

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
[data1]: <https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv>
[data2]: <https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.names>
