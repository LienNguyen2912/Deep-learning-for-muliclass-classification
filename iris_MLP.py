# mlp for muliclass classification
from numpy import argmax
from pandas import read_csv
from pandas.core.algorithms import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from tensorflow.python.keras.engine import input_layer

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

# main program
df = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv', header=None, names=['sepalLen', 'sepalWid', 'petalLen', 'petalWid', 'species'])

# Data exploration
df.info()
print(df.isnull().values.any())
print(df['species'].value_counts())
print(df.describe())
# Data visualization
# Show histogram
df_variables = ['sepalLen', 'sepalWid', 'petalLen', 'petalWid']
j = 1
for i in df_variables:
    pyplot.subplot(2, 2, j)
    pyplot.hist(df[i])
    pyplot.title(i)
    j +=1
pyplot.suptitle('The histogram representation of the univariate plots for each measurement')
fig = pyplot.gcf()
fig.set_size_inches(13.5, 6.5)
pyplot.show()
# Show bivariate relation between each pair of features
sns.set_palette('husl')
b = sns.pairplot(df,hue="species");
pyplot.show()

#Split and standardize data
data_splited = splitTrainDevTestSet(df, train_split= 0.8, dev_split= 0.1, test_split=0.1)

X_train, X_dev, X_test = data_splited['X_train'], data_splited['X_dev'], data_splited['X_test']
y_train, y_dev, y_test = data_splited['y_train'], data_splited['y_dev'], data_splited['y_test'] 
print('X_train.shape = %s, X_dev.shape = %s, X_test.shape = %s, y_train.shape = %s, y_dev.shape = %s, y_test.shape = %s' % (X_train.shape, X_dev.shape, np.array(X_test).shape, y_train.shape, y_dev.shape, np.array(y_test).shape ))
# Define the model
n_features = X_train.shape[1]
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(x = X_train, y = y_train, batch_size=32, epochs=150, verbose=0, validation_data=(X_dev, y_dev))

# Evaluate the model
loss_train, acc_train = model.evaluate(X_train, y_train, verbose = 0)
loss_dev, acc_dev = model.evaluate(X_dev, y_dev, verbose = 0)
print('loss_train = %.5f, acc_train = %.3f' %(loss_train, acc_train))
print('loss_dev = %.5f, acc_dev = %.3f' %(loss_dev, acc_dev))
# Plot learning curves
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()

# Make a prediction
row = [5.1,3.5,1.4,0.2]
yhat = model.predict([row])
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
# Report result on the test set
_, acc_test = model.evaluate(X_test, y_test, verbose=0)
print('acc_test_set = %.3f' %(acc_test))
