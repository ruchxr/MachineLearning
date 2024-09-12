# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

columns = ['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist','class']
data = pd.read_csv(r"C:\Users\Ruchir\Downloads\magic+gamma+telescope\magic04.data", sep=',', header=None, names=columns)
data.head()

# %%
data.describe()

# %%
data.groupby('class').describe()

# %%
data['class'].unique()

# %%
data['class'] = data['class'].apply(lambda x: 0 if x == 'g' else 1)
data

# %%
for label in columns[:-1]:
    plt.hist(data[data['class'] == 0][label], color='blue', label='gamma', alpha=0.8)
    plt.hist(data[data['class'] == 1][label], color='red', label='hadron', alpha=0.8)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

# %%
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

ros = RandomOverSampler()
X,y = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

scaler = MinMaxScaler(feature_range=(0,1))

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# %%
y_pred = knn_model.predict(X_test)
y_pred

# %%
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))

# %%
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# %%
y_pred = nb_model.predict(X_test)

# %%
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))

# %%
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# %%
y_pred = log_reg.predict(X_test)
y_pred

# %%
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))

# %%
from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train, y_train)

# %%
y_pred = svm_model.predict(X_test)
y_pred

# %%
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))

# %%
"""
Neural Network (Deep Learning)
"""

# %%
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD

# %%

def train_model(X_train, y_train, num_nodes, dropout_rate, learning_rate, batch_size, epochs):
    nn_model = tf.keras.Sequential(
        [
            Input((X_train.shape[1],)),
            Dense(num_nodes, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(num_nodes, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(1, activation='sigmoid')
        ]
    )

    nn_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    history = nn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    return nn_model, history

# %%
def plot_loss(model):
    plt.plot(model.history.history['loss'], label='loss')
    plt.plot(model.history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.title('Loss vs Validation Loss')
    plt.legend()
    plt.show()

# %%
def plot_accuracy(model):
    plt.plot(model.history.history['accuracy'], label='loss')
    plt.plot(model.history.history['val_accuracy'], label='val_loss')
    plt.xlabel('Epoch')
    plt.title('Accuracy vs Validation Accuracy')
    plt.legend()
    plt.show()

# %%
import math

epochs = 50
least_val_loss = math.inf
least_loss_model = None
for num_nodes in [32,64,128]:
    for dropout_rate in [0, 0.2]:
        for lr in [0.01, 0.001]:
            for batch_size in [32, 64]:
                print(f"Num nodes: {num_nodes}, Drop out rate: {dropout_rate}, Learning Rate: {lr}, Batch Size: {batch_size}")
                model, history = train_model(X_train, y_train, num_nodes, dropout_rate, lr, batch_size, epochs)
                plot_loss(model)
                plot_accuracy(model)
                loss = model.history.history['val_loss']
                val_loss = sum(loss)/len(loss)
                
                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    least_loss_model = model

# %%
y_pred = least_loss_model.predict(X_test)

# %%
y_pred = (y_pred > 0.5).astype(int)
y_pred = np.reshape(y_pred, (y_pred.shape[0]))
y_pred

# %%
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))