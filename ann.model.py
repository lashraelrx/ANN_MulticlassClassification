import preprocess_iris
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import pickle

#taking variables
with open('processed_data.pkl','rb') as file:
    processed_data=pickle.load(file)

X_train_scaled = processed_data['X_train_scaled']
X_test_scaled = processed_data['X_test_scaled']
y_train = processed_data['y_train']
y_test = processed_data['y_test']
#########################################################
#input layer, first hidden layer with relu function
model=Sequential()
model.add(Dense(10,activation='relu',
                input_shape=(X_train_scaled.shape[1],)))

#output layer with softmax function
model.add(Dense(y_train.shape[1],activation='softmax'))

#compile model
model.compile(optimizer='adam', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history=model.fit(X_train_scaled,y_train, validation_data=(X_test_scaled,y_test),epochs=100, batch_size=10)
loss, accuracy=model.evaluate(X_test_scaled,y_test)
print('Accuracy:', accuracy)
print('\nLoss:',loss)




