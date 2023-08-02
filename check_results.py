import pickle

with open('processed_data.pkl','rb') as file:
    processed_data=pickle.load(file)

X_train_scaled = processed_data['X_train_scaled']
X_test_scaled = processed_data['X_test_scaled']
y_train = processed_data['y_train']
y_test = processed_data['y_test']

print(processed_data)
