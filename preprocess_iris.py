import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
iris_data=pd.read_csv('iris_data.csv')
print(iris_data.to_string())

#seperate data to feature and target values
X=iris_data.drop('Species',axis=1)#features
y=iris_data['Species']#target

#encoding data with onehotencoder
encoder=OneHotEncoder()
y_encoded=encoder.fit_transform(y.values.reshape(-1,1)).toarray()

#split the data %80 for training, %20 for tessting
X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,test_size=0.2,random_state=42)

#feature scaling with standart scaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

processed_data={
    'X_train_scaled':X_train_scaled,
    'X_test_scaled':X_test_scaled,
    'y_train':y_train,
    'y_test':y_test
}


with open('processed_data.pkl','wb') as file:
    pickle.dump(processed_data,file)