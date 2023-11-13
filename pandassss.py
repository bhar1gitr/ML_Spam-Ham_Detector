import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv('mail_data.csv')

# Data Preprocessing
df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})

X = df['Message']
Y = df['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature Extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Model Training
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Model Evaluation
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print(f'Accuracy on Training Data: {accuracy_on_training_data}')

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test = accuracy_score(Y_test, prediction_on_test_data)
print(f'Accuracy on Test Data: {accuracy_on_test}')

# Input Mail Prediction
input_your_mail = ["Congratulations! You won 3000$ Walmart gift card. Go to http://bit.ly/123456 tp claim now."]

input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print('Ham')
else:
    print('Spam')

print(prediction)
