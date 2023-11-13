from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import secrets

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)

# Load the model and vectorizer
df = pd.read_csv('mail_data.csv')
df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})

X = df['Message']
Y = df['Category']

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_features = feature_extraction.fit_transform(X)

model = LogisticRegression()
model.fit(X_features, Y)

class MailForm(FlaskForm):
    mail_message = TextAreaField('Enter your mail:', render_kw={"rows": 5, "cols": 50})
    submit = SubmitField('Predict')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = MailForm()
    prediction = None

    if form.validate_on_submit():
        input_mail = [form.mail_message.data]
        print("Input Mail:", input_mail)

        input_data_features = feature_extraction.transform(input_mail)
        # print("Input Data Features:", input_data_features)

        prediction = model.predict(input_data_features)
        print("Prediction:", prediction)

    return render_template('index.html', form=form, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
