import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
symptoms_with_weights = {
    'Extreme Fatigue': 0.9,
    'Frequent Headaches': 0.7,
    'Dizziness': 0.8,
    'Lightheadedness': 0.8,
    'Unsteadiness': 0.7,
    'Breathlessness': 0.8,
    'Occasional Chest Pain': 0.9,
    'Unintentional Weight Loss': 0.9,
    'Winded Easily': 0.7
}
data = [
    {'Extreme Fatigue': 0.9, 'Frequent Headaches': 0.7, 'Dizziness': 0.8, 'Lightheadedness': 0.8, 'Unsteadiness': 0.7, 'Breathlessness': 0.8, 'Occasional Chest Pain': 0.9, 'Unintentional Weight Loss': 0.9, 'Winded Easily': 0.7, 'Disease': 'Anemia'},
    {'Extreme Fatigue': 0.8, 'Frequent Headaches': 0.7, 'Dizziness': 0.6, 'Lightheadedness': 0.7, 'Unsteadiness': 0.7, 'Breathlessness': 0.9, 'Occasional Chest Pain': 0.9, 'Unintentional Weight Loss': 0.6, 'Winded Easily': 0.8, 'Disease': 'Cardiovascular Disease'},
    {'Extreme Fatigue': 0.8, 'Frequent Headaches': 0.6, 'Dizziness': 0.7, 'Lightheadedness': 0.8, 'Unsteadiness': 0.7, 'Breathlessness': 0.7, 'Occasional Chest Pain': 0.4, 'Unintentional Weight Loss': 0.6, 'Winded Easily': 0.6, 'Disease': 'Hypothyroidism'}
]

df = pd.DataFrame(data)

X = df.drop(columns='Disease') 
y = df['Disease']

le = LabelEncoder()
y_encoded = le.fit_transform(y) 

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

model_lr = LogisticRegression(random_state=42)

model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression Accuracy: {accuracy_lr * 100:.2f}%')

new_symptoms = {
    'Extreme Fatigue': 0.9, 
    'Frequent Headaches': 0.7, 
    'Dizziness': 0.8, 
    'Lightheadedness': 0.8, 
    'Unsteadiness': 0.7, 
    'Breathlessness': 0.8, 
    'Occasional Chest Pain': 0.9, 
    'Unintentional Weight Loss': 0.9, 
    'Winded Easily': 0.7
}
new_symptoms_df = pd.DataFrame([new_symptoms])

predicted_disease_lr = model_lr.predict(new_symptoms_df)
predicted_disease_name_lr = le.inverse_transform(predicted_disease_lr)

print(f'The predicted disease (Logistic Regression) is: {predicted_disease_name_lr[0]}')
