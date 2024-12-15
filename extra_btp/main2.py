import pandas as pd
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

genai.configure(api_key="AIzaSyCYqCub2FoeKf9m0AzwT6dRLWUk7uz_hbA")
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)


symptom_precaution = pd.read_csv("./data/symptom_precaution.csv")
diseases = symptom_precaution[["Disease"]]
disease_dataset = diseases.to_dict(orient='records')

with open('./symptoms.json', 'r') as file:
    symptoms_json = json.load(file)
# print(symptoms_json)
with open('./knowledge_base.json', 'r') as file:
    knowledge_json = json.load(file)

conversation="""

Patient: "Doctor, I've been feeling unusually tired for the past few weeks. It's like a deep exhaustion that doesn\'t go away no matter how much rest I get. At first, I thought it was just due to my busy schedule, but it\'s been constant, and I find it harder to get through the day."

Doctor: "I see. How long have you been experiencing this fatigue? And is there anything else you've noticed with it?"

Patient: "It's been about three weeks now. Along with the tiredness, I\'ve been getting frequent headaches. They\'re not very severe, but they are constant, especially in the morning, and sometimes they last throughout the day. I\'ve also been feeling lightheaded and dizzy, especially when I stand up too quickly. It\'s really unsettling."

Doctor: "How would you rate the dizziness? Is it just a feeling of unsteadiness, or do you actually faint or lose balance?"

Patient: "It\'s more of a lightheaded feeling, but I have to be careful not to move too fast. I\'ve never actually fainted, but I do feel unstable sometimes, like I might fall. I try to sit down when it happens, and that helps a bit. I\'ve also noticed some breathlessness when I do even simple activities, like walking up stairs or carrying groceries. I get winded really easily."

Doctor: "Okay, it sounds like the fatigue, dizziness, and breathlessness are all affecting your daily life. Have you noticed any other symptoms, like chest pain, swelling, or changes in your appetite or weight?"

Patient: "Yes, actually. For the past few days, I\'ve been experiencing occasional chest pain, especially after physical exertion, but it's not constant. It\'s sharp, and it only lasts for a few seconds. I\'m also a little worried because I\'ve unintentionally lost about 5 pounds in the last couple of weeks. I haven't changed my diet or exercise routine. I\'m not sure if it's related, but I thought I should mention it."

Doctor: "I see. So, in summary, you\'re experiencing extreme fatigue, frequent headaches, dizziness, breathlessness, occasional chest pain, and unintentional weight loss. That\'s quite a range of symptoms. Have you had any other medical conditions in the past, or are you currently on any medications?"

Patient: "No major medical conditions in the past. I do have a history of mild high blood pressure, but I\'ve been managing it with medication for a few years. I\'m on lisinopril for that. I also take a multivitamin daily, but that\'s about it. I\'ve never had issues with anemia or heart problems before, so I\'m a bit concerned now."

Doctor: "Understood. It\'s important that we address these symptoms and run a few tests to get to the bottom of this. I\'ll need to do a physical exam and order some blood tests to check for things like anemia, thyroid problems, and any signs of cardiovascular issues. We\'ll also look at your heart function with an ECG and perhaps some imaging, just to be thorough. Does that sound okay to you?"

Patient: "Yes, that sounds good. I\'m just worried because it\'s affecting my ability to work and do normal daily activities. I hope we can find out what\'s going on."

Doctor: "We\'ll get to the bottom of this. It\'s always better to be cautious and check things out thoroughly. In the meantime, try to get as much rest as you can and avoid overexerting yourself. We\'ll follow up on the test results soon."

"""
def generate(conversation):
    diagnosis_prompt = f"""
    You are a medical assistant AI tasked with analyzing a medical conversation to identify potential health issues. Based on the provided conversation and the knowledge JSON, complete the tasks below and respond **only** in the specified format. Do not include any additional text or explanations.  

    Conversation:  
    {conversation}  

    Knowledge JSON:  
    {knowledge_json}  

    The knowledge JSON contains:  
    - Disease names with their associated symptoms.  
    - Severity scores for each symptom.  
    - Precautions for each disease.  

    Tasks:  
    1. **Extract Symptoms**:  
    - Identify all symptoms mentioned in the conversation.  
    - For each identified symptom, find the closest match in the knowledge JSON's symptom list and assign its corresponding severity score.  
    - Ignore any symptom that is not found in the knowledge JSON.  

    2. **Assign Severity Scores**:  
    - List the extracted symptoms along with their severity scores.  

    3. **Predict Diseases**:  
    - Using the extracted symptoms and their severity scores, identify the top 3 most likely diseases from the knowledge JSON.  

    4. **Provide Precautions**:  
    - For each predicted disease, list its corresponding precautions from the knowledge JSON.  

    ### Strict Response Format:  

    Extracted Symptoms:  
    [symptom_1: severity_score, symptom_2: severity_score, symptom_3: severity_score, ...]  

    Predicted Disease(s):  
    1. Disease Name  
    2. Disease Name  
    3. Disease Name  

    Precautions:  
    1. Disease Name: [precaution_1, precaution_2, precaution_3, ...]  
    2. Disease Name: [precaution_1, precaution_2, precaution_3, ...]  
    3. Disease Name: [precaution_1, precaution_2, precaution_3, ...]  
    """

    response = chat_session.send_message(diagnosis_prompt)
    print(response.text)
    
    structured_output=parse_medical_data(response.text)
    print(structured_output)
    dataset = prepare_dataset(knowledge_json)
    # model, label_encoder, feature_names = train_model(dataset)
    # predicted_disease = predict_disease(model, label_encoder, structured_output["symptoms"], feature_names)

    # # Display result
    # print(f"The most likely disease is: {predicted_disease}")
    # dataset = prepare_dataset(knowledge_json)
    # model, label_encoder, feature_names = train_model(dataset)
    # predicted_disease = predict_disease(model, label_encoder, structured_output["symptoms"], feature_names)

    # # Display result
    # print(f"The most likely disease is: {predicted_disease}")
    dataset = prepare_dataset(knowledge_json)
    # print(dataset)
    # Train the model
    model, label_encoder, feature_names = train_model(dataset)

    # Input symptoms extracted from LLM
    input_symptoms = structured_output["symptoms"]

    # Predict disease
    predicted_disease = predict_disease(model, label_encoder, input_symptoms, feature_names)

    # Display result
    print(f"The most likely disease is: {predicted_disease}")
    return predicted_disease

from sklearn.svm import SVC

def parse_medical_data(text):
    """Parse medical symptoms and diseases from text format."""
    symptoms = {}
    diseases = []
    precautions = {}
    
    lines = text.strip().split('\n')
    current_section = None
    
    for line in lines:
        if 'Extracted Symptoms:' in line:
            current_section = 'symptoms'
            continue
        elif 'Predicted Disease(s):' in line:
            current_section = 'diseases'
            continue
        elif 'Precautions:' in line:
            current_section = 'precautions'
            continue
            
        if current_section == 'symptoms' and '[' in line:
            # Parse symptoms and severity scores
            items = line.strip('[]').split(',')
            for item in items:
                symptom, score = item.strip().split(':')
                symptoms[symptom.strip()] = int(score)
                
        elif current_section == 'diseases' and '.' in line:
            # Parse diseases
            disease = line.split('.')[1].strip()
            diseases.append(disease)
            
        elif current_section == 'precautions' and '.' in line and '[' in line:
            # Parse precautions
            disease = line.split('.')[1].split(':')[0].strip()
            precautions_list = line.split('[')[1].strip('[]').split(',')
            precautions[disease] = [p.strip() for p in precautions_list]
    
    return {
        'symptoms': symptoms,
        'diseases': diseases,
        'precautions': precautions
    }
def prepare_dataset(knowledge_base):
    all_symptoms = set()
    for disease in knowledge_base:
        all_symptoms.update(disease['symptoms'].keys())

    # Create a dataframe
    data = []
    for disease in knowledge_base:
        row = {symptom: 0 for symptom in all_symptoms}  # Initialize all symptoms to 0
        row.update(disease['symptoms'])  # Update with the disease's symptoms
        row['disease'] = disease['disease']
        data.append(row)
    # print(data)
    return pd.DataFrame(data)

# # Train the model
def train_model(dataset):
    X = dataset.drop(columns=['disease'])
    y = dataset['disease']

    # Encode the disease labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train logistic regression model
    # model = LogisticRegression(max_iter=1000)
    # model.fit(X_train, y_train)
    model = SVC()
    model.fit(X_train, y_train)
    return model, label_encoder, X.columns

# Predict the disease
def predict_disease(model, label_encoder, symptom_input, feature_names):
    input_vector = {feature: symptom_input.get(feature, 0) for feature in feature_names}
    input_df = pd.DataFrame([input_vector])
    prediction = model.predict(input_df)
    predicted_disease = label_encoder.inverse_transform(prediction)
    return predicted_disease[0]


# # Train the model using Random Forest
# def train_model(dataset):
#     X = dataset.drop(columns=['disease'])
#     y = dataset['disease']

#     # Encode the disease labels
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#     # Train Random Forest model
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)

#     # Calculate accuracy
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Model Accuracy: {accuracy * 100:.2f}%")

#     return model, label_encoder, X.columns

# # Predict the disease
# def predict_disease(model, label_encoder, symptom_input, feature_names):
#     input_vector = {feature: symptom_input.get(feature, 0) for feature in feature_names}
#     input_df = pd.DataFrame([input_vector])
#     prediction = model.predict(input_df)
#     predicted_disease = label_encoder.inverse_transform(prediction)
#     return predicted_disease[0]

# Use the rest of the code as it is
# dataset = prepare_dataset(knowledge_json)
# model, label_encoder, feature_names = train_model(dataset)
# predicted_disease = predict_disease(model, label_encoder, structured_output["symptoms"], feature_names)

# # Display result
# print(f"The most likely disease is: {predicted_disease}")
# dataset = prepare_dataset(knowledge_json)
# # print(dataset)
# # Train the model
# model, label_encoder, feature_names = train_model(dataset)

# # Input symptoms extracted from LLM
# input_symptoms = structured_output["symptoms"]

# # Predict disease
# predicted_disease = predict_disease(model, label_encoder, input_symptoms, feature_names)

# # Display result
# print(f"The most likely disease is: {predicted_disease}")
generate(conversation)