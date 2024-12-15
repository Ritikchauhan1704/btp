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


symptom_precaution = pd.read_csv("C:/Users/criti/OneDrive/Desktop/btp/src/llm/data/symptom_precaution.csv")
diseases = symptom_precaution[["Disease"]]
disease_dataset = diseases.to_dict(orient='records')
with open('C:/Users/criti/OneDrive/Desktop/btp/src/llm/symptoms.json', 'r') as file:
    symptoms_json = json.load(file)
# print(symptoms_json)
with open('C:/Users/criti/OneDrive/Desktop/btp/src/llm/knowledge_base.json', 'r') as file:
    knowledge_json = json.load(file)

# conversation="""
# Doctor: Hello, how can I help you today?
# Patient: I\'ve been feeling very tired lately and have lost some weight without trying.

# Doctor: I see. Are you experiencing any other symptoms?
# Patient: Yes, I feel thirsty all the time, and I need to urinate frequently, especially at night.

# Doctor: Hmm, anything else unusual you\'ve noticed?
# Patient: Sometimes my vision feels blurry, and I get hungry even after eating a meal.

# Doctor: Have you noticed any changes in your mood or energy levels?
# Patient: I\'ve been feeling restless and irritable recently.
# """
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
    model, label_encoder, feature_names = train_model(dataset)
    predicted_disease = predict_disease(model, label_encoder, structured_output["symptoms"], feature_names)

    # Display result
    # print(f"The most likely disease is: {predicted_disease}")
    # return predicted_disease
    top_diseases = structured_output["diseases"][:3]
    other_diseases = [disease for disease in top_diseases if disease != predicted_disease]

    # Get precautions for each disease (predicted and other diseases)
    precautions = {}
    for disease in [predicted_disease] + other_diseases:
        precautions[disease] = structured_output["precautions"].get(disease, [])

    # Prepare the result for UI
    result_for_ui = {
        "predicted_disease": predicted_disease,
        "predicted_precautions": precautions.get(predicted_disease, []),
        "other_diseases": []
    }

    # Add other likely diseases and their precautions
    for disease in other_diseases:
        result_for_ui["other_diseases"].append({
            "disease": disease,
            "precautions": precautions.get(disease, [])
        })

    # Display results in the UI-friendly format
    ui_output = f"Predicted Disease: {result_for_ui['predicted_disease']}\n"
    ui_output += f"Precautions: {', '.join(result_for_ui['predicted_precautions'])}\n\n"
    
    ui_output += "Other Likely Diseases and Their Precautions:\n"
    for i, other_disease in enumerate(result_for_ui['other_diseases'], start=1):
        ui_output += f"{i}. {other_disease['disease']} - Precautions: {', '.join(other_disease['precautions'])}\n"
    
    print(ui_output)  # Display in terminal, can be displayed in UI
    return result_for_ui


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
# structured_output = parse_medical_data(response.text)
# # structured_output = parse_medical_data(raw_output)
# print(structured_output)

# Prepare the dataset
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
# def train_model(dataset):
#     X = dataset.drop(columns=['disease'])
#     y = dataset['disease']

#     # Encode the disease labels
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#     # Train logistic regression model
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)

#     return model, label_encoder, X.columns

# # Predict the disease
# def predict_disease(model, label_encoder, symptom_input, feature_names):
#     input_vector = {feature: symptom_input.get(feature, 0) for feature in feature_names}
#     input_df = pd.DataFrame([input_vector])
#     prediction = model.predict(input_df)
#     predicted_disease = label_encoder.inverse_transform(prediction)
#     return predicted_disease[0]


# Train the model using Random Forest
def train_model(dataset):
    X = dataset.drop(columns=['disease'])
    y = dataset['disease']

    # Encode the disease labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model, label_encoder, X.columns

# Predict the disease
def predict_disease(model, label_encoder, symptom_input, feature_names):
    input_vector = {feature: symptom_input.get(feature, 0) for feature in feature_names}
    input_df = pd.DataFrame([input_vector])
    prediction = model.predict(input_df)
    predicted_disease = label_encoder.inverse_transform(prediction)
    return predicted_disease[0]

# # Use the rest of the code as it is
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
