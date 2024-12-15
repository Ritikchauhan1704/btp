import json

with open('./llmtrain.json', 'r') as file:
    data = json.load(file)
with open('./knowledge_base.json', 'r') as file:
    knowledge_json = json.load(file)

import google.generativeai as genai

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


from difflib import SequenceMatcher

def match(predicted, expected, threshold=0.8):
    similarity = SequenceMatcher(None, predicted, expected).ratio()
    return similarity >= threshold

def llm_predict(input_text):
    diagnosis_prompt = f"""
        You are a medical assistant AI tasked with analyzing a medical conversation to identify the most likely disease. Based on the provided conversation and the knowledge JSON, complete the tasks below and respond **only** with the name of the most likely disease. Do not include any additional text, explanations, or formatting.  

        Conversation:  
        {input_text}  

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

        2. **Predict the Disease**:  
        - Using the extracted symptoms and their severity scores, identify **only one** most likely disease from the knowledge JSON.  

        ### Strict Response Format:  
        **Disease Name**  

        Respond only with the name of the most likely disease. Do not include any additional text, explanations, or formatting.
    """

    response = chat_session.send_message(diagnosis_prompt)
    print("Done")
    return response.text


from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

# Convert test data to input-output pairs
data2 = [(example["input"], example["output"]) for example in data]

for train_idx, test_idx in kf.split(data2):
    train_data = [data2[i] for i in train_idx]
    test_data = [data2[i] for i in test_idx]
    
    # Generate predictions for the test fold
    predictions = [llm_predict(input_) for input_, _ in test_data]
    
    correct = sum(
        match(pred, expected) for pred, (_, expected) in zip(predictions, test_data)
    )
    accuracies.append(correct / len(test_data))

mean_accuracy = np.mean(accuracies)
std_dev = np.std(accuracies)

print(f"Cross-Validated Accuracy: {mean_accuracy * 100:.2f}%")
print(f"Standard Deviation: {std_dev * 100:.2f}%")
