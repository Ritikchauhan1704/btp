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
conversation = """
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


diagnosis_prompt = f"""
Analyze the following medical conversation and perform the tasks below:  
{conversation}  

Tasks:  
1. Extract all symptoms mentioned in the conversation and assign a weight to each symptom (between 0 and 1). The weight should represent the relevance of the symptom to diagnosing potential diseases.  
2. Identify and list the top 3 most likely diseases based on the conversation. Only provide the disease names, with no additional text, brackets, or slashes.  

Respond strictly in this format:  

Extracted Symptoms:  
[symptom_1: weight, symptom_2: weight, symptom_3: weight, ...]  

Predicted Disease(s):  
1. Disease Name  
2. Disease Name  
3. Disease Name  

"""
response = chat_session.send_message(diagnosis_prompt)
def parse_llm_output(output):
    symptoms_section = output.split("Extracted Symptoms:")[1].split("Predicted Disease(s):")[0].strip()
    symptoms = dict(item.strip().split(": ") for item in symptoms_section.split(", "))
    symptoms = {key: float(value) for key, value in symptoms.items()} 
    diseases_section = output.split("Predicted Disease(s):")[1].strip()
    diseases = [line.split(". ")[1] for line in diseases_section.split("\n") if line.strip()]

    return diseases, symptoms

diseases, symptoms = parse_llm_output(response.text)

print(response.text)

print("Diseases:", diseases)
print("Symptoms with weights:", symptoms)

# Define a dictionary mapping diseases to their associated symptoms
# disease_symptom_map = {
#     'Cardiovascular Disease': ['Occasional chest pain', 'Breathlessness', 'Dizziness'],
#     'Anemia': ['Extreme tiredness', 'Dizziness', 'Unintentional weight loss'],
#     'Anxiety/Depression': ['Frequent headaches', 'Extreme tiredness', 'Breathlessness']
# }

# def predict_disease(diseases, symptoms_with_weights, disease_symptom_map):
#     disease_scores = {disease: 0 for disease in diseases}  # Initialize scores for all diseases

#     # Calculate the score for each disease
#     for disease, symptoms in disease_symptom_map.items():
#         for symptom in symptoms:
#             if symptom in symptoms_with_weights:
#                 disease_scores[disease] += symptoms_with_weights[symptom]

#     # Find the disease with the highest score
#     predicted_disease = max(disease_scores, key=disease_scores.get)

#     return predicted_disease, disease_scores

# # Predict the disease
# predicted_disease, disease_scores = predict_disease(diseases, symptoms_with_weights, disease_symptom_map)

# # Output results
# print("Disease Scores:", disease_scores)
# print("Predicted Disease:", predicted_disease)