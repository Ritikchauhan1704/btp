import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
import google.generativeai as genai
from sklearn.model_selection import train_test_split

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
def dataPreprocessing():
    df = pd.read_csv('./data/dataset.csv')
    df = shuffle(df,random_state=42)
    # Removing Hyphen from strings
    for col in df.columns:
        df[col] = df[col].str.replace('_',' ')

    cols = df.columns
    data = df[cols].values.flatten()
    s = pd.Series(data)
    s = s.str.strip()
    s = s.values.reshape(df.shape)
    df = pd.DataFrame(s, columns=df.columns)
    df = df.fillna(0)
    df1 = pd.read_csv('./data/Symptom-severity.csv')
    df1['Symptom'] = df1['Symptom'].str.replace('_',' ')
    vals = df.values
    symptoms = df1['Symptom'].unique()
    # Encode symptoms in the data with the symptom rank
    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
        
    d = pd.DataFrame(vals, columns=cols)

    d = d.replace('dischromic  patches', 0)
    d = d.replace('spotting  urination',0)
    df = d.replace('foul smell of urine',0)
    data = df.iloc[:,1:].values
    labels = df['Disease'].values
    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8,random_state=42)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, x_test, y_train, y_test,df1
x_train, x_test, y_train, y_test,df1=dataPreprocessing()

svc_model = SVC()
svc_model.fit(x_train, y_train)

discrp = pd.read_csv("./data/symptom_Description.csv")
preaca = pd.read_csv("./data/symptom_precaution.csv")


def predict(x,S1=' ',S2=' ',S3=' ',S4=' ',S5=' ',S6=' ',S7=' ',S8=' ',S9=' ',S10=' ',S11=' ',S12=' ',S13=' ',S14=' ',S15=' ',S16=' ',S17=' '):
    psymptoms = [S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17]
    #print(psymptoms)
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j]==a[k]:
                psymptoms[j]=b[k]
            if psymptoms[j]==' ':
                psymptoms[j]=0
    psy = [psymptoms]
    pred2 = x.predict(psy)
    disp= discrp[discrp['Disease']==pred2[0]]
    disp = disp.values[0][1]
    recomnd = preaca[preaca['Disease']==pred2[0]]
    c=np.where(preaca['Disease']==pred2[0])[0][0]
    precuation_list=[]
    for i in range(1,len(preaca.iloc[c])):
        precuation_list.append(preaca.iloc[c,i])
    print("The Disease Name: ",pred2[0])
    print("The Disease Discription: ",disp)
    print("Recommended Things to do at home: ")
    for i in precuation_list:
        print(i)
    return pred2 ,disp,precuation_list
import json
with open('./knowledge_base.json', 'r') as file:
    knowledge_json = json.load(file)

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
    return structured_output



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
            items = line.strip('[]').split(',')
            for item in items:
                symptom, score = item.strip().split(':')
                symptoms[symptom.strip()] = int(score)
                
        elif current_section == 'diseases' and '.' in line:
            disease = line.split('.')[1].strip()
            diseases.append(disease)
            
        elif current_section == 'precautions' and '.' in line and '[' in line:
            disease = line.split('.')[1].split(':')[0].strip()
            precautions_list = line.split('[')[1].strip('[]').split(',')
            precautions[disease] = [p.strip() for p in precautions_list]
    
    return {
        'symptoms': symptoms,
        'diseases': diseases,
        'precautions': precautions
    }

conversation="""
Patient: "Doctor, I've been feeling unusually tired for the past few weeks. It's like a deep exhaustion that doesn\'t go away no matter how much rest I get. At first, I thought it was just due to my busy schedule, but it\'s been constant, and I find it harder to get through the day."Doctor: "I see. How long have you been experiencing this fatigue? And is there anything else you've noticed with it?"Patient: "It's been about three weeks now. Along with the tiredness, I\'ve been getting frequent headaches. They\'re not very severe, but they are constant, especially in the morning, and sometimes they last throughout the day. I\'ve also been feeling lightheaded and dizzy, especially when I stand up too quickly. It\'s really unsettling."Doctor: "How would you rate the dizziness? Is it just a feeling of unsteadiness, or do you actually faint or lose balance?"Patient: "It\'s more of a lightheaded feeling, but I have to be careful not to move too fast. I\'ve never actually fainted, but I do feel unstable sometimes, like I might fall. I try to sit down when it happens, and that helps a bit. I\'ve also noticed some breathlessness when I do even simple activities, like walking up stairs or carrying groceries. I get winded really easily."Doctor: "Okay, it sounds like the fatigue, dizziness, and breathlessness are all affecting your daily life. Have you noticed any other symptoms, like chest pain, swelling, or changes in your appetite or weight?"Patient: "Yes, actually. For the past few days, I\'ve been experiencing occasional chest pain, especially after physical exertion, but it's not constant. It\'s sharp, and it only lasts for a few seconds. I\'m also a little worried because I\'ve unintentionally lost about 5 pounds in the last couple of weeks. I haven't changed my diet or exercise routine. I\'m not sure if it's related, but I thought I should mention it."Doctor: "I see. So, in summary, you\'re experiencing extreme fatigue, frequent headaches, dizziness, breathlessness, occasional chest pain, and unintentional weight loss. That\'s quite a range of symptoms. Have you had any other medical conditions in the past, or are you currently on any medications?"Patient: "No major medical conditions in the past. I do have a history of mild high blood pressure, but I\'ve been managing it with medication for a few years. I\'m on lisinopril for that. I also take a multivitamin daily, but that\'s about it. I\'ve never had issues with anemia or heart problems before, so I\'m a bit concerned now."Doctor: "Understood. It\'s important that we address these symptoms and run a few tests to get to the bottom of this. I\'ll need to do a physical exam and order some blood tests to check for things like anemia, thyroid problems, and any signs of cardiovascular issues. We\'ll also look at your heart function with an ECG and perhaps some imaging, just to be thorough. Does that sound okay to you?"Patient: "Yes, that sounds good. I\'m just worried because it\'s affecting my ability to work and do normal daily activities. I hope we can find out what\'s going on."Doctor: "We\'ll get to the bottom of this. It\'s always better to be cautious and check things out thoroughly. In the meantime, try to get as much rest as you can and avoid overexerting yourself. We\'ll follow up on the test results soon."
"""

res=generate(conversation)

symptoms = [i for i in res['symptoms']]
print(symptoms)
cleaned_symptoms = [symptom.replace('_', ' ').replace('-', ' ') for symptom in symptoms]
print(cleaned_symptoms)
result = predict(svc_model, *cleaned_symptoms)

# Display results
print(f"Disease Predicted: {result[0][0]}")
print(f"Description: {result[1]}")
print("Recommendations:")
for rec in result[2]:
    print(f"- {rec}")