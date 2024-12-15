import pandas as pd
import json

# Load datasets
symptom_precaution = pd.read_csv("./data/symptom_precaution.csv")
symptom_severity = pd.read_csv("./data/Symptom-severity.csv")
symptom_description = pd.read_csv("./data/symptom_Description.csv")
dataset = pd.read_csv("./data/dataset.csv")

symptom_map = symptom_severity.set_index("Symptom").to_dict()["weight"]
merged = symptom_precaution.merge(symptom_description, on="Disease", how="left")
# print(merged)
with open("symptoms.json", "w") as f:
    json.dump(symptom_map, f, indent=4)

merged = merged.fillna({
    "Description": "No description available",
    "Precaution_1": "No precaution available",
    "Precaution_2": "No precaution available",
    "Precaution_3": "No precaution available",
    "Precaution_4": "No precaution available"
})
merged_dict = merged.to_dict(orient="records")

with open("merged.json", "w") as f:
    json.dump(merged_dict, f, indent=4)


knowledge_base = []
for _, row in merged.iterrows():
    disease_symptoms = dataset[
        dataset["Disease"].str.strip().str.lower() == row["Disease"].strip().lower()
    ]
    disease_symptoms = disease_symptoms.iloc[:, 1:].values.flatten() 
    disease_symptoms = [symptom for symptom in disease_symptoms if pd.notna(symptom)]
    # print(disease_symptoms)
    relevant_symptoms = {
        symptom.strip(): symptom_map.get(symptom.strip().lower(), 0)
        for symptom in disease_symptoms
    }
    
    entry = {
        "disease": row["Disease"].strip(),
        "description": row["Description"],
        "symptoms": relevant_symptoms,
        "precautions": [
            row["Precaution_1"],
            row["Precaution_2"],
            row["Precaution_3"],
            row["Precaution_4"]
        ],
    }
    knowledge_base.append(entry)

with open("knowledge_base.json", "w") as f:
    json.dump(knowledge_base, f, indent=4)

print("Knowledge base saved as 'knowledge_base.json'")
