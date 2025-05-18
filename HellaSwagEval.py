from datasets import load_dataset
import requests
import json
import pandas as pd

import requests
import json

#Counts the Accuracy of the model for each activity label
CountAccDict = {}
#Counts the total number of examples for each activity label
CountTotalDict={}
def EvaluateHellaSwag(dataset):

    url = "http://localhost:11434/api/chat"

    correctanswer = 0

    #Iterates through the dataset and sends each question to the model and checks if the model gives the correct answer
    for i in range(len(dataset)):
        label = dataset['activity_label'][i]
        # Construct the message for the model
        message = (
            "Answer this question with only 0, 1, 2, or 3. Choose the best answer based on the following options:\n"
            f"Question: {dataset['ctx'][i]}\n"
            f"0: {dataset['endings'][i][0]}\n"
            f"1: {dataset['endings'][i][1]}\n"
            f"2: {dataset['endings'][i][2]}\n"
            f"3: {dataset['endings'][i][3]}\n"
            "Please respond with just 0, 1, 2, or 3."
        )
        # Prepare the payload for the request
        # The payload is a dictionary that contains the model name and the message
        payload = {
            "model": "gemma3:12b",
            "messages": [{"role": "user", "content": message}]
        }

        print(f"\n🧠 Question {i + 1}")
        print(message)
        print(f"Label (correct choice): {dataset['label'][i]}")

        response = requests.post(url, json=payload, stream=True)
        CountTotalDict[label] = CountTotalDict.get(label, 0) + 1 
        CountAccDict[label] = CountAccDict.get(label, 0)
        if response.status_code == 200:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        json_data = json.loads(line)

                        if "message" in json_data and "content" in json_data["message"]:
                            content = json_data["message"]["content"].strip()
                            print(f"Model response: {content}")
                            prediction = int(content)
                            if prediction == int(dataset['label'][i]):
                                correctanswer += 1
                                CountAccDict[label]+=1
                                print("✅ Correct")
                                
                            else:
                                print("❌ Incorrect")
                            break  # We got the answer, move to next question
                    except (json.JSONDecodeError, ValueError):
                        print(f"\nFailed to parse line or convert to int: {line}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    print(f"\n✅ Total Correct: {correctanswer} / {len(dataset)}")
    print(f"🎯 Accuracy: {correctanswer / len(dataset):.2%}")
    print(f"CountAccDict: {CountAccDict}")
    print(f"CountTotalDict: {CountTotalDict}")
    totalcount=0
    for i in CountTotalDict.values():
        totalcount+=i

    print(f"Total Count: {totalcount}")
#response = requests.post(url, json=payload, stream=True)

url = "http://localhost:11434/api/chat"
 
dataset = load_dataset("hellaswag", split="validation")  # for speed, just 1000 examples
print(type(dataset))
shuffled_dataset = dataset.shuffle(seed=42)
# Take the first 1000 examples
dataset = shuffled_dataset.select(range(10000))
dataset.to_csv("output.csv")

#valuateHellaSwag(dataset)
#with open('12b_Test3_CorrectAnswer1.json', 'w') as file:
#    json.dump(CountAccDict, file, indent=4)
#with open('12b_Test3_TotalAnswer.json', 'w') as file:
#    json.dump(CountTotalDict, file, indent=4)



#print(dataset['activity_label'][0])
""" 
 # Printing individual elements from the dataset for a specific index
print(f"Activity Label: {dataset['ind'][1]}")
print(f"Activity Label: {dataset['activity_label'][1]}")
print(f"Context A: {dataset['ctx_a'][1]}")
print(f"Context B: {dataset['ctx_b'][1]}")
print(f"Full Context: {dataset['ctx'][1]}")
print(f"Endings: {dataset['endings'][1]}")
print(f"Source ID: {dataset['source_id'][1]}")
print(f"Split: {dataset['split'][1]}")
print(f"Split Type: {dataset['split_type'][1]}")
print(f"Label (correct choice): {dataset['label'][1 ]}")"""
