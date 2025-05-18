import requests
import json
import pandas as pd

# Set up the base URL for the local Ollama API
url = "http://localhost:11434/api/chat"


DataFrame = pd.read_csv('data/test/abstract_algebra_test.csv',header=None)

for index, row in DataFrame.iterrows():
    Question = row.iloc[0]
    AnswerA = row.iloc[1]
    AnswerB = row.iloc[2]
    AnswerC = row.iloc[3]
    AnswerD = row.iloc[4]
    Answer = row.iloc[5] 
    message = (
        "Answer this question with only A, B, C, or D. Choose the best answer based on the following options:\n"
        f"Question: {Question}\n"
        f"A: {AnswerA}\n"
        f"B: {AnswerB}\n"
        f"C: {AnswerC}\n"
        f"D: {AnswerD}\n"
        "Please respond with just A, B, C, or D."
    )
    payload = {
    "model": "gemma3",  # Replace with the model name you're using
    "messages": [{"role": "user", "content": message}]
}
    
    print(Question)
    # Send the HTTP POST request with streaming enabled
    response = requests.post(url, json=payload, stream=True)

    # Check the response status
    if response.status_code == 200:
        print("Streaming response from Ollama:")
        for line in response.iter_lines(decode_unicode=True):
            if line:  # Ignore empty lines
                try:
                    # Parse each line as a JSON object
                    json_data = json.loads(line)
                    # Extract and print the assistant's message content
                    if "message" in json_data and "content" in json_data["message"]:
                        print(json_data["message"]["content"], end="")
                except json.JSONDecodeError:
                    print(f"\nFailed to parse line: {line}")
        print()  # Ensure the final output ends with a newline
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
