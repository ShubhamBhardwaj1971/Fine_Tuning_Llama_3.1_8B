from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from flask_cors import CORS
from huggingface_hub import login
import requests
import os
import time
import json
import re

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

# Environment variables for Hugging Face Token and Dr Droid Token
hf_token = os.getenv('HF_TOKEN')
dr_droid_token = os.getenv('DR_DROID_TOKEN')

headers = {
    'Authorization': f'Bearer {dr_droid_token}',  
}

def execute_workflows(workflow_names):
    for workflow_name in workflow_names:
        json_data = {
            'workflow_name': workflow_name,
        }
        response = requests.post('http://localhost/executor/workflows/api/execute', headers=headers, json=json_data)
        print(f'Response for {workflow_name}:', response.json()) 
        
workflow_list = ['Fine Tuning Data - Integrity Checks', 'App Data Validation', 'Batch Inference Performance Debugging'] 
         
# Log in to the Hugging Face Hub with your token
login(token=hf_token)

model_id = "ShubhamBhardwaj994/medical_research_llama_3.1-8b"

# Configure for 4-bit quantization
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# Load the model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create a text generation pipeline
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.bfloat16}
)

# Load the JSONL file and return the list of entries
def load_jsonl_file(file_path):
    entries = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            entries.append(entry)
    return entries

@app.route('/finetuned_llm', methods=['POST'])
def finetuned_llm():
    data = request.get_json()
    if not data or 'content' not in data:
        return jsonify({"error": "Invalid input"}), 400

    # Extract the structured input, providing default if input is None
    content = data['content']
    instruction = content.get('instruction', "")
    input_content = content.get('input', "")  # Use an empty string if input is None

    # Prepare the messages structure
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_content or ""}
    ]

    # Load JSONL entries
    jsonl_entries = load_jsonl_file('fine_tuning_dataset.jsonl')  # Replace with your actual file path

    # Find matching entry in JSONL file
    expected_output = None
    for entry in jsonl_entries:
        if entry['instruction'] == instruction and entry['input'] == input_content:
            expected_output = entry['output']
            break

    # Measure the time taken for generating the response
    start_time = time.time()

    try:
        # Generate the response
        outputs = text_pipeline(
            messages,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Response generation time:", elapsed_time)

        # Ensure outputs are as expected
        if outputs and isinstance(outputs, list) and "generated_text" in outputs[0]:
            generated_text = outputs[0]["generated_text"]
            print("Generated Text (Raw):", generated_text)

            # Extract the first 'assistant' role message
            assistant_response = next(
                (message["content"] for message in generated_text if message["role"] == "assistant"), 
                "No answer found."
            )
            print("Extracted Response:", assistant_response)
            
            # Extract content before the stop word directly
            stop_word = "<|im_end|>"  # Define the stop word
            stop_word_index = assistant_response.lower().find(stop_word.lower())
            if stop_word_index != -1:
                assistant_response = assistant_response[:stop_word_index]

            # Extract content after the colon
            colon_index = assistant_response.find(':')
            if colon_index != -1:
                assistant_response = assistant_response[colon_index + 1:]

            # Clean the assistant response
            assistant_response = assistant_response.replace("\n", " ")  # Replace newline characters with spaces
            assistant_response = re.sub(r"\s+", " ", assistant_response)  # Normalize whitespace
            assistant_response = assistant_response.strip()  # Trim leading and trailing whitespace

            print("Final Cleaned Response:", assistant_response)

            # Compare with expected output
            is_match = assistant_response == expected_output
            print("Expected Output:", expected_output)
            print("Match:", is_match)
            
            if not is_match:
             execute_workflows(workflow_list) 

        else:
            assistant_response = "Error: Unable to generate text."
            is_match = False

    except Exception as e:
        print(f"Error during text generation: {e}")
        assistant_response = "Error: Exception during processing."
        is_match = False

    return jsonify({
        "answer": assistant_response,
        "expected": expected_output,
        "match": is_match
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)