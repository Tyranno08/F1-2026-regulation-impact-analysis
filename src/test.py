import json
import os

file_path = 'models/simulation_results.json'

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found!")
else:
    with open(file_path) as f:
        data = json.load(f)
    
    print('Available keys in JSON:', data.keys())
    
    # Use .get() to avoid crashing if the key is missing
    circuits = data.get('circuits', [])
    validation = data.get('validation', [])
    summary = data.get('summary', "No summary found in file")

    print('Circuits in simulation:', len(circuits))
    print('Validation circuits:', len(validation))
    print('Summary:', summary)