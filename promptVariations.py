import ollama
from datasets import load_dataset
import re
import json
import random

# Clean up each variant to remove numbering or bullets
def clean_variant(variant):
     # Remove leading bullets/numbers
    variant = re.sub(r"^[\d\-\.\)\s]+", "", variant)

    # Remove leading labels like "Formal:", "More Conversational:", with or without bold/italics
    variant = re.sub(r"^\*{0,2}[\w\s&]+:\*{0,2}", "", variant)

    # Remove inline or trailing parenthetical commentary
    variant = re.sub(r"\s*\([^)]*\)", "", variant)

    # Remove surrounding quotation marks (single or double)
    variant = variant.strip().strip('\'"“”‘’')

    # Remove markdown bold/italic
    variant = variant.replace("**", "").replace("*", "")

    return variant.strip()



# This function generates question variants for testing model prompt robustness
def generate_variants(question, num_variants=5):
    prompt = f"Generate {num_variants} different ways to ask the following question: {question}"

    response = ollama.chat(model="gemma3:12b", messages=[{"role": "user", "content": prompt}])

    # Extract and clean each variant
    content = response['message']['content']
    lines = content.strip().split('\n')
    question_lines = [
            clean_variant(line)
         for line in lines
            if '?' in line and not line.lower().startswith(("okay", "here", "aiming", "this", "the following", "to answer"))
        ]

    return question_lines[:num_variants]


# This function generates question variants for testing model prompt robustness
def generate_squad_variants(dataset, num_variants=5, num_questions=30):
    all_variants = []
    
    # Convert the dataset to a list of examples (this will make random.sample work)
    dataset_list = list(dataset)
    
    # Randomly sample `num_questions` from the dataset
    sampled_data = random.sample(dataset_list, num_questions)
    
    # Loop through the randomly selected questions
    for i, example in enumerate(sampled_data):
        question = example['question']
        answer = example['answers']['text'][0]  # get the first answer (if any)
        context = example['context']
        
        print(f"Generating variants for question {i + 1}/{num_questions}: {question}")  # Debugging line
        
        # Add the original question as a variant
        all_variants.append({
            'context': context,
            'original_question': question,
            'variant': question,
            'expected_answer': answer,
            'type': 'original'
        })


        # Generate question variants using your function
        variants = generate_variants(question, num_variants)

        # Store the generated variants
        for variant in variants:
            all_variants.append({
                'context': context,
                'original_question': question,
                'variant': variant,
                'expected_answer': answer,
                'type': 'variant'
            })
    
    # Save to a JSON file for further evaluation
    with open("squad_question_variants.json", "w") as f:
        json.dump(all_variants, f, indent=2)
    
    print(f"Finished generating {len(all_variants)} variants.")  # Debugging line
    
    return all_variants

    
