import ollama
import json
from tqdm import tqdm
from rapidfuzz import fuzz


def is_answer_correct(expected, predicted, threshold=85):
    return fuzz.partial_ratio(expected.lower(), predicted.lower()) >= threshold

# Function to test the model on each question variant
def test_model_on_variants(model_name,variants_data):
    results = []

    for entry in tqdm(variants_data, desc="Evaluating Variants"):
        context = entry['context']
        variant = entry['variant']
        expected_answer = entry['expected_answer']
        entry_type = entry.get('type', 'variant')  # default to 'variant' if missing

        prompt = f"Context: {context}\nQuestion: {variant}\nAnswer:"
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        model_answer = response['message']['content'].strip()

        is_correct = is_answer_correct(expected_answer, model_answer)

        results.append({
            'variant': variant,
            'expected_answer': expected_answer,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'type': entry_type  # track type for analysis
        })

    return results


# Function to evaluate the model's performance
def evaluate_results(results):
    total = len(results)
    correct = sum(r['is_correct'] for r in results)
    accuracy = correct / total if total else 0

    original = [r for r in results if r['type'] == 'original']
    variants = [r for r in results if r['type'] == 'variant']

    original_accuracy = sum(r['is_correct'] for r in original) / len(original) if original else 0
    variant_accuracy = sum(r['is_correct'] for r in variants) / len(variants) if variants else 0

    print(f"\nOverall Accuracy: {accuracy:.2f} ({correct}/{total})")
    print(f"Original Questions Accuracy: {original_accuracy:.2f} ({sum(r['is_correct'] for r in original)}/{len(original)})")
    print(f"Variants Accuracy: {variant_accuracy:.2f} ({sum(r['is_correct'] for r in variants)}/{len(variants)})")

    return {
        "overall": accuracy,
        "original": original_accuracy,
        "variant": variant_accuracy
    }

