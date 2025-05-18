import json
import os
from datasets import load_dataset
from promptVariations import generate_squad_variants
from EvalVariations import test_model_on_variants, evaluate_results
import matplotlib.pyplot as plt

MODEL_SIZES = ["gemma3:1b", "gemma3:4b", "gemma3:12b"]
VARIANTS_FILE = "squad_question_variants.json"

# Step 1: Load existing or generate new variants
if os.path.exists(VARIANTS_FILE):
    print("Loading existing variants from file...")
    with open(VARIANTS_FILE, "r") as f:
        squad_variants = json.load(f)
else:
    print("Generating new variants...")
    squad = load_dataset("squad")["train"]
    squad_variants = generate_squad_variants(squad, num_questions=30)

# Step 2: Run model and evaluate
all_model_results = {}
accuracy_data = {}

for model_name in MODEL_SIZES:
    print(f"\n=== Running evaluation for model: {model_name} ===")
    results = test_model_on_variants(model_name, squad_variants)
    
    # Attach model name to each result entry for tracking
    for r in results:
        r["model"] = model_name

    accuracy = evaluate_results(results)
    all_model_results[model_name] = results
    accuracy_data[model_name] = accuracy

    # Save individual model results
    result_filename = f"results_{model_name.replace(':', '_')}.json"
    with open(result_filename, "w") as f:
        json.dump(results, f, indent=2)

# Step 3: Save results
# Save all results together
with open("all_model_results.json", "w") as f:
    json.dump(all_model_results, f, indent=2)

#step 3.5: Save model accuracy to another json
with open("model_accuracy_data.json", "w") as f:
    json.dump(accuracy_data, f, indent=2)

# Step 4: Plot Accuracy Graph
models = list(accuracy_data.keys())
overall = [accuracy_data[m]["overall"] for m in models]
original = [accuracy_data[m]["original"] for m in models]
variant = [accuracy_data[m]["variant"] for m in models]

x = range(len(models))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x, original, width, label='Original')
plt.bar([i + width for i in x], variant, width, label='Variant')
plt.bar([i - width for i in x], overall, width, label='Overall')

plt.xticks(x, models)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison (Gemma3 Variants)')
plt.ylim(0, 1.0)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("gemma_accuracy_comparison.png")
plt.show()