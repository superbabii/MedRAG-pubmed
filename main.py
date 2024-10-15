import json
import random
from src.medrag import MedRAG

# Load benchmark data
with open("benchmark.json", "r") as file:
    benchmark_data = json.load(file)

# Select 5 random samples from the benchmark
random_questions = random.sample(list(benchmark_data.items()), 1)

# Initialize the MedRAG model
medrag = MedRAG(llm_name="meta-llama/Llama-3.2-1B", rag=True, retriever_name="MedCPT", corpus_name="PubMed")

# Store the results of comparisons
results = []
correct_count = 0

# Loop through the selected questions and generate answers
for question_id, data in random_questions:
    question = data["question"]
    options = data["options"]
    correct_answer = data["answer"]

    # Get model's prediction
    answer, snippets, scores = medrag.answer(question=question, options=options, k=5)
  
    # Parse the generated answer and compare with correct answer
    try:
        generated_answer_dict = json.loads(answer)
        generated_choice = generated_answer_dict.get('answer_choice', None)
    except (json.JSONDecodeError, KeyError):
        generated_choice = None
        
    # Check if generated_choice is valid and compare with correct answer
    if generated_choice and len(generated_choice) > 0:
        is_correct = correct_answer == generated_choice[0]
    else:
        is_correct = False  # If no valid choice, consider it incorrect

    if is_correct:
        correct_count += 1

    result = {
        'question': question,
        'correct_answer': correct_answer,
        'generated_answer': generated_choice,
        'is_correct': is_correct,
        'snippets': snippets,
        'scores': scores
    }
    results.append(result)

# Print the results of the comparison
for result in results:
    print(f"Score: {result['scores']}")
    print(f"Correct Answer: {result['correct_answer']}")
    print(f"Generated Answer: {result['generated_answer']}")
    print(f"Is Correct: {result['is_correct']}")
    print('-' * 50)

# Calculate accuracy
accuracy = correct_count / len(results) * 100
print(f"Accuracy: {accuracy:.2f}%")
