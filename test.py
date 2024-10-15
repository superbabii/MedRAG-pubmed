import torch
from transformers import pipeline

# Ensure the appropriate device is selected (GPU if available)
device = 0 if torch.cuda.is_available() else -1

# Define the model ID
model_id = "meta-llama/Llama-3.2-1B"

try:
    # Initialize the text generation pipeline
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",  # Automatically map model to available devices
    )
    print(f"Model {model_id} loaded successfully.")
    
    # Generate text with the pipeline
    result = pipe(
        "The key to life is", 
        max_length=50,  # Specify max length for generated text
        num_return_sequences=1,  # Number of sequences to generate
        do_sample=True,  # Enable sampling to make output more diverse
        top_k=50,  # Only consider the top 50 token options for sampling
        top_p=0.95,  # Nucleus sampling (truncates to top 95% probability mass)
        truncation=True,  # Explicitly enable truncation
    )
    
    print(result[0]['generated_text'])

except Exception as e:
    print(f"Error loading model or generating text: {e}")
