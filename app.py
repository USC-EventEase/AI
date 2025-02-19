from llama_cpp import Llama
models = ["llama-2-7b.Q5_K_M.gguf","llama-30b.Q5_K_M.gguf","llama-2-13b.Q5_K_M.gguf"]
# Download the above models from TheBloke huggingface account
llm = Llama(
      model_path=models[0],
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
    #   n_ctx=2048, # Uncomment to increase the context window
      n_threads=10 # Uncomment if speed is required.
)
question = "What are the names of famous musicians from India?"
output = llm(
      "Question: "+question+" Answer: ", # Prompt
      max_tokens=None, # Generate up to 32 tokens
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=False # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output["choices"][0]["text"])