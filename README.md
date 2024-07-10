1. **Import Libraries**:
   - The code imports necessary modules from the `transformers` library and the `torch` library.

2. **Check for GPU Availability**:
   - It checks if a GPU (CUDA) is available and sets the `torch_device` variable accordingly to use either the GPU or CPU.

   ```python
   torch_device = "cuda" if torch.cuda.is_available() else "cpu"
   ```

3. **Load Tokenizer and Model**:
   - The code loads a pre-trained GPT-2 tokenizer.
   ```python
   tokenizer = AutoTokenizer.from_pretrained("gpt2")
   model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)
   ```

4. **Get User Input**:
   - The code prompts the user to enter a text prompt, the maximum number of words to generate, and the number of different sequences to generate.

   ```python
   user_prompt = input("Enter text to generate prompt:")
   max_words = int(input("Enter max words:"))
   max_sequences = int(input("Enter max sequences:"))
   ```

5. **Tokenize User Input**:
   - It converts the user input into a format that the model can process.

   ```python
   model_inputs = tokenizer(user_prompt, return_tensors='pt').to(torch_device)
   ```

6. **Set Random Seed**:
   - It sets a random seed for reproducibility, ensuring the results are consistent across runs.

   ```python
   from transformers import set_seed
   set_seed(42)
   ```

7. **Generate Text**:
   - The model generates new text based on the user input and the specified parameters, such as the number of words, sampling settings, and temperature.

   ```python
   sample_outputs = model.generate(
       **model_inputs,
       max_new_tokens=max_words,
       do_sample=True,
       top_k=50,
       top_p=0.95,
       no_repeat_ngram_size=2,
       temperature=.4,
       num_return_sequences=max_sequences,
   )
   ```

8. **Decode and Process Outputs**:
   - It decodes the generated text into a readable format and selects the longest sequence. It also ensures the text ends at the last full stop to complete the sentence.

   ```python
   samples = [
       tokenizer.decode(sample_output, skip_special_tokens=True)
       for sample_output in sample_outputs
   ]

   result = max(samples, key=len)

   last_full_stop_index = result.rfind('.')
   if last_full_stop_index != -1:
       result = result[:last_full_stop_index + 1]
   ```

9. **Print the Result**:
   - Finally, it prints the generated text.

   ```python
   print(result)
   ```
