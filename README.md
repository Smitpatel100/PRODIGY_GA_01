from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)


user_prompt = input("Enter text to generate prompt:")
max_words = int(input("Enter max words:"))
max_sequences = int(input("Enter max sequences:"))
    
model_inputs = tokenizer(user_prompt, return_tensors='pt').to(torch_device)

from transformers import set_seed

set_seed(42)

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

print("Generated Expansions Based On Context:\n" + 100 * '-')

samples = [
    tokenizer.decode(sample_output, skip_special_tokens=True)
    for sample_output in sample_outputs
]

result = max(samples,key=len)

last_full_stop_index = result.rfind('.')

if last_full_stop_index != -1:
    result = result[:last_full_stop_index + 1]

print(result)


PS C:\Users\smitp\OneDrive\Desktop - Copy\python\project> python -u "c:\Users\smitp\OneDrive\Desktop - Copy\python\project\main.py"

Enter text to generate prompt:starting a new startup

Enter max words:400

Enter max sequences:3

2024-07-06 15:14:20.245334: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-06 15:14:22.076849: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
----------------------------------------------------------------------------------------------------
starting a new startup in the world of startups, and now the company is looking to launch its own startup.

"We're very excited about the opportunity to bring the best of the Internet to our customers," said Dan Hirsch, CEO of Harker, which is based in San Francisco. "We have a lot of great ideas, but we need to get them in place and get this product out to the customer."
 (Hirsch said the startup will be called HARKER, after the name of its founder, Dan Harr.)
, a startup that aims to make the web a better place for people to connect with each other, is a part of a larger effort to improve the Web. The company has a $50 million investment from venture capital firm Andreessen Horowitz and a handful of other investors. (The company's website, HOKER.com, also features a video about it.) The new venture is called "Harkers," and Harks is the first of many startups to be founded in Silicon Valley. Hasker's goal is to build a web that is more accessible to people, more affordable, less expensive, easier to use, faster, cheaper to learn, better for businesses and more secure. It's a strategy that Harkin said will help HAWKERS make its first foray into the marketplace. In the coming months, the Hokey team will build on the success of their first venture, "The Web," which was launched in 2013.
