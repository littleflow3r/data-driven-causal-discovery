# from huggingface_hub import login
# from credentials import hug_key
# login(token=hug_key)
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
# import utils as u

def load_models(model_name, is_use_fast=False):
  print (f'## Loading the models.. {model_name=}')
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=is_use_fast)
  tokenizer.pad_token = tokenizer.eos_token
  # tokenizer.padding_side = "right"
  # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
  model = AutoModelForCausalLM.from_pretrained(model_name)
  return model, tokenizer

# def inference_with_score(model, tokenizer, input_prompt, max_new_tokens=50):
#   print (f'## Inference..')
#   inputs= tokenizer(input_prompt, return_tensors="pt")
#   output = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True, output_scores=True)
#   transition_scores = model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
#   input_length = inputs.input_ids.shape[1]
#   generated_tokens = output.sequences[:, input_length:]
#   out_llm = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
#   out_verb = u.verbalizer(out_llm)
#   # all_score = [[x, y] for x, y in zip(generated_tokens[0], transition_scores[0]]
#   options = ['A', 'B', 'C']
#   all_outs = []
#   for tok, score in zip(generated_tokens[0], transition_scores[0]):
#     print(f"| {tok:5d} | {repr(tokenizer.decode(tok)):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2%}")
#     tokk = tokenizer.decode(tok).strip()
#     if any(v in tokk for v in options) and not tokk.startswith('<'):
#       all_outs.append([str(tok), repr(tokk), score.numpy()])
#   return all_outs

def inference(model, tokenizer, input_prompt, max_new_tokens=50):
  print (f'## Inference..')
  # inputs= tokenizer.encode(input_prompt, return_tensors="pt") --> only return input_ids
  # output = model.generate(inputs)
  # inputs= tokenizer(input_prompt, return_tensors="pt").to("cuda")
  inputs= tokenizer(input_prompt, return_tensors="pt")
  output = model.generate(
          input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)[:, inputs["input_ids"].shape[1]:]
  output_decode = tokenizer.decode(output[0], skip_special_tokens=True)
  options = ['A', 'B', 'C']
  all_outs = []
  for x in output_decode.split():
    if any(v in x for v in options):
      all_outs.append(x)
  # print (f'{output_decode=}')
  return output_decode, all_outs


if __name__ == '__main__':
  # u.set_seed(1234)
  model_name="meta-llama/Meta-Llama-3-8B-Instruct"

  model, tokenizer = load_models(model_name)
  input_prompt = '''Choose a correct statement regarding the causal relationship between "dihydrotachysterol" and "hypercalcemia":
  A. hypercalcemia causes dihydrotachysterol
  B. dihydrotachysterol causes hypercalcemia
  C. There is no causal relationship between hypercalcemia and dihydrotachysterol
  Answer: 
  '''
  # print (inference(model, tokenizer, input_prompt, max_new_tokens=50))
  output_decode, all_outs = inference(model, tokenizer, input_prompt, max_new_tokens=50)
  print (output_decode, all_outs, all_outs[0])
  # print (inference_with_score(model, tokenizer, input_prompt, max_new_tokens=50))
