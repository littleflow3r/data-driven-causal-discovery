import numpy as np
from openai import OpenAI
import random
from tqdm.autonotebook import tqdm
from .utils import PreviousEdges
# from .llm_model import *
from itertools import combinations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

import os, sys
root_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.insert(0, root_path)
import credentials

def inference(model, tokenizer, input_prompt, max_new_tokens=50):
  print (f'## Inference..')
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

def llm_pairwise(var_names_and_desc, prompts, df, temperature, include_statistics=False, include_obs=False):
    client = OpenAI(api_key=credentials.oai_key)
    # list all edges
    edges = list(combinations(df.columns, 2))
    previous_edges = PreviousEdges()
    for e in tqdm(edges):
        if random.random() < 0.5:
            head,tail = e
        else:
            tail,head = e            
        head = var_names_and_desc[head]
        tail = var_names_and_desc[tail]

        query = f'''{prompts["user"]} 
        Here is a description of the causal variables in this causal graph:
        '''
        for var in var_names_and_desc:
            causal_var = var_names_and_desc[var]
            query += f'''{causal_var.name}: {causal_var.description}\n'''
        
        # query = f'''{prompts["user"]}\n 
        query += f'''{prompts["user"]}\n 
        Choose a correct statement regarding the causal relationship between "{head.name}" and "{tail.name}".
        '''
        # query += f'''
        # Here are the causal relationships you know so far:
        # {previous_edges.get_previous_relevant_edges_string(head.name, tail.name)}
        # We are interested in the causal relationship between "{head.name}" and "{tail.name}".
        # '''

        if include_statistics:
            arr = df[[head.symbol, tail.symbol]].to_numpy()
            corr_coef = np.corrcoef(arr)[0,1]
            corr_coef = round(corr_coef, 2)
            query += f'''
            To help you, the Pearson correlation coefficient between "{head.name}" and "{tail.name}" is {corr_coef}
            '''
        if include_obs:
            obs_data = df[[head.symbol, tail.symbol]]
            query += f'''
            Here is some observational data for "{head.name}" and "{tail.name}": {obs_data.to_string(index=False)} \n
            '''

        query += f''' 
        A. "{head.name}" causes "{tail.name}". 
        B. "{tail.name}" causes "{head.name}". 
        C. There is no causal relationship between "{head.name}" and "{tail.name}".
        Provide your final answer within the tags <Answer>A/B/C</Answer>.'''
        # Let’s work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags <Answer>A/B/C</Answer>.'''
        print (f'{query=}')
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
            {
                "role": "system",
                "content": prompts["system"]
            },
            {
                "role": "user",
                "content": query
            }
            ],
            temperature=temperature,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        answer = response.choices[0].message.content
        idx = answer.find('<Answer>')
        if idx == -1:
            print("NO ANSWER FOUND")
            print("This was the answer:", answer)
            continue
        choice = answer[idx+8]

        previous_edges.add_edge_pair_wise(head.name, tail.name, choice)

    adj_matrix = previous_edges.get_adjacency_matrix([var_names_and_desc[var].name for var in df.columns])

    return adj_matrix

def filter_string(s):
    allowed_chars = ['A', 'B', 'C']
    pattern = f"[^{''.join(map(re.escape, allowed_chars))}]"
    return re.sub(pattern, '', s)

def midllm_pairwise(var_names_and_desc, prompts, df, model, tokenizer, max_new_tokens=50, include_statistics=False):
    # list all edges
    edges = list(combinations(df.columns, 2))
    previous_edges = PreviousEdges()
    for e in tqdm(edges):
        if random.random() < 0.5:
            head,tail = e
        else:
            tail,head = e            
        head = var_names_and_desc[head]
        tail = var_names_and_desc[tail]
        
        query = f'''{prompts["user"]}\n
        Choose a correct statement regarding the causal relationship between "{head.name}" and "{tail.name}".
        '''

        if include_statistics:
            arr = df[[head.symbol, tail.symbol]].to_numpy()
            corr_coef = np.corrcoef(arr)[0,1]
            corr_coef = round(corr_coef, 2)
            query += f'''
            To help you, the Pearson correlation coefficient between "{head.name}" and "{tail.name}" is {corr_coef}
            '''
        if include_obs:
            obs_data = df[[head.symbol, tail.symbol]]
            query += f'''
            Here is some observational data for "{head.name}" and "{tail.name}": {obs_data.to_string(index=False)} \n
            '''

        query += f''' 
        A. "{head.name}" causes "{tail.name}". 
        B. "{tail.name}" causes "{head.name}". 
        C. There is no causal relationship between "{head.name}" and "{tail.name}".
        Answer: '''
        print (f'{query=}')
        output_decode, all_outs = inference(model, tokenizer, query, max_new_tokens=50)
        try:
          choice = filter_string(all_outs[0])
          print (f'{choice=}')
          if choice not in ['A', 'B', 'C']:
            choice = 'C'
            print ('Forced choice C!')
        except:
          choice = 'C'
          print ('Forced choice C!')

        previous_edges.add_edge_pair_wise(head.name, tail.name, choice)

    adj_matrix = previous_edges.get_adjacency_matrix([var_names_and_desc[var].name for var in df.columns])

    return adj_matrix
