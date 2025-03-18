import numpy as np
import pickle
# This code is for v1 of the openai package: pypi.org/project/openai
from openai import OpenAI
import random
from tqdm.autonotebook import tqdm
from .utils import PreviousEdges
from .bfs_prompts import *

import os, sys
root_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.insert(0, root_path)
import credentials

client = OpenAI(api_key=credentials.oai_key)
model = "gpt-4-0125-preview"
# model = "gpt-4o-mini"
temperature = 0.7

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

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

def llm_bfs(var_names_and_desc, dataset, df, temperature, include_statistics=False, include_obs=False): 

    corr = df.corr()
    # print ('corr:', corr)
    # sys.exit()
    # if include_statistics:
    #     corr = df.corr()
        # arr = df[[head.symbol, tail.symbol]].to_numpy()
        # print ('df:', df.to_numpy())
        # print ('corr:', corr)
        #corr = df
    # sys.exit()

    if dataset == "asia":
        message_history = asia_prompt
    elif dataset == "child":
        message_history = child_prompt
    elif dataset == "sachs":
        message_history = sachs_prompt
    elif dataset == "insurance":
        message_history = insurance_prompt
    elif dataset == "cancer":
        message_history = cancer_prompt
    elif dataset == "earthquake":
        message_history = earthquake_prompt
    elif dataset == "alarm":
        message_history = alarm_prompt
    elif dataset == "sprinkler":
        message_history = sprinkler_prompt
    elif dataset == "survey":
        message_history = survey_prompt
    # message_history = f'{dataset}_prompt

    nodes = [var for var in var_names_and_desc]
    for var in var_names_and_desc:
        causal_var = var_names_and_desc[var]
        message_history[1]['content'] += f'''{var}: {causal_var.description}\n'''

    message_history[1]['content'] += prompt_init
    print('msg history:', message_history[1]['content'])

    response = client.chat.completions.create(
    model=model,
    messages=message_history,
    temperature=temperature,
    max_tokens=4095,
    # max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    answer = response.choices[0].message.content
    
    message_history.append({
        "role": "assistant",
        "content": answer
    })
    # print(f'{answer=}')
    answer = answer.split('<Answer>')[1].split('</Answer>')[0].split(', ')
    print(f'{answer=}')
  
    independent_nodes = answer.copy()
    unvisited_nodes = nodes.copy()
    for node in answer:
        unvisited_nodes.remove(node)
    frontier = []

    predict_graph = dict()

    for to_visit in independent_nodes:
        prompt = 'Given ' + ', '.join(independent_nodes) + ' is(are) not affected by any other variable'
        if len(predict_graph) == 0:
            prompt += '.\n'
        else:
            prompt += ' and the following causal relationships.\n'
            for head,tails in predict_graph.items():
                if len(tails) > 0:
                    prompt += f'{head} causes ' + ', '.join(tails) + '\n'
                    # prompt += f'{head} affects ' + ', '.join(tails) + '\n'

        prompt += f'Select variables that are caused by {to_visit}.\n'
        # prompt += f'Select variables that are affected by {to_visit}.\n'
        
        if include_statistics:
          if include_obs:
            prompt += get_data_prompt_obs(to_visit, df)
          else:
            prompt += get_data_prompt(to_visit, corr)
            # 

        prompt += prompt_format
        print(f'{prompt=}')
        # sys.exit()

        message_history.append({
                "role": "user",
                "content": prompt
            })
        
        response = client.chat.completions.create(
        model=model,
        messages=message_history,
        temperature=temperature,
        max_tokens=4095,
        # max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        answer = response.choices[0].message.content
        message_history.append({
            "role": "assistant",
            "content": answer
        })
        answer = answer.split('<Answer>')[1].split('</Answer>')[0].split(', ')
        print(f'{answer=}')
        for node in answer:
            if node in independent_nodes:
                answer.remove(node)
            if len(node) == 0:
                answer.remove(node)
            elif node not in nodes:
                print("ERROR: ", node)
                answer.remove(node)

        predict_graph[to_visit] = answer
        for node in answer:
            if node in unvisited_nodes and node not in frontier:
                frontier.append(node)

    while len(frontier) > 0:
        print("Frontier: ", frontier)
        print("Unvisited nodes: ", unvisited_nodes)
        to_visit = frontier.pop(0)
        unvisited_nodes.remove(to_visit)
        print("Visiting: ", to_visit)
        prompt = 'Given ' + ', '.join(independent_nodes) + ' is(are) not affected by any other variable and the following causal relationships.\n'  
        for head,tails in predict_graph.items():
            if len(tails) > 0:
                prompt += f'{head} causes ' + ', '.join(tails) + '\n'
                # prompt += f'{head} affects ' + ', '.join(tails) + '\n'

        prompt += f'Select variables that are caused by {to_visit}.\n'
        # prompt += f'Select variables that are affected by {to_visit}.\n'

        if include_statistics:
          if include_obs:
            prompt += get_data_prompt_obs(to_visit, df)
          else:
            prompt += get_data_prompt(to_visit, corr)

        # if include_statistics:
        #     prompt += get_data_prompt(to_visit, corr)
        #     # prompt += get_data_prompt_obs(to_visit, df)

        prompt += prompt_format
        
        print(f'{prompt=}')
        print('Start generating...')
        message_history.append({
            "role": "user",
            "content": prompt
        })
        response = client.chat.completions.create(
        model=model,
        messages=message_history,
        temperature=temperature,
        max_tokens=4095,
        # max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        answer = response.choices[0].message.content
        message_history.append({
            "role": "assistant",
            "content": answer
        })
        answer = answer.split('<Answer>')[1].split('</Answer>')[0].split(', ')
        print(answer)

        for node in answer:
            if node in independent_nodes:
                answer.remove(node)
            if len(node) == 0:
                answer.remove(node)
            elif node not in nodes:
                print("ERROR: ", node)
                answer.remove(node)

            
        predict_graph[to_visit] = answer
        for node in answer:
            if node in unvisited_nodes and node not in frontier:
                frontier.append(node)
        


    print(predict_graph)
    df_order = [var for var in df.columns]
    n = len(df_order)
    adj_matrix = np.zeros((n, n))
    for i, var in enumerate(df_order):
        if var in predict_graph:
            for node in predict_graph[var]:
                j = df_order.index(node)
                adj_matrix[i][j] = 1
    print(adj_matrix)
    return adj_matrix