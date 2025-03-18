
insurance_prompt=[
    {
      "role": "system",
      "content": "You are an expert on insurance."
    },
    {
      "role": "user",
      "content": "You are a helpful assistant to experts in insurance. Our goal is to construct a causal graph between the following variables.\n"
    }
  ]

sprinkler_prompt=[
    {
      "role": "system",
      "content": "You are an expert on causality."
    },
    {
      "role": "user",
      "content": "You are a helpful assistant to experts in causality. Our goal is to construct a causal graph between the following variables.\n"
    }
  ]

survey_prompt=[
    {
      "role": "system",
      "content": "You are an expert on social causality."
    },
    {
      "role": "user",
      "content": "You are a helpful assistant to experts in social causality. Our goal is to construct a causal graph illustrating how public transport usage differs among social groups, between the following variables.\n"
    }
  ]


cancer_prompt=[
    {
      "role": "system",
      "content": "You are an expert on cancer."
    },
    {
      "role": "user",
      "content": "You are a helpful assistant to experts in cancer research. Our goal is to construct a causal graph between the following variables.\n"
    }
  ]

asia_prompt=[
    {
      "role": "system",
      "content": "You are an expert on lung diseases."
    },
    {
      "role": "user",
      "content": "You are a helpful assistant to experts in lung diesease research. Our goal is to construct a causal graph between the following variables.\n"
    }
  ]


child_prompt=[
    {
      "role": "system",
      "content": "You are an expert on children's diseases."
    },
    {
      "role": "user",
      "content": "You are a helpful assistant to experts in children's diesease research. Our goal is to construct a causal graph between the following variables.\n"
    }
  ]

sachs_prompt=[
    {
      "role": "system",
      "content": "You are an expert on protein and gene research."
    },
    {
      "role": "user",
      "content": "You are a helpful assistant to experts in protein and gene research. Our goal is to construct a causal graph between the following protein or gene interaction.\n"
    }
  ]

earthquake_prompt=[
    {
      "role": "system",
      "content": "You are an expert on earthquake research."
    },
    {
      "role": "user",
      "content": "You are a helpful assistant to experts in earthquake research. Our goal is to construct a causal graph between the following protein or gene interaction.\n"
    }
  ]

alarm_prompt=[
    {
      "role": "system",
      "content": "You are an expert on alarm systems for patient monitoring."
    },
    {
      "role": "user",
      "content": "You are a helpful assistant to experts in alarm systems for patient monitoring. Our goal is to construct a causal graph between the following protein or gene interaction.\n"
    }
  ]




prompt_init = 'Now you are going to use the data to construct a causal graph. You will start with identifying the variable(s) that are unaffected by any other variables.\n'
prompt_format = 'Think step by step. Then, provide your final answer (variable names only) within the tags <Answer>...</Answer>, separated by ", ". '
prompt_init += prompt_format

def get_data_prompt(to_visit, corr):
    prompt = f'Addtionally, the Pearson correlation coefficient between {to_visit} and other variables are as follows:\n'
    for var in corr:
        if var == to_visit:
            continue
        prompt += f'{var}: {corr[var][to_visit]:.2f}\n'
    print (prompt)
    return prompt

# def get_data_prompt(to_visit, corr):
#     prompt = f'Addtionally, the Pearson correlation coefficient between {to_visit} and other variables are as follows:\n'
#     for var in corr:
#         if var == to_visit:
#             continue
#         prompt += f'{var}: {corr[var][to_visit]:.2f}\n'
#     return prompt

def get_data_prompt_obs(to_visit, df):
    # prompt = f'Addtionally, the Pearson correlation coefficient between {to_visit} and other variables are as follows:\n'
    prompt = f'Additionally, some sample of the observational data for {to_visit} and other variables are as follows:\n'
    # take = df[:10]
    # take = df.sample(n=n_samples, random_state=1111)
    prompt += f'{df.to_string(index=False)}\n'
    
      
    # for var in corr:
    #     if var == to_visit:
    #         continue
    #     # To help you, here is the observational data for "{head.name}" and "{tail.name}": {arr[:10]}
    #     #     '''
    #     arr = corr[[var, to_visit]].to_numpy()[:10]
    #     prompt += f'with {var}: {arr}\n'
    #     # prompt += f'{var}: {corr[var][to_visit]:.2f}\n'
    return prompt