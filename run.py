import bnlearn as bn
from algs.pc import pc
from algs.ges import ges
from algs.pairwise import llm_pairwise , midllm_pairwise
from algs.bfs import llm_bfs
from algs.dagma_baseline.dagma_wrapper import dagma_nonlinear_wrapper, dagma_linear_wrapper
import numpy as np
from metrics import compute_metrics
from llm_model import load_models
from args import get_args
import json
import os
import pandas as pd
from init_prompts import *
from sklearn.cluster import KMeans

# python run.py --dataset asia --alg llm_pairwise --n_samples 100

def write_metric(adj_mat, predicted_adj_mat, logdir, dataset, n_samples, alg):
  print (f'{predicted_adj_mat=}')
  metrics = compute_metrics(adj_mat, predicted_adj_mat)
  print (f'{metrics=}')
  logdir = f"{logdir}/{dataset}/{n_samples}"
  os.makedirs(logdir, exist_ok=True)
  logfile = f"{logdir}/{alg}" 
  logfile += ".json"
  with open(logfile, "w") as outfile: 
      json.dump(metrics, outfile)

def data_sampling(df, size, seed, sample_method):
  if sample_method == 'systematic':
    df_sampled = df.iloc[::10] #step=10 pool/size n
  elif sample_method == 'cluster':
    df['cluster'] = np.random.randint(0, 10, size=1000)  # Assigning clusters (10 clusters: 0-9)
    # Step 1: Take 10 samples from each cluster
    df_sampled = df.groupby('cluster').apply(lambda x: x.sample(n=10, random_state=seed)).reset_index(drop=True)
  elif sample_method == 'adaptive':
    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    df['cluster'] = kmeans.fit_predict(df)  # Use relevant numerical columns
    df_sampled = df.groupby('cluster').apply(lambda x: x.sample(n=10, random_state=seed)).reset_index(drop=True)
  else:
    df_sampled = df.sample(n=size, random_state=seed)
  return df_sampled

args = get_args()
print (args)

if args.dataset == "neuropathic":
  adj_mat = np.load('data/neuropathic_dag_gt.npy')
  df = pd.read_csv(f'data/neuropathic_data_{args.pool_size}.csv')
elif args.dataset in ["sprinkler", "titanic", "sachs", "auto"]:
  try:
    dag = bn.import_DAG(args.dataset, verbose=True)
  except:
    dag = bn.import_example(args.dataset, verbose=True)
  var_names = dag['adjmat'].columns
  adj_mat = dag['adjmat'].to_numpy().astype(int)
  df = bn.sampling(dag, n=args.pool_size)
else:
  dag = bn.import_DAG(f'./data/{args.dataset}.bif')
  var_names = dag['adjmat'].columns
  adj_mat = dag['adjmat'].to_numpy().astype(int)
  df = bn.sampling(dag, n=args.pool_size)

df = data_sampling(df, args.n_samples, args.seed, args.sample_method)
data = df.to_numpy()
# print (f'{df.shape=}')
# print (df.head())
# print (df.columns.tolist(), len(df.columns.tolist()))
# print (adj_mat)
# sys.exit()
#      asia  tub  smoke  lung  bronc  either  xray  dysp
# 276     1    1      1     1      0       1     1     0
# 202     1    1      1     1      0       1     1     1
# 557     1    1      1     1      1       1     1     1
# 400     1    1      0     1      0       1     1     0
# 15      1    1      1     1      1       1     1     1

# np.random.seed(args.seed)
gaussian_noise = np.random.normal(loc=0.0, scale=0.00001, size=data.shape)

if args.alg == 'stat_all':
  data = data + gaussian_noise
  print (f'\nRunning PC...')
  predicted_adj_mat_pc = pc(data)
  write_metric(adj_mat, predicted_adj_mat_pc, args.logdir, args.dataset, args.n_samples, 'pc')
  # print (f'\nRunning GES...')
  # predicted_adj_mat_ges = ges(data, score_func='local_score_BIC', maxP=None, parameters=None)
  # write_metric(adj_mat, predicted_adj_mat_ges, args.logdir, args.dataset, args.n_samples, 'ges')
  # print (f'\nRunning NOTEARS...')
  # predicted_adj_mat_note = notears(data, lambda1=args.lambda1, loss_type='l2', w_threshold=args.w_threshold)
  # write_metric(adj_mat, predicted_adj_mat_note, args.logdir, args.dataset, args.n_samples, 'notears')
  # print (f'\nRunning DAGMA NONLINEAR...')
  # predicted_adj_mat_dnonl = dagma_nonlinear_wrapper(data, len(adj_mat), args.lambda1, args.lambda2)
  # write_metric(adj_mat, predicted_adj_mat_dnonl, args.logdir, args.dataset, args.n_samples, 'dagma_nonlinear')
  # print (f'\nRunning DAGMA LINEAR...')
  # predicted_adj_mat_dl = dagma_linear_wrapper(data, args.lambda1)
  # write_metric(adj_mat, predicted_adj_mat_dl, args.logdir, args.dataset, args.n_samples, 'dagma_linear')

elif args.alg == 'llm_obs':
  print (f'\nRunning LLM BFS with statistics...')
  predicted_adj_mat_bfsobs = llm_bfs(VAR_NAMES_AND_DESC[args.dataset], args.dataset, df, args.temp, include_statistics=True, include_obs=False)
  write_metric(adj_mat, predicted_adj_mat_bfsobs, args.logdir, args.dataset, args.n_samples, f'llm{args.temp}_bfs_with_stat')

  print (f'\nRunning LLM pairwise with statistics...')
  predicted_adj_mat_llmpobs = llm_pairwise(VAR_NAMES_AND_DESC[args.dataset], prompts[args.dataset], df, args.temp, include_statistics=True)
  write_metric(adj_mat, predicted_adj_mat_llmpobs, args.logdir, args.dataset, args.n_samples, f'llm{args.temp}_pairwise_with_stat')

  print (f'\nRunning LLM BFS with obs...')
  predicted_adj_mat_bfsobs = llm_bfs(VAR_NAMES_AND_DESC[args.dataset], args.dataset, df, args.temp, include_statistics=True, include_obs=True)
  write_metric(adj_mat, predicted_adj_mat_bfsobs, args.logdir, args.dataset, args.n_samples, f'llm{args.temp}{args.sample_method}_bfs_with_obs')
  
  print (f'\nRunning LLM pairwise with obs...')
  predicted_adj_mat_llmpobs = llm_pairwise(VAR_NAMES_AND_DESC[args.dataset], prompts[args.dataset], df, args.temp, include_statistics=True)
  write_metric(adj_mat, predicted_adj_mat_llmpobs, args.logdir, args.dataset, args.n_samples, f'llm{args.temp}{args.sample_method}_pairwise_with_obs')

elif args.alg == 'llm_only':
  print (f'\nRunning LLM BFS...')
  predicted_adj_mat_bfs = llm_bfs(VAR_NAMES_AND_DESC[args.dataset], args.dataset, df, args.temp)
  write_metric(adj_mat, predicted_adj_mat_bfs, args.logdir, args.dataset, args.n_samples, f'llm{args.temp}_bfs')

  print (f'\nRunning LLM pairwise...')
  predicted_adj_mat_llmp = llm_pairwise(VAR_NAMES_AND_DESC[args.dataset], prompts[args.dataset], df, args.temp, include_statistics=False)
  write_metric(adj_mat, predicted_adj_mat_llmp, args.logdir, args.dataset, args.n_samples, f'llm{args.temp}_pairwise')

# elif args.alg == 'llm_pairwise_all':
#   # print (f'\nRunning LLM pairwise...')
#   # predicted_adj_mat_llmp = llm_pairwise(VAR_NAMES_AND_DESC[args.dataset], prompts[args.dataset], df, include_statistics=False)
#   # write_metric(adj_mat, predicted_adj_mat_llmp, args.logdir, args.dataset, args.n_samples, 'llm_pairwise')
#   # print (f'\nRunning LLM pairwise with obs...')
#   # predicted_adj_mat_llmpobs = llm_pairwise(VAR_NAMES_AND_DESC[args.dataset], prompts[args.dataset], df, include_statistics=True)
#   # write_metric(adj_mat, predicted_adj_mat_llmpobs, args.logdir, args.dataset, args.n_samples, 'llm_pairwise_with_obs')
#   print (f'\nRunning LLM pairwise with other models...')
#   # model_name="meta-llama/Meta-Llama-3-8B-Instruct"
#   # model_name = "mistralai/Mistral-7B-Instruct-v0.1"
#   model_name="google/gemma-7b-it"
#   model, tokenizer = load_models(model_name)
#   # predicted_adj_mat_llm_mid = midllm_pairwise(VAR_NAMES_AND_DESC[args.dataset], df, model, tokenizer, max_new_tokens=50, include_statistics=False)
#   # write_metric(adj_mat, predicted_adj_mat_llm_mid, args.logdir, args.dataset, args.n_samples, f'llm_pairwise_{model_name[:5]}')
#   print (f'\nRunning LLM pairwise with other models with obs data...')
#   predicted_adj_mat_llm_mid_obs = midllm_pairwise(VAR_NAMES_AND_DESC[args.dataset], prompts[args.dataset], df, model, tokenizer, max_new_tokens=50, include_statistics=True)
#   write_metric(adj_mat, predicted_adj_mat_llm_mid_obs, args.logdir, args.dataset, args.n_samples, f'llm_pairwise_{model_name[:5]}_obs')


# elif args.alg == 'llm_bfs_all':
#   # print (f'\nRunning LLM BFS...')
#   # predicted_adj_mat_bfs = llm_bfs(VAR_NAMES_AND_DESC[args.dataset], args.dataset, df)
#   # write_metric(adj_mat, predicted_adj_mat_bfs, args.logdir, args.dataset, args.n_samples, 'llm_bfs')
#   # print (f'\nRunning LLM BFS with corr...')
#   # predicted_adj_mat_bfscorr = llm_bfs(VAR_NAMES_AND_DESC[args.dataset], args.dataset, df, include_statistics=True)
#   # write_metric(adj_mat, predicted_adj_mat_bfscorr, args.logdir, args.dataset, args.n_samples, 'llm_bfs_with_corr')
#   print (f'\nRunning LLM BFS with obs...')
#   predicted_adj_mat_bfsobs = llm_bfs(VAR_NAMES_AND_DESC[args.dataset], args.dataset, df, include_statistics=True, include_obs=True)
#   write_metric(adj_mat, predicted_adj_mat_bfsobs, args.logdir, args.dataset, args.n_samples, 'llm_bfs_with_obs')

# elif args.alg == 'ges':
#   data = data + gaussian_noise
#   predicted_adj_mat = ges(data, score_func='local_score_BIC', maxP=None, parameters=None)
# elif args.alg == 'pc':
#   data = data + gaussian_noise
#   predicted_adj_mat = pc(data)
# elif args.alg == 'dagma_nonlinear':
#   predicted_adj_mat = dagma_nonlinear_wrapper(data, len(adj_mat), args.lambda1, args.lambda2)
# elif args.alg == 'dagma_linear':
#   predicted_adj_mat = dagma_linear_wrapper(data, args.lambda1)
# elif args.alg == 'notears':
#   predicted_adj_mat = notears(data, lambda1=args.lambda1, loss_type='l2', w_threshold=args.w_threshold)
# elif args.alg == 'llm_pairwise':
#   predicted_adj_mat = llm_pairwise(VAR_NAMES_AND_DESC[args.dataset], prompts[args.dataset], df, include_statistics=False)
# elif args.alg == 'llm_pairwise_with_obs': #with obs
#   predicted_adj_mat = llm_pairwise(VAR_NAMES_AND_DESC[args.dataset], prompts[args.dataset], df, include_statistics=True)
# elif args.alg == 'llm_bfs':
#   predicted_adj_mat = llm_bfs(VAR_NAMES_AND_DESC[args.dataset], args.dataset, df, include_statistics=False)
# elif args.alg == 'llm_bfs_with_statistics':
#   predicted_adj_mat = llm_bfs(VAR_NAMES_AND_DESC[args.dataset], args.dataset, df, include_statistics=True)
# elif args.alg == 'llm_bfs_with_obs':
#   predicted_adj_mat = llm_bfs(VAR_NAMES_AND_DESC[args.dataset], args.dataset, df, include_statistics=True, include_obs=True)
else:
  raise ValueError(f"Unknown algorithm {args.alg}")

# metrics = compute_metrics(adj_mat, predicted_adj_mat)
# logdir = f"{args.logdir}/{args.dataset}/{args.n_samples}"
# os.makedirs(logdir, exist_ok=True)
# logfile = f"{logdir}/{args.alg}" 
# if args.alg in ['dagma_nonlinear', 'dagma_linear', 'notears']:
#   logfile += f"_lambda1={args.lambda1}"
# logfile += ".json"
# with open(logfile, "w") as outfile: 
#     json.dump(metrics, outfile)
