#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

python run.py --dataset cancer --alg llm_obs --n_samples 100 --sample_method cluster --temp 0.5 #5
python run.py --dataset cancer --alg llm_obs --n_samples 100 --sample_method systematic --temp 0.5 #5
python run.py --dataset cancer --alg llm_obs --n_samples 100 --sample_method adaptive --temp 0.5 #5

python run.py --dataset cancer --alg llm_obs --n_samples 100 --sample_method cluster --temp 0.7 #5
python run.py --dataset cancer --alg llm_obs --n_samples 100 --sample_method systematic --temp 0.7 #5
python run.py --dataset cancer --alg llm_obs --n_samples 100 --sample_method adaptive --temp 0.7 #5

python run.py --dataset cancer --alg llm_obs --n_samples 100 --sample_method cluster --temp 1 #5
python run.py --dataset cancer --alg llm_obs --n_samples 100 --sample_method systematic --temp 1 #5
python run.py --dataset cancer --alg llm_obs --n_samples 100 --sample_method adaptive --temp 1 #5

# python run.py --dataset asia --alg llm_obs --n_samples 100 --temp 0.5 #5
# python run.py --dataset asia --alg llm_obs --n_samples 100 --temp 0.7 #5
# python run.py --dataset asia --alg llm_obs --n_samples 100 --temp 1 #5

# python run.py --dataset survey --alg llm_obs --n_samples 100 --temp 0 #5
# python run.py --dataset survey --alg llm_obs --n_samples 100 --temp 0.7 #5
# python run.py --dataset survey --alg llm_obs --n_samples 100 --temp 0.5 #5
# python run.py --dataset survey --alg llm_obs --n_samples 100 --temp 1 #5

# python run.py --dataset survey --alg stat_all --n_samples 1000
 #5
# python run.py --dataset survey --alg stat_all --n_samples 100

# python run.py --dataset cancer --alg stat_all --n_samples 500 --temp 0.7
# python run.py --dataset survey --alg stat_all --n_samples 1000 --temp 0.7

# python run.py --dataset survey --alg llm_only --n_samples 100 --temp 0.7

# python run.py --dataset cancer --alg stat_all --n_samples 1000 --temp 0.5
# # python run.py --dataset survey --alg llm_only --n_samples 100 --temp 0.5

# python run.py --dataset cancer --alg stat_all --n_samples 1000 --temp 0
# # python run.py --dataset survey --alg llm_only --n_samples 100 --temp 0

# python run.py --dataset cancer --alg stat_all --n_samples 1000 --temp 1
# # python run.py --dataset survey --alg llm_only --n_samples 100 --temp 1

# python run.py --dataset survey --alg llm_obs --n_samples 100 --temp 1 #5
# python run.py --dataset survey --alg llm_only --n_samples 100 --temp 1

# python run.py --dataset earthquake --alg llm_only --n_samples 100 --temp 0
# python run.py --dataset earthquake --alg llm_obs --n_samples 100 --temp 0

# python run.py --dataset earthquake --alg llm_obs --n_samples 100 --temp 0.5
# python run.py --dataset earthquake --alg llm_only --n_samples 100 --temp 0.5

# python run.py --dataset earthquake --alg llm_only --n_samples 100 --temp 1
# python run.py --dataset earthquake --alg llm_obs --n_samples 100 --temp 1

#['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'] 

# python run.py --dataset earthquake --alg llm_obs --n_samples 100 --temp 0.7 #5
# #['Burglary', 'Alarm', 'Earthquake', 'JohnCalls', 'MaryCalls']

# python run.py --dataset insurance --alg llm_obs --n_samples 100 --temp 0.7 #27
# python run.py --dataset insurance --alg llm_only --n_samples 100 --temp 0.7


# python run.py --dataset survey --alg llm_obs --n_samples 100 --temp 0.7 #??
# python run.py --dataset water --alg llm_obs --n_samples 100 --temp 0.7 #32
# python run.py --dataset diabetes --alg llm_obs --n_samples 100 --temp 0.7 #413

