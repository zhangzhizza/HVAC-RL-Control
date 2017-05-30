python ../a3c_eplus_v0.1.py --window_len 24 --e_weight 0.6 --p_weight 0.4 --ppd_penalty_limit 0.15 --action_space default --eval_epi_num 1 --dropout_prob 0.0 \
--is_warm_start True --model_dir ../a3c-res-v0.1/case20-2/a3c-res-v0.1/Eplus-v1-run1/model_data/model.ckpt-15000000 --job_mode Test \
--test_env Eplus-multiagent-v1 --agent_num 5
