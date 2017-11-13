python ../../a3c_eplus_iw_v0.1.py --env IW-v57 --max_interactions 15000000 --window_len 24 --state_dim 11 --num_threads 16 \
--e_weight 1.0 --p_weight 0.0 --action_space iw_2 --save_freq 500000 --eval_freq 500000 --job_mode Train \
--test_env IW-eval-v57 --err_penalty_scl 40.0 --act_func 2 --reward_func 2 --init_e 0.5 --end_e 0.0 --decay_steps 2000000
