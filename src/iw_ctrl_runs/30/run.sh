python ../../a3c_eplus_iw_v0.1.py --env IW-v5702 --max_interactions 15000000 --window_len 24 --state_dim 13 --num_threads 16 \
--action_space iw_af5_1 --save_freq 500000 --eval_freq 250000 --job_mode Train \
--test_env IW-eval-v5702 --err_penalty_scl 0.30 --act_func 5 --reward_func 7 --init_e 0.0 --rwd_e_para 0.5 --rwd_p_para 0.5 \
--h_regu_frac 0.1
