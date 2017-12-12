python ../../a3c_eplus_iw_v0.1.py --env IW-tmy3Weather-v9606 --max_interactions 10000000 --window_len 24 --state_dim 13 --num_threads 16 \
--action_space iw_af5_1 --save_freq 500000 --eval_freq 250000 --job_mode Train \
--test_env IW-realWeather-v9606 --err_penalty_scl 0.15 --act_func 6 --reward_func 11 --init_e 0.0 --rwd_e_para 0.5 --rwd_p_para 0.5 \
--h_regu_frac 1 0.5 0.1 0.05 --h_decay_bounds 2000000 4000000 6000000 --forecast_dim 0 --train_freq 10
