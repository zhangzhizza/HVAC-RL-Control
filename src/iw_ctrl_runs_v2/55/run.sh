python ../../a3c_eplus_iw_v0.1.py --env IW-tmy3Weather-v9602 --max_interactions 10000000 --window_len 3 --state_dim 13 --num_threads 16 \
--action_space iw_af4_2 --save_freq 500000 --eval_freq 250000 --job_mode Train \
--test_env IW-realWeather-v9606 --err_penalty_scl 0.15 --act_func 7 --reward_func 13 --init_e 0.0 --rwd_e_para 1.0 --rwd_p_para 1.0 \
--h_regu_frac 0.1 0.01 --h_decay_bounds 1000000 --forecast_dim 0 --rmsprop_decay 0.90 --train_freq 5 --violation_penalty_scl 5.0
