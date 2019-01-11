python ../../a3c_eplus_iw_v0.1.py --env IW-tmy3Weather-v9602 --max_interactions 10000000 --window_len 8 --state_dim 13 --num_threads 16 \
--action_space iw_af5_1 --save_freq 500000 --eval_freq 250000 --job_mode Train \
--test_env IW-realWeather-v9601 --err_penalty_scl 0.15 --act_func 6 --reward_func 10 --init_e 0.0 --rwd_e_para 0.9 --rwd_p_para 0.1 \
--h_regu_frac 0.01 --forecast_dim 0
