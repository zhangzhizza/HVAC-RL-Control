python ../../a3c_eplus_iw_v0.1.py --env IW-imp-v9704 --max_interactions 10000000 --window_len 3 --state_dim 13 --num_threads 1 \
--action_space iw_af4_3 --save_freq 500000 --eval_freq 0 --job_mode Train \
--test_env IW-imp-v9704 --err_penalty_scl 0.20 --train_act_func 10 --eval_act_func 10 --reward_func 13 --init_e 0.0 --rwd_e_para 2.5 --rwd_p_para 1.0 \
--h_regu_frac 0.05 0.05 --h_decay_bounds 2000000 --forecast_dim 0 --rmsprop_decay 0.90 --train_freq 5 --violation_penalty_scl 5.0 --is_warm_start True \
--model_dir v93_model_data/model.ckpt-6500000 --debug_log_prob 1.0
