python ../../../a3c_eplus_rlParametric_v0.1.py --env Part4-Light-Pit-Train-Repeat-v2 --max_interactions 2500000 --window_len 10 \
--state_dim 11 --num_threads 16 --action_space part4_v3 --save_freq 500000 --eval_freq 50000 \
--job_mode Train --test_env Part4-Light-Pit-Test-Repeat-v3 Part4-Light-Pit-Test-Repeat-v4 \
--train_act_func part4_v4 --eval_act_func part4_v4 \
--reward_func part4_heuri_v7 --metric_func part4_v2 --init_e 0.0 --rwd_e_para 1.0 --rwd_p_para 15 \
--h_regu_frac 0.0 --forecast_dim 0 --rmsprop_decay 0.99 --rmsprop_momet 0.0 --train_freq 25 \
--violation_penalty_scl 5 --eval_epi_num 1 --activation relu --model_type nn --model_param 64 8 \
--learning_rate 0.001 --learning_rate_decay_rate 1.0 --learning_rate_decay_steps 100000 --debug_log_prob 0.001 \
--isNoisyNet True --isNoisyNetEval_rmNoise True --eval_env_res_max_keep 50 --clip_norm 1.0
