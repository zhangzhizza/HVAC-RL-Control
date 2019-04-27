python ../../../a3c_eplus_rlParametric_v0.1.py --env Part4-Light-Pit-Train-Repeat-v1 --max_interactions 2000000 --window_len 7 \
--state_dim 10 --num_threads 16 --action_space part4_v2 --save_freq 500000 --eval_freq 20000 \
--job_mode Train --test_env Part4-Light-Pit-Test-Repeat-v1 Part4-Light-Pit-Test-Repeat-v2 \
--train_act_func part4_v3 --eval_act_func part4_v3 \
--reward_func part4_heuri_v3 --metric_func part4_v1 --init_e 0.0 --rwd_e_para 1.0 --rwd_p_para 0.5 \
--h_regu_frac 0.0 --forecast_dim 0 --rmsprop_decay 0.99 --rmsprop_momet 0.0 --train_freq 5 \
--violation_penalty_scl 10 --eval_epi_num 1 --activation relu --model_type nn --model_param 64 2 \
--learning_rate 0.00001 --learning_rate_decay_rate 1.0 --learning_rate_decay_steps 100000 --debug_log_prob 0.005 \
--isNoisyNet True --isNoisyNetEval_rmNoise True --eval_env_res_max_keep 50
