python ../../../a3c_eplus_rlParametric_v0.1.py --env Part4-Heavy-Pit-Train-Repeat-Cmp97-v1 --max_interactions 10000000 --window_len 3 \
--state_dim 13 --num_threads 16 --action_space iw_af4_3 --save_freq 500000 --eval_freq 50000 \
--job_mode Train --test_env Part4-Heavy-Pit-Test-Repeat-Real97-v1 \
--train_act_func 8 --eval_act_func 8 \
--reward_func part4_heuri_cmp97 --metric_func part4_cmp97 --init_e 0.0 --rwd_e_para 1.0 --rwd_p_para 10 \
--h_regu_frac 0.0 --forecast_dim 0 --rmsprop_decay 0.99 --rmsprop_momet 0.0 --train_freq 5 \
--violation_penalty_scl 5 --eval_epi_num 1 --activation relu --model_type nn --model_param 512 4 \
--learning_rate 0.00005 --learning_rate_decay_rate 1.0 --learning_rate_decay_steps 100000 --debug_log_prob 0.001 \
--isNoisyNet True --isNoisyNetEval_rmNoise True --eval_env_res_max_keep 200 --clip_norm 5.0
