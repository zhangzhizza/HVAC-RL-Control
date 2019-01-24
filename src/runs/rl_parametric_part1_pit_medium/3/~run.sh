python ../../../a3c_eplus_rlParametric_v0.1.py --env Part1-Medium-Pit-Train-v1 --max_interactions 5000000 --window_len 1 \
--state_dim 71 --num_threads 16 --action_space part1_v1 --save_freq 500000 --eval_freq 250000 \
--job_mode Train --test_env Part1-Medium-Pit-Test-v1 Part1-Medium-Pit-Test-v2 --train_act_func part1_v1 --eval_act_func part1_v1 \
--reward_func part1_v1 --metric_func part1_v1 --init_e 0.0 --rwd_e_para 1.0 --rwd_p_para 1.0 \
--h_regu_frac 0.0 --forecast_dim 0 --rmsprop_decay 0.99 --rmsprop_momet 0.0 --train_freq 5 \
--violation_penalty_scl 10 --eval_epi_num 1 --activation relu --model_type nn --model_param 64 4 \
--learning_rate 0.00001 --learning_rate_decay_rate 0.90 --learning_rate_decay_steps 100000 --debug_log_prob 0.0005 \
--isNoisyNet True --isNoisyNetEval_rmNoise True --check_args_only True
