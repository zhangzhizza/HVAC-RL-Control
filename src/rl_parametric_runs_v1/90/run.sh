python ../../a3c_eplus_rlParametric_v0.1.py --env Model1-Cool-v1 --max_interactions 10000000 --window_len 6 \
--state_dim 49 --num_threads 16 --action_space cslDxCool_1 --save_freq 500000 --eval_freq 250000 \
--job_mode Train --test_env Model1-Test-Cool-v1 --train_act_func cslDxActCool_1 --eval_act_func cslDxActCool_1 \
--reward_func cslDxCool_1 --metric_func cslDxCool_1 --init_e 0.0 --rwd_e_para 1.0 --rwd_p_para 1.0 \
--h_regu_frac 0.1 --forecast_dim 0 --rmsprop_decay 0.99 --rmsprop_momet 0.0 --train_freq 5 \
--violation_penalty_scl 10 --eval_epi_num 1 --activation linear --model_type nn --model_param 256 8 \
--learning_rate 0.001 --learning_rate_decay_rate 0.90 --learning_rate_decay_steps 100000 --debug_log_prob 0.0005
