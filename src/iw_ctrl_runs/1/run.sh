python ../../a3c_eplus_iw_v0.1.py --env IW-v57 --max_interactions 10000000 --window_len 24 --state_dim 11 --num_threads 1 \
--e_weight 0.5 --p_weight 0.5 --action_space iw_1 --save_freq 50000 --eval_freq 500000 --job_mode Train \
--test_env IW-eval-v57 --err_penalty_scl 13.0 