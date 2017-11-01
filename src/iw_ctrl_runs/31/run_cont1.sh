python ../../a3c_eplus_iw_v0.1.py --env IW-v5702 --max_interactions 3500000 --window_len 12 --state_dim 13 --num_threads 16 \
--action_space iw_af5_1 --save_freq 500000 --eval_freq 250000 --job_mode Train \
--test_env IW-eval-v5702 --err_penalty_scl 0.30 --act_func 5 --reward_func 10 --init_e 0.0 --rwd_e_para 0.5 --rwd_p_para 0.5 \
--h_regu_frac 0.1 --is_warm_start True --model_dir ~/Documents/HVAC-RL-Control/src/iw_ctrl_runs/31/a3c-res-v0.1/IW-v5702-run1/model_data/model.ckpt-11500000
