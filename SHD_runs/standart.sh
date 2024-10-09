python3 ../run_exp.py --sweep_name S4orig128x2nbSteps --method grid --s4 True --lr 0.01 --nb_state 2 --nb_layers 2 --activation GELU --use_readout_layer True --pure_complex True --normalization batchnorm --max_time 1 --spatial_bin 5 --time_offset 0 10 --nb_steps 250 150 --data_folder ~/local/SHD 



--prenorm True False --premix True False --mix GLU Linear --residual2 True False --drop2 True False --thr 0 1.0
