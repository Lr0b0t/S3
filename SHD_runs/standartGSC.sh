python3 ../run_exp.py --sweep_name S4_128x2 --data_folder ~/local/GSC --dataset_name sc --method grid --s4 True --lr 0.001 --scheduler_patience 5 --batch_size 32 --nb_state 2 4 64 128 --nb_hidden 128 512 --nb_layers 2 3 --activation step --use_readout_layer True False --pure_complex True --normalization batchnorm layernorm  --seed 13 42 73 190 268 --gpu_device 0


python3 ../run_exp.py --sweep_name S4_128x2 --data_folder ~/local/GSC --dataset_name sc --method grid --s4 True --gpu_device 0


python3 ../run_exp.py --sweep_name S4_128x2 --data_folder /Data/pgi-15/datasets/GSC--dataset_name sc --s4 False --method grid --lr 0.001 --scheduler_patience 5 --batch_size 32 --nb_state 2 4 64 128 --nb_hidden 128 512 --nb_layers 2 3 --activation step --use_readout_layer True False --pure_complex True --normalization batchnorm layernorm  --seed 13 42 73 190 268 --gpu_device 0

python3 ../run_exp.py --sweep_name S4_128_gelu --data_folder /Data/pgi-15/datasets/GSC --dataset_name sc --s4 True --method grid --lr 0.001 --scheduler_patience 5 --batch_size 32 --nb_state 2 4 --nb_hidden 128  --nb_layers 3 --activation GELU --use_readout_layer True False --pure_complex True --normalization batchnorm layernorm  --seed 13 42 73 190 268 --gpu_device 0


python3 ../run_exp.py --sweep_name alphaImgValues --data_folder /Users/dudchenko/local/GSC --dataset_name sc --s4 False --method grid --lr 0.001 --scheduler_patience 5 --batch_size 32 --nb_layers 3 --nb_hidden 512 --use_readout_layer True --pdrop 0.25 --seed 13 --alpha_imag 0.785 1.57 6.28

python3 ../run_exp.py --sweep_name alphaImgValues --model_type LIFcomplex --data_folder /Users/dudchenko/local/GSC --dataset_name sc --s4 False --method grid --lr 0.001 --scheduler_patience 5 --batch_size 32 --nb_layers 3 --nb_hidden 512 --use_readout_layer True --pdrop 0.25 --seed 13 --alpha_imag 1.57 6.28 --alpha_range False
python3 ../run_exp.py --sweep_name alphaImgRange --model_type LIFcomplex --data_folder /Users/dudchenko/local/GSC --dataset_name sc --s4 False --method grid --lr 0.001 --scheduler_patience 5 --batch_size 32 --nb_layers 3 --nb_hidden 512 --use_readout_layer True --pdrop 0.25 --seed 13 --alpha_imag 3.14 1.57 --alpha_range False --alpha_rand Rand RandN
python3 ../run_exp.py --sweep_name alphaReValues --model_type LIFcomplex --data_folder /Users/dudchenko/local/GSC --dataset_name sc --s4 False --method grid --lr 0.001 --scheduler_patience 5 --batch_size 32 --nb_layers 3 --nb_hidden 512 --use_readout_layer True --pdrop 0.25 --seed 13 --alpha_imag 3.14 --alpha_range False --alpha_rand False --alpha 0.75 0.25 1 0.1 1.5
python3 ../run_exp.py --sweep_name alphaReValues --model_type LIFcomplex --data_folder /Users/dudchenko/local/GSC --dataset_name sc --s4 False --method grid --lr 0.001 --scheduler_patience 5 --batch_size 32 --nb_layers 3 --nb_hidden 512 --use_readout_layer True --pdrop 0.25 --seed 13 --alpha_imag 3.14 --alpha_range False --alpha_rand False Rand RandN --alphaRe_rand Rand RandN --alpha 0.5