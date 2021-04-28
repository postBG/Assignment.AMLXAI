python3 main.py --trainer vanilla --dataset CIFAR100 --nepochs 60 --lr 0.001 --device_idx 1 --seed=2
python3 main.py --trainer l2 --dataset CIFAR100 --nepochs 60 --lr 0.001 --device_idx 1 --lamb 2.0 --seed=2
python3 main.py --trainer ewc --dataset CIFAR100 --nepochs 60 --lr 0.001 --device_idx 1 --lamb 2.0 --seed=2