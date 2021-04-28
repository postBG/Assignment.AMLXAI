python3 main.py --trainer vanilla --dataset CIFAR100 --nepochs 60 --lr 0.001 --device_idx 0
python3 main.py --trainer ewc --dataset CIFAR100 --nepochs 60 --lr 0.001 --device_idx 0 --lamb 0.1
python3 main.py --trainer l2 --dataset CIFAR100 --nepochs 60 --lr 0.001 --device_idx 0 --lamb 0.1

