# task num tuning
python3 main.py --trainer reptile --dataset=omniglot --n_way=5 --k_spt=5 --k_qry=10  --inner_step_test=50  --meta_lr=0.5 --task_num=10 --inner_step=5  --inner_lr=1e-3
python3 main.py --trainer reptile --dataset=omniglot --n_way=5 --k_spt=5 --k_qry=10  --inner_step_test=50  --meta_lr=0.5 --task_num=20 --inner_step=5  --inner_lr=1e-3
python3 main.py --trainer reptile --dataset=omniglot --n_way=5 --k_spt=5 --k_qry=10  --inner_step_test=50  --meta_lr=0.5 --task_num=30 --inner_step=5  --inner_lr=1e-3
python3 main.py --trainer reptile --dataset=omniglot --n_way=5 --k_spt=5 --k_qry=10  --inner_step_test=50  --meta_lr=0.5 --task_num=40 --inner_step=5  --inner_lr=1e-3
