# fomaml + inner lr tuning
python3 main.py --trainer reptile --dataset=omniglot --n_way=5 --k_spt=1 --k_qry=10  --inner_step_test=50  --meta_lr=1e-3 --task_num=10 --inner_step=20  --inner_lr=0.01
python3 main.py --trainer fomaml --dataset=omniglot --n_way=5 --k_spt=1 --k_qry=10  --inner_step_test=50  --meta_lr=1e-3 --task_num=10 --inner_step=5  --inner_lr=0.4
python3 main.py --trainer fomaml --dataset=omniglot --n_way=5 --k_spt=1 --k_qry=10  --inner_step_test=50  --meta_lr=1e-3 --task_num=30 --inner_step=5  --inner_lr=0.4

