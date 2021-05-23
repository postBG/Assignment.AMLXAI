# inner step test tuning
python3 main.py --trainer fomaml --dataset=omniglot --n_way=5 --k_spt=5 --k_qry=10  --inner_step_test=50  --meta_lr=1e-3 --task_num=10 --inner_step=5  --inner_lr=0.4
python3 main.py --trainer fomaml --dataset=omniglot --n_way=5 --k_spt=5 --k_qry=10  --inner_step_test=50  --meta_lr=1e-3 --task_num=10 --inner_step=5  --inner_lr=0.2


