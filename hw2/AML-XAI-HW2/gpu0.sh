# meta-lr tuning
python3 main.py --trainer fomaml --dataset=sine --n_way=5 --k_spt=5 --k_qry=10  --inner_step_test=10  --meta_lr=1e-3 --task_num=1000 --inner_step=1  --inner_lr=0.01
python3 main.py --trainer maml --dataset=sine --n_way=5 --k_spt=5 --k_qry=10  --inner_step_test=10  --meta_lr=0.001 --task_num=1000 --inner_step=1  --inner_lr=0.01