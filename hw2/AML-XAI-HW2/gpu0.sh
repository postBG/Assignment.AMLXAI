python3 main.py --trainer maml --dataset=sine --n_way=5 --k_spt=5 --k_qry=10  --inner_step_test=10  --meta_lr=1e-3 --task_num=1000 --inner_step=1  --inner_lr=1e-4 --epoch=10000
python3 main.py --trainer maml --dataset=sine --n_way=5 --k_spt=5 --k_qry=10  --inner_step_test=10  --meta_lr=1e-4 --task_num=1000 --inner_step=1  --inner_lr=1e-4 --epoch=3000
python3 main.py --trainer maml --dataset=sine --n_way=5 --k_spt=5 --k_qry=10  --inner_step_test=10  --meta_lr=1e-4 --task_num=1000 --inner_step=1  --inner_lr=1e-3 --epoch=3000
python3 main.py --trainer reptile --dataset=sine --n_way=5 --k_spt=5 --k_qry=10  --inner_step_test=10  --meta_lr=1e-3 --task_num=10 --inner_step=5  --inner_lr=0.001 --epoch=5000
