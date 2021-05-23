# meta-lr tuning
python3 main.py --trainer fomaml --dataset=sine --n_way=5 --k_spt=5 --k_qry=10  --inner_step_test=10  --meta_lr=1e-4 --task_num=10 --inner_step=5  --inner_lr=0.01 --epoch=1000