python3 -u GCN+FM.py --load_model ./ConMask_DEFAULT_0.ckpt --lr 0.01 --neg_num 10 --scale_1 10 --scale_2 10 --dim 200 --batch 512 --data ./data/OW_FB15K-237/ --save_per 5 --eval_per 1 --worker 16 --eval_batch 32 --max_iter 100 --generator 16