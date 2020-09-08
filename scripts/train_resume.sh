#./train.py --segmentation 1 --batchsize 32 --gpu 1 --detection 1 --multitask_loss 1 --attention 1 --resume result/final_voc/20190224_205959/snapshot_iter_80000 
#python train.py --segmentation 1 --batchsize 32 --gpu 1 --detection 1 --multitask_loss 1 --attention 1 --resume result/final_voc/20190224_205959/snapshot_iter_120000 --snap_step=125000 130000 --eval_step 125000 130000

python train.py --segmentation  --batchsize 32 --gpu 1 --detection  --multitask_loss  --attention  --resume result/final_voc/20190224_205959/snapshot_iter_120000 --snap_step 125000 130000 --eval_step 125000 130000

