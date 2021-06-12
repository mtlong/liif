
GPU=0
MODEL_PATH=./download/rdn-liif.pth
INPUT_PATH=/mnt/HDD1/home/mtlong/workspace/2021/Aggregative_Learning/Sand_Box/liif/data/samples/4X
GT_PATH=/mnt/HDD1/home/mtlong/workspace/2021/Aggregative_Learning/Sand_Box/liif/data/samples/GT
RESULT_PATH=/mnt/HDD1/home/mtlong/workspace/2021/Aggregative_Learning/Sand_Box/liif/data/samples/Results
# MODE=SR_lowest_res
# MODE=SR_GT_res
MODE=SR_pred_res

python MTL_inspect_cross_scale.py --gpu $GPU --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 
