
GPU=0
MODE=SR_lowest_res_indv
# MODE=SR_GT_res
# MODE=SR_pred_res
GT_PATH=/mnt/HDD1/home/mtlong/workspace/2021/Aggregative_Learning/Sand_Box/liif/data/samples/GT


MODEL_PATH=./save/examplar_finetune_0802.png/epoch-700.pth
INPUT_PATH=/mnt/HDD1/home/mtlong/workspace/2021/Aggregative_Learning/Sand_Box/liif/data/samples/4X/0802.png
RESULT_PATH=/mnt/HDD1/home/mtlong/workspace/2021/Aggregative_Learning/Sand_Box/liif/data/samples/Results/test_examplar_finetune_decoder/0802.png

python MTL_inspect_cross_scale.py --gpu $GPU --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


MODEL_PATH=./save/examplar_finetune_0803.png/epoch-700.pth
INPUT_PATH=/mnt/HDD1/home/mtlong/workspace/2021/Aggregative_Learning/Sand_Box/liif/data/samples/4X/0803.png
RESULT_PATH=/mnt/HDD1/home/mtlong/workspace/2021/Aggregative_Learning/Sand_Box/liif/data/samples/Results/test_examplar_finetune_decoder/0803.png

python MTL_inspect_cross_scale.py --gpu $GPU --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


MODEL_PATH=./save/examplar_finetune_0804.png/epoch-700.pth
INPUT_PATH=/mnt/HDD1/home/mtlong/workspace/2021/Aggregative_Learning/Sand_Box/liif/data/samples/4X/0804.png
RESULT_PATH=/mnt/HDD1/home/mtlong/workspace/2021/Aggregative_Learning/Sand_Box/liif/data/samples/Results/test_examplar_finetune_decoder/0804.png

python MTL_inspect_cross_scale.py --gpu $GPU --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# MODEL_PATH=./save/examplar_finetune_0802.png/epoch-700.pth
# INPUT_PATH=/mnt/HDD1/home/mtlong/workspace/2021/Aggregative_Learning/Sand_Box/liif/data/samples/4X/0802.png
# RESULT_PATH=/mnt/HDD1/home/mtlong/workspace/2021/Aggregative_Learning/Sand_Box/liif/data/samples/Results/test_examplar_finetune_decoder/0802.png

# python MTL_inspect_cross_scale.py --gpu $GPU --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 
