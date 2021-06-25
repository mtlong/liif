
CONFIG=./configs/train-div2k/train_rdn-liif-2DLoss.yaml
NAME=Loss_2DL1
GPU_ID=1
VER=01_sampleq_2304

python train_liif_2DLoss.py --config $CONFIG --name $NAME --tag $VER --gpu $GPU_ID

## Generate test result
MODEL_PATH=./save/Loss_2DL1_$VER/epoch-last.pth
INPUT_PATH=./data/samples/4X
RESULT_PATH=./data/samples/Results/test_2DLossL1
GT_PATH=./data/samples/GT
MODE=SR_lowest_res

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 
