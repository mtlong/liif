
CONFIG=./configs/train-div2k/train_rdn-liif-2DLPIPS.yaml
NAME=Loss_2DLPIPS
GPU_ID=0
VER=01_sampleq_2304

python train_liif_2DLoss.py --config $CONFIG --name $NAME --tag $VER --gpu $GPU_ID

## Generate test result
MODEL_PATH=./save/Loss_2DLPIPS_$VER/epoch-last.pth
INPUT_PATH=./data/samples/4X
RESULT_PATH=./data/samples/Results/test_2DLossLPIPS
GT_PATH=./data/samples/GT
MODE=SR_lowest_res

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 
