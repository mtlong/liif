
GPU=1
# MODE=SR_lowest_res_indv
# MODE=SR_GT_res
MODE=SR_pred_res
GT_PATH=./data/samples/GT


MODEL_PATH=./save/Loss_2DLPIPS_01_sampleq_2304/epoch-last.pth
INPUT_PATH=./data/samples/4X
RESULT_PATH=./data/samples/Results/test_2DLossLPIPS

python MTL_inspect_cross_scale.py --gpu $GPU --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


