
CONFIG=./configs/train-examplar/train_LIFF_LPIPS-full-LPIPSNN_2.yaml
NAME=examplar_train_full_LPIPSNNInternal
INPUT_PATH_PREFIX=./data/samples/4X
RESULT_PATH_PREFIX=./data/samples/Results/test_examplar_train_full_LPIPSNNInternal
GT_PATH=./data/samples/GT
MODE=SR_lowest_res_indv
GPU_ID=0

# for IMG in 0802 0803 0804 0809 0810 0818 0820 0822 0823 0825 0826 0834 0835 0836 0837 0839 0846 0849 0855 0879 0883 0898
for IMG in 0822 0823 0825 0826 0834 0835 0836
do
    python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
    MODEL_PATH=./save/${NAME}_$IMG.png/epoch-best.pth
    INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
    RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

    python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 
done

