
CONFIG=./configs/train-examplar/train_LIFF_LPIPS-full-LPIPSNN.yaml
NAME=examplar_train_full_LPIPSNN
INPUT_PATH_PREFIX=./data/samples/4X
RESULT_PATH_PREFIX=./data/samples/Results/test_examplar_train_full_LPIPSNN
GT_PATH=./data/samples/GT
MODE=SR_lowest_res_indv
GPU_ID=0

# for IMG in 0802 0803 0804 0809 0810 0818 0820 0822 0823 0825 0826 0834 0835 0836 0837 0839 0846 0849 0855 0879 0883 0898
for IMG in 0825 0826 0835 0837 0846 0879 0883
do
    python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
    MODEL_PATH=./save/${NAME}_$IMG.png/epoch-last.pth
    INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
    RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

    python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 
done


# IMG=0802
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0803
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0804
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0809
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0810
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0818
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0820
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0822
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0823
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0825
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0826
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0834
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0835
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0836
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0837
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0839
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0846
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0849
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0855
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0879
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0883
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


# IMG=0898
# python train_liif_examplar_2DLoss.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
# MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
# INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
# RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

# python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 
