
CONFIG=./configs/train-examplar/finetune_rdn-liif-encoder_1NN_2.yaml
NAME=examplar_finetune
INPUT_PATH_PREFIX=./data/samples/4X
RESULT_PATH_PREFIX=./data/samples/Results/test_examplar_finetune_decoder
GT_PATH=./data/samples/GT
MODE=SR_lowest_res_indv

GPU_ID=1


IMG=0834
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0835
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0836
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0837
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0839
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0846
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0849
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0855
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0879
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0883
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0898
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 
