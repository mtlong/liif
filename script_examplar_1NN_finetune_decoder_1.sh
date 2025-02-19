
CONFIG=./configs/train-examplar/finetune_rdn-liif-encoder_1NN_1.yaml
NAME=examplar_finetune
INPUT_PATH_PREFIX=./data/samples/4X
RESULT_PATH_PREFIX=./data/samples/Results/test_examplar_finetune_decoder
GT_PATH=./data/samples/GT
MODE=SR_lowest_res_indv

GPU_ID=0


IMG=0802
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0803
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0804
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0809
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0810
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0818
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0820
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0822
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0823
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0825
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


IMG=0826
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG.png --gpu $GPU_ID
MODEL_PATH=./save/examplar_finetune_$IMG.png/epoch-last.pth
INPUT_PATH=$INPUT_PATH_PREFIX/$IMG.png
RESULT_PATH=$RESULT_PATH_PREFIX/$IMG.png

python MTL_inspect_cross_scale.py --gpu $GPU_ID --model $MODEL_PATH --input_path $INPUT_PATH --gt_path $GT_PATH --result_path $RESULT_PATH --mode $MODE 


