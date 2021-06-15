
CONFIG=./configs/train-examplar/finetune_rdn-liif-decoder.yaml
NAME=examplar_finetune_encoder
GPU_ID=0

IMG=0802.png
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG --gpu $GPU_ID

IMG=0803.png
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG --gpu $GPU_ID

IMG=0804.png
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG --gpu $GPU_ID

IMG=0809.png
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG --gpu $GPU_ID

IMG=0810.png
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG --gpu $GPU_ID

IMG=0822.png
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG --gpu $GPU_ID

IMG=0823.png
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG --gpu $GPU_ID

IMG=0826.png
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG --gpu $GPU_ID

IMG=0849.png
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG --gpu $GPU_ID

IMG=0855.png
python train_liif_examplar.py --config $CONFIG --name $NAME --tag $IMG --gpu $GPU_ID

