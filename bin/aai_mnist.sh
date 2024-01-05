cuda=0
src_dataset=AAIMNIST_02468_0123456798_10
tar_dataset=AAIMNIST_13579_0123456798_10

python src/main.py \
    --cuda ${cuda} \
    --src_dataset ${src_dataset} \
    --tar_dataset ${tar_dataset} \
    --transfer_ebd \
    --lr 0.001 \
    --weight_decay 0.01 \
    --patience 5
