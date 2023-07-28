export MODEL_NAME=ft_name_insi
export SAVE_PATH=saved_models/${MODEL_NAME}
export LOG_DIR=log_${MODEL_NAME}
export DATA=/home/ubuntu/name_insi
mkdir ${LOG_DIR}
export BSZ=8
export valid_BSZ=16

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 $(which fairseq-train) --data-type STR --user-dir ./ --task text_recognition --input-size 384 --arch trocr_base --seed 1111 --optimizer adam --lr 2e-05 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-8 --warmup-updates 500 --weight-decay 0.0001 --log-format tqdm --log-interval 10 --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} --save-dir ${SAVE_PATH} --tensorboard-logdir ${LOG_DIR} --max-epoch 300 --patience 20 --ddp-backend legacy_ddp --num-workers 8 --preprocess DA2 --update-freq 1 --bpe gpt2 --decoder-pretrained roberta2 --finetune-from-model /home/ubuntu/trocr_bkp/pretrained_models/trocr-base-handwritten.pt  ${DATA}
