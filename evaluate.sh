export DATA=custom_data
export MODEL=saved_models/ft_custom/checkpoint_best.pt
export RESULT_PATH=results
export BSZ=16

$(which fairseq-generate) \
        --data-type STR --user-dir ./ --task text_recognition --input-size 384 \
        --beam 10 --scoring cer --gen-subset valid --batch-size ${BSZ} \
        --path ${MODEL} --results-path ${RESULT_PATH} --preprocess DA2 \
        --bpe gpt2 --dict-path-or-url "https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D" ${DATA}