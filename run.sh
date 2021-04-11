#!/bin/sh

# # Download and unzip data
# wget http://www.cs.cmu.edu/~pengchey/iwslt2014_ende.zip
# unzip iwslt2014_ende.zip

# # Prepare vocab
# python src/vocab.py \
#     --train-src=data/train.de-en.de.wmixerprep \
#     --train-tgt=data/train.de-en.en.wmixerprep \
#     data/vocab.json

vocab="data/vocab.json"
train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

work_dir="work_dir"

# mkdir -p ${work_dir}
# echo "save results to ${work_dir}"

# # training
# python src/main.py \
#     train \
#     --cuda \
#     --vocab ${vocab} \
#     --train-src ${train_src} \
#     --train-tgt ${train_tgt} \
#     --dev-src ${dev_src} \
#     --dev-tgt ${dev_tgt} \
#     --input-feed \
#     --valid-niter 2400 \
#     --batch-size 64 \
#     --hidden-size 256 \
#     --embed-size 256 \
#     --uniform-init 0.1 \
#     --label-smoothing 0.1 \
#     --dropout 0.2 \
#     --clip-grad 5.0 \
#     --save-to ${work_dir}/model.bin \
#     --lr-decay 0.5 

# # decoding
# python src/main.py \
#     decode \
#     --cuda \
#     --beam-size 10 \
#     --max-decoding-time-step 100 \
#     ${work_dir}/model.bin \
#     ${test_src} \
#     ${work_dir}/decode_bs10.txt

# perl ./src/multi-bleu.perl ${test_tgt} < ${work_dir}/decode_bs10.txt

# # compare gt sentence log prob and decoded sentence log prob
# python src/main.py \
#     compare \
#     --cuda \
#     ./work_dir/model.bin \
#     ./data/test.de-en.de.wmixerprep \
#     ./data/test.de-en.en.wmixerprep \
#     ./work_dir/decode_bs10.txt

# # opt-decoding
# python src/main.py \
#     opt-decode \
#     --cuda \
#     --opt-lr 0.01 \
#     --opt-step 1000 \
#     --ent-reg 5.0 \
#     --max-decoding-time-step 100 \
#     ${work_dir}/model.bin \
#     ${test_src} \
#     ${work_dir}/decode_opt.txt

# inspect
python src/main.py \
    opt-decode \
    --cuda \
    --chunk-size 2 \
    --opt-lr 0.5 \
    --opt-step 2000 \
    --ent-reg 3.0 \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ./data/test.de-en.de.wmixerprep \
    ./work_dir/decode_greedy.txt \
    ${work_dir}/decode_opt.txt