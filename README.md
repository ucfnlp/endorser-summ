# Getting Started
```
# Clone repo and download models
git clone https://github.com/ucfnlp/endorser-summ.git
cd endorser-summ
chmod +x download_pretrained_bart.sh
./download_pretrained_bart.sh
```

Download data and unzip: https://drive.google.com/uc?export=download&id=18kn_yzLIKmBV4rxMhXbuC-0BRocdsG_3

```
# Create environment and install packages
conda create -n endorse python=3.8
conda activate endorse
conda install pytorch=1.4 cudatoolkit=10.2 -c pytorch -y
yes | pip install transformers==2.0.0 bert_score spacy scikit-learn tqdm pyrouge nltk absl-py numpy pandas matplotlib Unidecode py-rouge
pip install --editable . -y
python -m spacy download en_core_web_sm
```

# For WCEP Dataset

## Run Endorsement
```
python endorse.py --dataset_name=duc_2004 --endorse_method=bertscore && python endorse.py --dataset_name=duc_2004 --endorse_method=bertscore --consolidation_method=sequential
```

## Preprocess
```
CUDA_VISIBLE_DEVICES=1 python preprocess_endorsements_for_training.py --dataset_name=wcep_mds --endorse_method=bertscore && CUDA_VISIBLE_DEVICES=1 python preprocess_endorsements_for_training.py --dataset_name=wcep_mds --endorse_method=bertscore --consolidation_method=sequential


for SPLIT in train val
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "endorse_data/wcep_mds_bertscore/$SPLIT.$LANG" \
    --outputs "endorse_data/wcep_mds_bertscore/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "endorse_data/wcep_mds_bertscore/train.bpe" \
  --validpref "endorse_data/wcep_mds_bertscore/val.bpe" \
  --destdir "endorse_data/wcep_mds_bertscore-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
for SPLIT in train val
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "endorse_data/wcep_mds_bertscore_sequential/$SPLIT.$LANG" \
    --outputs "endorse_data/wcep_mds_bertscore_sequential/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "endorse_data/wcep_mds_bertscore_sequential/train.bpe" \
  --validpref "endorse_data/wcep_mds_bertscore_sequential/val.bpe" \
  --destdir "endorse_data/wcep_mds_bertscore_sequential-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
```

## Train
```
# Train with Reciprocal setting
CUDA_VISIBLE_DEVICES=0 python train.py endorse_data/wcep_mds_bertscore-bin --endorse --restore-file ../bart.large/model.pt --max-tokens 1024 --max-source-positions 1024 --task translation --source-lang source --target-lang target --truncate-source --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch bart_large --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --optimizer adam --adam-eps 1e-08 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr 3e-05 --total-num-update 20000 --warmup-updates 500 --update-freq 4 --skip-invalid-size-inputs-valid-test --find-unused-parameters --num-workers 0 --max-sentences 1 --save-dir models/checkpoints_bertscore_dup_origscale0.8 --endorse-duplicate-heads --endorse-original-head-scale 0.8 --log-format simple --patience 2

# Train with Sequential setting
CUDA_VISIBLE_DEVICES=0 python train.py endorse_data/wcep_mds_bertscore_sequential-bin --endorse --restore-file ../bart.large/model.pt --max-tokens 1024 --max-source-positions 1024 --task translation --source-lang source --target-lang target --truncate-source --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch bart_large --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --optimizer adam --adam-eps 1e-08 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr 3e-05 --total-num-update 20000 --warmup-updates 500 --update-freq 4 --skip-invalid-size-inputs-valid-test --find-unused-parameters --num-workers 0 --max-sentences 1 --save-dir models/checkpoints_bertscore_sequential_dup_origscale0.8 --endorse-duplicate-heads --endorse-original-head-scale 0.8 --log-format simple --patience 2
```

## Test
```
# Test with Reciprocal setting
CUDA_VISIBLE_DEVICES=0 python bart_test.py --endorse_method=bertscore --endorse_duplicate_heads --endorse_original_head_scale=0.8 --dataset_name=wcep_mds

# Test with Sequential setting
CUDA_VISIBLE_DEVICES=0 python bart_test.py --endorse_method=bertscore --consolidation_method=sequential --endorse_duplicate_heads --endorse_original_head_scale=0.8 --dataset_name=wcep_mds
```

# For DUC-04 Dataset
Must have already trained a model on WCEP or use pretrained model
```
python endorse.py --dataset_name=duc_2004 --endorse_method=bertscore && python endorse.py --dataset_name=duc_2004 --endorse_method=bertscore --consolidation_method=sequential

CUDA_VISIBLE_DEVICES=1 python preprocess_endorsements_for_training.py --dataset_name=duc_2004 --endorse_method=bertscore && CUDA_VISIBLE_DEVICES=1 python preprocess_endorsements_for_training.py --dataset_name=duc_2004 --endorse_method=bertscore --consolidation_method=sequential

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
for SPLIT in test
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "endorse_data/duc_2004_bertscore/$SPLIT.$LANG" \
    --outputs "endorse_data/duc_2004_bertscore/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --validpref "endorse_data/duc_2004_bertscore/test.bpe" \
  --destdir "endorse_data/duc_2004_bertscore-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
for SPLIT in test
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "endorse_data/duc_2004_bertscore_sequential/$SPLIT.$LANG" \
    --outputs "endorse_data/duc_2004_bertscore_sequential/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --validpref "endorse_data/duc_2004_bertscore_sequential/test.bpe" \
  --destdir "endorse_data/duc_2004_bertscore_sequential-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;

CUDA_VISIBLE_DEVICES=1 python bart_test.py --endorse_method=bertscore --endorse_duplicate_heads --endorse_original_head_scale=0.8 --dataset_name=duc_2004 && CUDA_VISIBLE_DEVICES=1 python bart_test.py --endorse_method=bertscore --consolidation_method=sequential --endorse_duplicate_heads --endorse_original_head_scale=0.8 --dataset_name=duc_2004
```

# For TAC-11 Dataset
Must have already trained a model on WCEP or use pretrained model
```
python endorse.py --dataset_name=tac_2011 --endorse_method=bertscore && python endorse.py --dataset_name=tac_2011 --endorse_method=bertscore --consolidation_method=sequential

CUDA_VISIBLE_DEVICES=1 python preprocess_endorsements_for_training.py --dataset_name=tac_2011 --endorse_method=bertscore && CUDA_VISIBLE_DEVICES=1 python preprocess_endorsements_for_training.py --dataset_name=tac_2011 --endorse_method=bertscore --consolidation_method=sequential

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
for SPLIT in test
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "endorse_data/tac_2011_bertscore/$SPLIT.$LANG" \
    --outputs "endorse_data/tac_2011_bertscore/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --validpref "endorse_data/tac_2011_bertscore/test.bpe" \
  --destdir "endorse_data/tac_2011_bertscore-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
for SPLIT in test
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "endorse_data/tac_2011_bertscore_sequential/$SPLIT.$LANG" \
    --outputs "endorse_data/tac_2011_bertscore_sequential/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --validpref "endorse_data/tac_2011_bertscore_sequential/test.bpe" \
  --destdir "endorse_data/tac_2011_bertscore_sequential-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;

CUDA_VISIBLE_DEVICES=1 python bart_test.py --endorse_method=bertscore --endorse_duplicate_heads --endorse_original_head_scale=0.8 --dataset_name=tac_2011 && CUDA_VISIBLE_DEVICES=1 python bart_test.py --endorse_method=bertscore --consolidation_method=sequential --endorse_duplicate_heads --endorse_original_head_scale=0.8 --dataset_name=tac_2011
```