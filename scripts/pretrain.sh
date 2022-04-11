GPU=$1
ALPHA=$2
C=192
S=384
Z=192
LOSS=mse
dembed=384
depth=6
heads=6
mlpratio=4
scale=2
MODEL_PATH=checkpoint/entroformer_c${C}_s${S}_z${Z}_hyper_ar_pretrain/relativenum7_tokenmask0.5_topk32/${LOSS}${ALPHA}/
mkdir -p $MODEL_PATH
export CUDA_VISIBLE_DEVICES=$GPU
python main_trans_hyper_ar.py --alpha ${ALPHA} \
       --nEpochs 500 --lr 1e-4 \
       --distribution gauss \
       --na bidirectional \
       --scale ${scale} \
       --norm GDN \
       --num_parameter 2 \
       --channels ${C} \
       --last_channels ${S} \
       --hyper_channels ${Z} \
       --model_prefix $MODEL_PATH \
       --loss_type ${LOSS} \
       --batchSize 8 \
       --patchSize 256 \
       --dim_embed ${dembed} \
       --depth ${depth} \
       --head ${heads} \
       --mlp_ratio ${mlpratio} \
       --dropout 0. \
       --position_num 7 \
       --mask_ratio 0.5 \
       --attn_topk 32 \
       --grad_norm_clip 1.0 \
       --train_dir ./path-to-traindata/ \
       --test_dir ./path-to-testdata/ \