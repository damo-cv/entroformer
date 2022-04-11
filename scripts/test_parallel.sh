INPUT=$1
MODEL=$2
C=192
S=384
Z=192
dembed=384
depth=6
heads=6
mlpratio=4
scale=2
export CUDA_VISIBLE_DEVICES=0
python main_trans_hyper_ar.py --mode test \
       --distribution gauss \
       --na bidirectional \
       --scale ${scale} \
       --norm GDN \
       --num_parameter 2 \
       --channels ${C} \
       --last_channels ${S} \
       --hyper_channels ${Z} \
       --dim_embed ${dembed} \
       --depth ${depth} \
       --head ${heads} \
       --mlp_ratio ${mlpratio} \
       --dropout 0. \
       --position_num 7 \
       --attn_topk 32 \
       --test_dir ${INPUT} \
       --model_pretrained ${MODEL}