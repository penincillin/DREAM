data_dir=data
if [ ! -d $data_dir ]; then
    mkdir $data_dir
fi

CUDA_VISIBLE_DEVICES=0,1 python eval_cfp.py \
    --img_dir ../../data/CFP/ \
