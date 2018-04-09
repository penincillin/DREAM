curr_date=$(date +'%m_%d_%H_%M') 
mkdir log
log_file="./log/$curr_date.log"
CUDA_VISIBLE_DEVICES=0 python branch_train.py 2>&1 | tee $log_file
