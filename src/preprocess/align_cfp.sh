# prepare CFP data first
dst_dir=../../data/CFP
list_file_name=/mnt/SSD/rongyu/data/cfp/cfp_dataset/Data/list_name.txt
dst_file_name=align_img_list.txt
python pre_cfp_data.py $dst_dir $list_file_name $dst_file_name


# align CFP image
pnp_file=pnp.txt
image_prefix=/mnt/SSD/rongyu/data/cfp/cfp_dataset/
alignment_file=cfp_alignment.txt
aligned_img_file=align_img_list.txt
pose_file=estimate_pose.txt

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
./test_process_align $pnp_file $image_prefix $alignment_file $aligned_img_file $pose_file
