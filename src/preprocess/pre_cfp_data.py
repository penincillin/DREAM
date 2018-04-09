import os, sys, shutil

if __name__ == '__main__':

    assert(len(sys.argv)==4)

    dst_dir = sys.argv[1]  # ../data/cfp
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    celeb_name_file = sys.argv[2] # /ssd/data/cfp/cfp_dataset/Data/list_name.txt
    with open(celeb_name_file, 'r') as in_f:
        name_mapping = {idx+1:'_'.join(line.strip().split()) \
                for idx, line in enumerate(in_f)}
    
    dst_data_dir = os.path.join(dst_dir, 'data')
    if not os.path.exists(dst_data_dir):
        os.mkdir(dst_data_dir)


    align_img_list = sys.argv[3]
    with open(align_img_list, 'w') as out_f:

        celeb_num = len(name_mapping)
        for i in range(celeb_num):

            celeb_name = name_mapping[i+1]

            sub_dir = os.path.join(dst_data_dir, celeb_name)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)

            frontal_sub_dir = os.path.join(sub_dir, 'frontal')
            profile_sub_dir = os.path.join(sub_dir, 'profile')
            if not os.path.exists(frontal_sub_dir):
                os.mkdir(frontal_sub_dir)
            if not os.path.exists(profile_sub_dir):
                os.mkdir(profile_sub_dir)

            frontal_num, profile_num = 10, 4
            infos = [(frontal_sub_dir, frontal_num), (profile_sub_dir, profile_num)]
            for sub_dir_name, img_num in infos:
                for im_n in range(1, img_num+1):
                    img_id = '0'+str(im_n) if len(str(im_n))<2 else str(im_n)
                    img_name = img_id + '.jpg'
                    out_f.write(os.path.join(sub_dir_name, img_name)+'\n')
