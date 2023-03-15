import os
from shutil import copyfile

# You only need to change this line to your dataset download path
#download_path = '/home/lxwang/MyProjects/first_test_reid/Market'
#download_path = '/home/lxwang/MyProjects/first_test_reid/DukeMTMC-reID'
#download_path = '/home/lxwang/MyProjects/first_test_reid/cuhk03'
download_path = '/home/nfs/nfsstorage_tmp/ai_research/public/ReID_Group/data/market1501'

if not os.path.isdir(download_path):            #判断是否存在这个目录
    print('please change the download_path')

save_path = os.path.join(download_path, 'pytorch')
if not os.path.isdir(save_path):
    os.mkdir(save_path)                       #创建这个目录
#-----------------------------------------
#query
query_path = os.path.join(download_path, 'query')
query_save_path = os.path.join(download_path, 'pytorch', 'query')
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for _, _, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = os.path.join(query_path, name)
        dst_path = os.path.join(query_save_path, ID[0])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, os.path.join(dst_path, name))

# #-----------------------------------------
# #multi-query
# query_path = download_path + '/gt_bbox'
# # for dukemtmc-reid, we do not need multi-query
# if os.path.isdir(query_path):
#     query_save_path = download_path + '/pytorch/multi-query'
#     if not os.path.isdir(query_save_path):
#         os.mkdir(query_save_path)
#
#     for root, dirs, files in os.walk(query_path, topdown=True):
#         for name in files:
#             if not name[-3:]=='jpg':
#                 continue
#             ID  = name.split('_')
#             src_path = query_path + '/' + name
#             dst_path = query_save_path + '/' + ID[0]
#             if not os.path.isdir(dst_path):
#                 os.mkdir(dst_path)
#             copyfile(src_path, dst_path + '/' + name)

#-----------------------------------------
#gallery
gallery_path = os.path.join(download_path, 'bounding_box_test')
gallery_save_path = os.path.join(download_path, 'pytorch', 'gallery')
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = os.path.join(gallery_path, name)
        dst_path = os.path.join(gallery_save_path, ID[0])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, os.path.join(dst_path, name))

#---------------------------------------
#train_all
train_path = os.path.join(download_path, 'bounding_box_train')
train_save_path = os.path.join(download_path, 'pytorch', 'train_all')
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = os.path.join(train_path, name)
        dst_path = os.path.join(train_save_path, ID[0])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, os.path.join(dst_path, name))


#---------------------------------------
#train_val
train_path = os.path.join(download_path, 'bounding_box_train')
train_save_path = os.path.join(download_path, 'pytorch', 'train')
val_save_path = os.path.join(download_path, 'pytorch', 'val')
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = os.path.join(train_path, name)
        dst_path = os.path.join(train_save_path, ID[0])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = os.path.join(val_save_path, ID[0])  #first image is used as val image
            os.mkdir(dst_path)
        copyfile(src_path, os.path.join(dst_path, name))
