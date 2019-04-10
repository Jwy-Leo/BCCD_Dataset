import os
import xml.etree.cElementTree as ET
import numpy as np
from matplotlib.pyplot import imread, imsave
import matplotlib.pyplot as plt
from scipy.misc import imresize 
image_folder_path = "/home/odin/test_process/BCCD_Dataset/BCCD/JPEGImages/"
annotation_path = "/home/odin/test_process/BCCD_Dataset/BCCD/Annotations"
setting_folder_path = "/home/odin/test_process/BCCD_Dataset/BCCD/ImageSets/Main/"
setting_file = ['train.txt', 'test.txt', 'trainval.txt', 'val.txt']
setting_file_name = {}
for setting in setting_file:
    with open(os.path.join(setting_folder_path,setting),'r') as F:
        setting = "".join(str.split(setting,".")[:-1])
        filenames = F.readlines()
        filenames = [name[:-1] for name in filenames ]
        setting_file_name[setting] = filenames
for name in setting_file_name.keys():
    print(name,len(setting_file_name[name]))
main_data_path = "data"
mode_usage = ["feature_loader", "npy_loader","all"] 
mode = 2 #"feature_loader"# npy_loader
for name in setting_file_name.keys():
    npy_data_array = []
    npy_meta = {0:"R", 1:"W", 2:"P"}
    if not os.path.exists(os.path.join(main_data_path,name)):
        os.makedirs(os.path.join(main_data_path,name))
    for path in setting_file_name[name]:
        # deal with annotation
        anno_path = os.path.join(annotation_path,path) +".xml"
        data_tree = ET.ElementTree(file = anno_path)
        bbox_items_list = []
        for elem in data_tree.iter():
            if 'object' in elem.tag or 'part' in elem.tag:
                for attr in list(elem):
                    if 'name' in attr.tag:
                        filename = attr.text
                    if 'bndbox' in attr.tag:  
                        bbox_info = {}
                        bbox_attr = ['feature_name','xmin','ymin','xmax','ymax']                      
                        for item in list(attr):
                            bbox_info[item.tag] = int(round(float(item.text)))
                        bbox_info['feature_name'] = filename[0]
                        bbox_items_list.append(bbox_info)
        # deal with row data 
        img_path = os.path.join(image_folder_path,path) +".jpg"
        feature_count = np.zeros(3)
        for item in bbox_items_list:
            new_image_folder_path = os.path.join(main_data_path,name,item['feature_name'])
            if mode!=1:
                if not os.path.exists(new_image_folder_path):
                    os.makedirs(new_image_folder_path)
            img = imread(img_path)
            xmin,xmax,ymin,ymax = item['xmin'],item['xmax'],item['ymin'],item['ymax']
            noise_b_ratio = 0.1
            noise_range = [int(float(xmax-xmin)*noise_b_ratio),int(float(ymax - ymin) * noise_b_ratio)]
            _func = lambda noise_range: np.random.choice(noise_range) if np.random.choice(2) else -1 * np.random.choice(noise_range) 
            if noise_range[0]!=0 and noise_range[1]!=0:
                random_noise = [ _func(noise_range[i%2]) for i in range(4)] 
            elif noise_range[0]==0 and noise_range[1]!=0:
                random_noise = []
                random_noise.append(0)
                random_noise.append(_func(noise_range[1]))
                random_noise.append(0)
                random_noise.append(_func(noise_range[1]))
            elif noise_range[0]!=0 and noise_range[1]==0:
                random_noise = []
                random_noise.append(_func(noise_range[0]))
                random_noise.append(0)
                random_noise.append(_func(noise_range[0]))
                random_noise.append(0)
            else:
                random_noise = [ 0 for i in range(4)] 
            xmin,ymin,xmax,ymax = min(max(0,xmin+random_noise[0]),img.shape[1]), min(max(0,ymin+random_noise[1]),img.shape[0]), max(0,min(img.shape[1],xmax+random_noise[2])), max(0,min(img.shape[0],ymax+random_noise[3]))
            print("image_size : ({0:<3},{1:<3}),\t({2:<3},{3:<3},{4:<3},{5:<3})".format(ymax-ymin,xmax-xmin,xmin,xmax,ymin,ymax))
            if ymin==ymax or xmin==xmax:
                break
            crop_img = img[ymin:ymax,xmin:xmax,:]
            reshape_img = imresize(crop_img,(64,64))
            if mode!=1:
                new_image_folder_path = os.path.join(main_data_path,name,item['feature_name'])
                if item["feature_name"]=="R":
                    image_output_path = os.path.join(new_image_folder_path,path) + "_{}_{}.jpg".format(item['feature_name'],int(feature_count[0]))
                    feature_count[0] +=1
                    label = 0
                if item["feature_name"]=="W":
                    image_output_path = os.path.join(new_image_folder_path,path) + "_{}_{}.jpg".format(item['feature_name'],int(feature_count[1]))
                    feature_count[1] +=1
                    label = 1
                if item["feature_name"]=="P":
                    image_output_path = os.path.join(new_image_folder_path,path) + "_{}_{}.jpg".format(item['feature_name'],int(feature_count[2]))
                    feature_count[2] += 1
                    label = 2
                print(image_output_path)
                imsave(image_output_path,reshape_img)
            if item["feature_name"]=="R":
                label = 0
            if item["feature_name"]=="W":
                label = 1
            if item["feature_name"]=="P":
                label = 2
            npy_data_array.append(np.hstack([reshape_img.reshape(-1),np.array([label])]))
    if mode != 0:
        data_dict = {}
        npy_data_array = np.vstack(npy_data_array)
        data_dict['data'] = npy_data_array
        data_dict["label_name"] = npy_meta
        if not os.path.exists(os.path.join(main_data_path,"npy_dataset")):
            os.makedirs(os.path.join(main_data_path,"npy_dataset"))
        npy_path = os.path.join(main_data_path,"npy_dataset", name+"_{}".format(npy_data_array.shape[0]))
        np.save(npy_path,data_dict)
        '''
        plt.subplot(1,2,1)
        plt.imshow(crop_img)
        plt.title(item["feature_name"])
        plt.subplot(1,2,2)
        plt.imshow(reshape_img)
        plt.pause(0.1)
        '''
            
#os.makedirs("data")
