import os
import cv2
import tqdm
import numpy as np
import pdb
import json, argparse

txt_path="/nfs/neolix_data1/neolix_dataset/test_dataset/lane_detection/neolix_lane_3mm/百度标注结果_lane_test_3mm.txt"
json_path="/nfs/neolix_data1/neolix_dataset/test_dataset/lane_detection/neolix_lane_3mm/test_label_85.json"
h_samples=[370,380,390,400,410,420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710,720,730,740,750,760,770,780,790,800,810,820,830,840,850,860,870,880,890,900,910,920,930,940,950,960,970,980,990,1000,1010,1020,1030,1040,1050,1060,1070]
shuaixuan_lane=5

def get_neolix_list(root, label_file):
    '''
    Get all the files' names and line points from the json annotation
    '''
    label_path = os.path.join(root, label_file)
    with open(label_path, "r") as f:
        label_content = f.readlines()
    names = []
    labels = []
    for line in label_content:
        if "url" in line:
            continue
        content = line.split()
        raw_file, img_name, str_result = content
#         print(raw_file)
        raw_file=raw_file.split('/')
        raw_file="images/"+raw_file[-2]+"/"+raw_file[-1]
        # print(raw_file)
        result = json.loads(str_result)
#         print(result)
        names.append(raw_file)
        label = []
        # pdb.set_trace()
        for lane in result["result"][0]["elements"]: 
            # 因为双黄线和混合线标注时会有两个分的一个总的，这里过滤掉两个分的
            if ("sub_ID" in lane["attribute"]) and (lane["attribute"][ "sub_ID"] != "null"):
                #print("have sub_ID")
                continue
            # 本次实验要加路延边，所以不过滤
            #if "single_line" in lane["attribute"] and lane["attribute"]["single_line"] == "road_edge":
            #    continue
            #label.append(sorted(lane["points"], key=lambda x:x["y"]))
            label.append(lane["points"])
        labels.append(label)

    return names,labels

names,line_txts = get_neolix_list("", txt_path)

def cal_x(point1,point2,Y):  #根据Y求X
    # X=(point2["x"]-point1["x"])/(point2["y"]+0.01-point1["y"])*Y+point1["x"]-(point2["x"]-point1["x"])/(point2["y"]-point1["y"]+0.01)*point1["y"]
    p=np.polyfit([point1["y"],point2["y"]],[point1["x"],point2["x"]],deg=1)
    # print(point1,point2)
    fx=np.poly1d(p)
    X=fx(Y)
    return X

lane=[-2 for i in range(len(h_samples))]

for num in range(len(names)):
    img_lane = []
    line_num=0
    numofnottwo=0
    for i in line_txts[num]:
        i=i[::-1]  #从小到大排序
        line_num+=1
        if line_num>8:
            break
        lane=[-2 for j in range(len(h_samples))]

        min_y = i[0]["y"]
        max_y = i[-1]["y"]

        for idx,point in enumerate(h_samples):
            if point < min_y or point > max_y:  #超出实际范围的不予考虑
                continue
            for idp,p in enumerate(i):
                if point<p["y"] and i[idp-1] is not None:
                    if p==i[idp-1]:
                        continue
                    X=cal_x(p,i[idp-1],h_samples[idx])  #计算X坐标
                    if X>0 and X<1920:   #在图片范围内的赋值
                        lane[idx]=int(X)
                        break
        for pp in lane:   #过滤掉点过少的线
            if pp!=-2:
                numofnottwo+=1
            if numofnottwo>shuaixuan_lane:
                img_lane.append(lane)
                break
        numofnottwo=0

    img_lane_new=[]
    info=[]
    cishu=0
    if len(img_lane)>shuaixuan_lane:#筛出当前行驶的五车道线
        for idx,lane in enumerate(img_lane):
            for p in lane:
                if p!=-2:
                    dis=abs(p-960)
                    one_info=[idx,dis]
                    info.append(one_info)
                    break
        info_new=sorted(info,key=lambda x:x[1])
        while(len(info_new)>shuaixuan_lane):
            info_new.pop(-1)
        for i in range(len(info_new)):
            index=info_new[i][0]
            img_lane_new.append(img_lane[index])
    else:
        img_lane_new=img_lane

    one_dict={
        "lanes":img_lane_new,
        "h_samples":h_samples,
        "raw_file":names[num]
    }
    # print(names[num])
    with open(json_path,"a") as f:
        json.dump(one_dict,f)
        f.write("\n")