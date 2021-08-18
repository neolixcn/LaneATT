#功能介绍
#image  ：检测一张图片
#camera  ：打开摄像头检测并实时显示检测结果
#path_save_avi ：检测目录下所有图片,保存为视频（按照公司图片命名的采集时间顺序）
#path_save_images : 检测目录下所有图片，保存为相同命名的图片
#txt_save_avi : 检测txt下所有图片，保存为相同命名的图片 切记
#txt_save_images : 检测txt下所有图片，保存为视频
import argparse
import numpy
import torch
import cv2
from lib.config import Config
import os

#python test.py mode -pt -path -cfg （功能模式 权重 读取路径 配置文件）
def parse_args():
    parser = argparse.ArgumentParser(description="Train lane detector")
    #Fix the description.
    parser.add_argument(
        "mode", choices=["camera", "image","path_save_images","path_save_avi","txt_save_avi","txt_save_images"], help="camera or image or images?")
    parser.add_argument("-pt", help=".pt", required=True)
    parser.add_argument("-path", help="image/video path", required=True)
    #Options should start with --.
    parser.add_argument("-cfg", help="Config file", required=True)
    args = parser.parse_args()
    return args


def process(model, image, img_w, img_h, cfg, device,type,img_path=[],video_writer=[]):
    image = cv2.resize(image, (img_w, img_h))
    #You should use the NoLabelDataset class so that you don't need to 
    #repeat the code used to draw predictions nor the steps required to
    #transform raw images to input tensors.
    data = torch.from_numpy(image.astype(numpy.float32)
                            ).permute(2, 0, 1) / 255.
    data = data.reshape(1, 3, img_h, img_w)
    images = data.to(device)
    test_parameters = cfg.get_test_parameters()
    output = model(images, **test_parameters)
    prediction = model.decode(output, as_lanes=True)
    #print(prediction)
    #print("\n")
    #print(prediction[0])
    for i, l in enumerate(prediction[0]):
        color = (0, 0, 255)
        points = l.points

        points[:, 0] *= image.shape[1]
        points[:, 1] *= image.shape[0]
        points = points.round().astype(int)
        # points += pad
        xs, ys = points[:, 0], points[:, 1]
        #print("线的横坐标：",xs)
        #print("线的纵坐标：",ys)
        length = numpy.sqrt((xs[0]-xs[-1])**2 + (ys[0]-ys[-1])**2)
        if(length < 50):
            continue
        for curr_p, next_p in zip(points[:-1], points[1:]):
            #画线——设置起点和终点，颜色，线条宽度
            #print("起点：",tuple(curr_p))
            #print("终点：",tuple(next_p))
            image = cv2.line(image,
                             tuple(curr_p),
                             tuple(next_p),
                             color=color,
                             thickness=3)
    if(type == "image"):
        cv2.imwrite("hhh.png",image)
        #cv2.imshow("", image)
        #cv2.waitKey(0)
    if(type == "camera"):
        cv2.imshow("", image)   

    if(type == "path_save_images"):
        cv2.imwrite("test_result/"+img_path,image)

    if(type == "path_save_avi"):
        video_writer.write(image)

    if(type == "txt_save_images"):
        cv2.imwrite("test_result/"+img_path,image)

    if(type == "txt_save_avi"):
        video_writer.write(image)


def main():
    args = parse_args()

    cfg_path = args.cfg
    cfg = Config(cfg_path)
    device = torch.device(
        'cpu') if not torch.cuda.is_available()else torch.device('cuda')

    model = cfg.get_model()
    #check_point = "./experiments/laneatt_"+"r"+cfg["model"]["parameters"]["backbone"][6:]+"_"+cfg["datasets"]["train"]["parameters"]["dataset"] +"/models/"
    check_point = args.pt
    #Use the get_checkpoint_path function in the Experiment class for that.

    #check_point+=os.listdir(check_point)[0]
    print("load check point:",check_point)
    dict = torch.load(check_point)
    model.load_state_dict(dict["model"])
    model = model.to(device)
    model.eval()
    img_h = cfg["model"]["parameters"]["img_h"]
    img_w = cfg["model"]["parameters"]["img_w"]
    if args.mode == "image":
        image = cv2.imread(args.path)
        process(model, image, img_w, img_h, cfg, device,args.mode)

    elif args.mode == "path_save_images":
        image_list = [img for img in os.listdir(args.path) if not os.path.isdir(img)]
        image_list = sorted(image_list, key=lambda x:float(x.split("_")[1][:-4]))
        for img_path in image_list:
            image = cv2.imread(os.path.join(args.path,img_path))
            process(model, image, img_w, img_h, cfg, device,args.mode,img_path)

    elif args.mode == "path_save_avi":
        size = (640,360)
        fps = 10
        fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')#cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter("test.avi", fourcc, fps, size)
        image_list = [img for img in os.listdir(args.path) if not os.path.isdir(img)]
        #公司图片是根据时序命名，所以需要排序读取，这样才能保存视频是按顺序
        image_list = sorted(image_list, key=lambda x:float(x.split("_")[1][:-4]))
        for img_path in image_list:
            image = cv2.imread(os.path.join(args.path,img_path))
            process(model, image, img_w, img_h, cfg, device,args.mode,img_path,video_writer)

    elif args.mode == "txt_save_avi":
        file=open(args.path,"r")     #以读模式打开文件\n",
        img_list=file.readlines()       #读取全部行\n",
        size = (640,360)
        fps = 10
        fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')#cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter("test.avi", fourcc, fps, size)
        i = 0
        for img_name in img_list:
            print(i)
            #这里很坑，要去掉换行符才能读取图片
            img_name = img_name.strip('\n')
            image = cv2.imread(img_name)
            process(model, image, img_w, img_h, cfg, device,args.mode,img_name.split("/")[-1],video_writer)
            i=i+1

    elif args.mode == "txt_save_images":
        file=open(args.path,"r")     #以读模式打开文件\n",
        img_list=file.readlines()       #读取全部行\n",
        i = 0
        for img_name in img_list:
            print(i)
            img_name = img_name.strip('\n')
            image = cv2.imread(img_name)
            process(model, image, img_w, img_h, cfg, device,args.mode,img_name.split("/")[-1])
            i=i+1

    elif args.mode == "camera":
        video = cv2.VideoCapture(args.path)
        while(True):
            rval, frame = video.read()
            process(model, frame, img_w, img_h, cfg, device,args.mode)
            if cv2.waitKey(1) == 27:
                break

if __name__ == '__main__':
    main()