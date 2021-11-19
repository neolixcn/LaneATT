# 测试横向误差指标
import argparse
import numpy as np
import torch
import cv2
from lib.config import Config
import os
from scipy import integrate
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train lane detector")
    parser.add_argument(
        "--mode", choices=["torch", "fp32", "fp16", "int8"], help="specity the mode for converting trt engine")
    parser.add_argument("--onnx", help="specify the path for onnx model", required=True)
    parser.add_argument("-pt", help=" pytorch pth path", required=True)
    parser.add_argument("--cfg", help="Config file", required=True)
    args = parser.parse_args()
    return args

def calc_k(line): 
    '''
    Calculate the direction of lanes
    in: line，list type
    out: rad between line and positive direction of x-axis
    '''
    line_x = [point[0] for point in line]
    line_y = [point[1] for point in line]
    length = np.sqrt((line_x[0]-line_x[-1])**2 + (line_y[0]-line_y[-1])**2)
    if length < 90:
        return -10

    p = np.polyfit(line_x, line_y,deg = 1)
    rad = np.arctan(p[0])
    
    return rad

def convert_2_camera(points, scale=(1,1), K=None, camera_height=1.456):
    # remap to org_img
    # import pdb; pdb.set_trace()
    points = np.array(scale) * points
    if not K:
        K = np.array([[1000, 0,   960],
                                    [0,   1000, 540],
                                    [0,        0,        1]])
    K_inv = np.linalg.inv(K)
    camera_points = []
    for point in points:
        norm_camera_point = np.dot(K_inv, np.concatenate((point, np.array([1]))))
        ratio = camera_height / norm_camera_point[1]
        camera_point = norm_camera_point * ratio
        # import pdb; pdb.set_trace()
        camera_points.append(camera_point[::2])
    
    return np.array(camera_points)

def abs_poly_integration(coeff, start, end):
    roots = sorted([root for root in np.roots(coeff) if root > start and root < end and not isinstance(root, complex)])
    area = 0
    func = lambda x,a,b,c,d: a*x**3+b*x**2+c*x+d
    # func = lambda x,a,b,c,d: abs(a*x**3+b*x**2+c*x+d)
    roots.append(end)
    for i,root in enumerate(roots):
        if i == 0:
            start_x = start
            end_x = root
        else:
            start_x = roots[i-1]
            end_x = root
        area += abs(integrate.quad(func, start_x, end_x, args=tuple(coeff))[0])
    return area

def calculate_error(prediction, label, img_size, org_img_size, min_dist=0, max_dist=20):
    # 1. decode ego_left and ego_right
    # img_width = img_size[0]
    img_height = img_size[1]
    lane_ys = torch.linspace(1, 0, 72, dtype=torch.float32, device=torch.device(
        'cpu')) * img_height
    prediction[:, 4] = torch.round(prediction[:, 4])
    ego_left_rad = 0
    ego_right_rad = 0
    point_dict={}
    for i,lane in enumerate(prediction):
        lane_xs = lane[5:]
        start = int(round(lane[2].item() * 71))
        length = int(round(lane[4].item()))
        end = start + length - 1
        end = min(end, len(lane_ys) - 1)
        lane_points = convert_2_camera(torch.stack((lane_xs[start:end+1], lane_ys[start:end+1]), dim=1).cpu().numpy(), (org_img_size[0]/img_size[0], org_img_size[1]/img_size[1]))
        rad = calc_k(lane_points)
        if rad < ego_left_rad:
            ego_left_rad = rad
            point_dict["left"] = (i, lane_points)
        elif rad > ego_right_rad:
            ego_right_rad = rad
            point_dict["right"] = (i, lane_points)
        else:
            continue

    error_dict = {"left":0,"right":0}
    for key,val in point_dict.items():
        idx, point_set1 = val
        # label_xs = label[idx]
        start = int(round(label_xs[2].item() * 71))
        length = int(round(label_xs[4].item()))
        end = start + length - 1
        point_set2 = convert_2_camera(torch.stack((label_xs[5+start:5+end+1], lane_ys[start:end+1]), dim=1).cpu().numpy(), (org_img_size[0]/img_size[0], org_img_size[1]/img_size[1]))
        # 2. calculate
        # if one set has no point, return 0 error.
        if not(point_set1.shape[0] and point_set2.shape[0]):
            print("one set has no point!")
            return 0
        # import pdb; pdb.set_trace()
        print(point_set1.shape, " ", point_set2.shape)
        point_set1 = point_set1.tolist()
        point_set2 = point_set2.tolist()
        point_set1.sort(key=lambda x:x[1])
        point_set2.sort(key=lambda x:x[1])
        point_set1 = list(filter(lambda x: x[1]>min_dist and x[1] <max_dist, point_set1))
        point_set2 = list(filter(lambda x: x[1]>min_dist and x[1] <max_dist, point_set2))
        # import pdb; pdb.set_trace()
        coeff1 = np.polyfit(np.array(point_set1)[:,0], np.array(point_set1)[:,1], 3)
        coeff2 = np.polyfit(np.array(point_set2)[:,0], np.array(point_set2)[:,1], 3)

        diff_coeff = coeff1 - coeff2
        start = max(min_dist, point_set1[0][1], point_set2[0][1])
        end = min(max_dist, point_set1[-1][1], point_set2[-1][1])
        error_dict[key] = abs_poly_integration(diff_coeff, start, end) / (end - start)

    return error_dict.values()

def main():
    args = parse_args()

    cfg_path = args.cfg
    cfg = Config(cfg_path)
    device = torch.device(
        'cpu') if not torch.cuda.is_available()else torch.device('cuda')

    # model
    model = cfg.get_model()
    check_point = args.pt
    print("load check point:",check_point)
    dict = torch.load(check_point)
    model.load_state_dict(dict["model"])
    model = model.to(device)
    model.eval()

    # dataloader
    test_dataset = cfg.get_dataset('test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=8)
    print("test dataset size:", len(test_dataset))
    img_h = cfg["model"]["parameters"]["img_h"]
    org_img_w, org_img_h = 1920, 1080
    img_w = cfg["model"]["parameters"]["img_w"]
    test_parameters = cfg.get_test_parameters()
    left_error_sum = 0
    right_error_sum = 0
    with torch.no_grad():
        for idx, (images, labels,_) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            # labels = labels.to(device)
            output = model(images, **test_parameters)
            prediction = model.decode(output, as_lanes=False)
            # import pdb; pdb.set_trace()
            left_error, right_error = calculate_error(prediction[0].cpu(), labels[0], (img_w, img_h), (org_img_w, org_img_h))
            left_error_sum += left_error
            right_error_sum += right_error
    print(f"average left error {left_error_sum / len(test_dataset)}, average right error {right_error_sum / len(test_dataset)}, \
                average lane error {(left_error_sum + right_error_sum) / (2 * len(test_dataset))}")

if __name__ == '__main__':
    """
    python test_metric.py  --mode fp32 --onnx ./laneATT_noatt.onnx -pt ./exps/neolix_undistort/models/model_0093.pt --cfg ./cfgs/neolix_undistort.yml
    """
    main()