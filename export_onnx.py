import torch
import argparse
import numpy as np
from lib.config import Config
from lib.models.laneatt_list_onnx import LaneATT
import os
import onnx
import onnxruntime
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description="export lane detector")
    parser.add_argument("-ckpt", help="specify the model path", required=True)
    parser.add_argument("-onnx_path", default= 'laneATT.onnx',help="specify the path for exported onnx", required=False)
    parser.add_argument("-cfg", help="Config file", required=True)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    cfg_path = args.cfg
    cfg = Config(cfg_path)

    # model_org = cfg.get_model()
    parameters = cfg['model']['parameters']
    model = LaneATT(**parameters)
    print("load check point:", args.ckpt)
    pretrained_dict = torch.load(args.ckpt)["model"]
    model.load_state_dict(pretrained_dict, strict=True)
    # model_org.load_state_dict(pretrained_dict, strict=True)

    # load state_dict with mismatched param ignored
    # model_dict = model.state_dict()
    # for k,v in model_dict.items():
    #     if v.shape != pretrained_dict[k].shape:
    #         print("size mismatch in", k)
    #         # model_dict[k] = change_param(pretrained_dict[k])
    #     else:
    #         model_dict[k] = pretrained_dict[k]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    # model_org = model.to(device)
    # model_org.eval()
    
    img_h = cfg["model"]["parameters"]["img_h"]
    img_w = cfg["model"]["parameters"]["img_w"]
    input_tensor = torch.randn(1, 3, img_h, img_w).to(device)

    # pdb.set_trace()
    out = model(input_tensor)
    # out2 = model_org(input_tensor)

    torch.onnx.export(model,
                                        input_tensor,
                                        args.onnx_path,
                                        export_params=True,
                                        opset_version=11,
                                        verbose=True,
                                        input_names=["input"],
                                        output_names=["reg_proposals", "attention_matrix"],
                                        dynamic_axes={"input":{0:"batch_size"}, "reg_proposals":{0:"batch_size"}, "attention_matrix":{0:"batch_size"}}
                                        )

    # check 没有问题
    # onnx_model = onnx.load(args.onnx_path)
    # onnx.checker.check_model(onnx_model)

    # onnxruntime 验证
    ort_session = onnxruntime.InferenceSession(args.onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.detach().cpu().numpy()}
    outputs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(out[0].detach().cpu().numpy(), outputs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(out[1].detach().cpu().numpy(), outputs[1], rtol=1e-03, atol=1e-05)
    print("Onnx model output looks good!")

