import torch
import numpy as np
from open3dsg.models.SGRec3D import SGRec3D
import json
from open3dsg.scripts.sgrec_eval import SGRecEvaluator
import open3d as o3d
# load a .pth file
# data = torch.load("sgrec_output/v14/output_155000/training_batch_155000.pth", weights_only=False)
data = torch.load("sgrec_output/v16/output_85000/training_batch_85000.pth", weights_only=False)
print(data.keys())
config_path = "/home/ubuntu/SceneGraphEdit/open3dsg/config/config_v14.json"
config = json.load(open(config_path, "r"))
# load the model
model = SGRec3D(hparams=config['hparams'])
evaluator = SGRecEvaluator(config=config,
                model=model,
                dataset_path=config['data_dir'],
                device='cuda:1',
                re_angle=config['re_angle'] if 're_angle' in config else False,
                swap_yz=config['swap_yz'] if 'swap_yz' in config else False)
colors = np.random.rand(10, 3)

data_first_batch = {}
for k, v in data.items():
    if k == "objects_dec":
        # print(k, v.shape)
        data_first_batch[k] = v[:10, ...].detach()
        # print(k, data_first_batch[k].shape)
    elif isinstance(v, torch.Tensor):
        # print(k, v.shape)
        data_first_batch[k] = v[0, ...].detach().unsqueeze(0)
        # print(k, data_first_batch[k].shape)
    else:
        data_first_batch[k] = v
        
pred, _ = evaluator.reconstruct_pred(data_first_batch, pt_colors=colors)
pred_ply_file_path = "pred.ply"
o3d.io.write_point_cloud(pred_ply_file_path, pred)

gt = evaluator.reconstruct_gt(data_first_batch, pt_colors=colors)
gt_ply_file_path = "gt.ply"
o3d.io.write_point_cloud(gt_ply_file_path, gt)
print("Pred and GT point clouds saved to pred.ply and gt.ply")