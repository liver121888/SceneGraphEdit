from open3dsg.data.sgrec_dataloader import SGRecDataset
from open3dsg.models.SGRec3D import SGRec3D
import open3d as o3d
import os
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(1124)
class SGRecEvaluator:
    def __init__(self, config, model = None,  model_path=None, dataset = None, dataset_path = None, ply_saving_path ='./reconstruction', device='cuda:0', swap_yz=False, re_angle=False):
        if model is not None:
            assert model_path is None, "If model is provided, model_path should be None"
            self.model = model
            self.model.to(device)
            self.model.train()
            self.use_full_bbox = self.model.use_full_bbox
            # print("AtlasNet parameters: ", float(sum(p.sum().item() for p in self.model.atlas_net.parameters())))
            print("use full box", self.use_full_bbox)
        else:
            assert model_path is not None, "If model is None, model_path should be provided"
            self.hparams = config["hparams"]
            self.use_full_bbox = self.hparams.get('use_full_bbox', False)
            self.model = SGRec3D(hparams=self.hparams, device=device)
            self.model.pretraining_mode = self.hparams.get('pretraining_mode', True)
            # model_state_dict = torch.load(model_path)
            # for key in list(model_state_dict.keys()):
            #     print(key)
            # input()
            ckpt = torch.load(model_path)
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)
            print("Using trained model from: ", model_path)
            # atlas_net_state_dict = torch.load('/home/ubuntu/SceneGraphEdit/AtlasNetLocal/checkpoint/network.pth')s
            self.model.train()
        self.device = device
        self.swap_yz = swap_yz
        
        if dataset is not None:
            self.dataset = dataset
        else:
            assert dataset_path is not None, "If dataset is None, dataset_path should be provided"
            hparams = config['hparams']
            self.dataset = SGRecDataset(max_objects=hparams['max_nodes'],
                                        max_rels=hparams['max_edges'],
                                        device=device,
                                        downsample_points=1024,
                                        data_root_dir=dataset_path,
                                        swap_yz=swap_yz,
                                        re_angle=re_angle)
        
        self.ply_saving_path = ply_saving_path
        
        # Print Parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters in the model: {total_params}")
    
    def reconstruct_gt(self, model_result_dict, pt_colors):
        self.model.train()
        gt_pcl = model_result_dict['objects_pcl'].clone().cpu().numpy() # （B, object_num, pts, 9）
        gt_pcl = gt_pcl.reshape(-1, gt_pcl.shape[2], gt_pcl.shape[3])[..., :3]
        

        gt_center = model_result_dict['objects_center'].clone().cpu().numpy() # (B, object_num, 3)
        gt_center = gt_center.reshape(-1, gt_center.shape[-1])
        gt_scale = model_result_dict['objects_scale'].cpu().numpy() # (B, object_num, 1)
        gt_scale = gt_scale.reshape(-1, gt_scale.shape[-1])
        if self.use_full_bbox:
            gt_bbox = model_result_dict['objects_bbox'].cpu().numpy()
            gt_bbox = gt_bbox.reshape(-1, gt_bbox.shape[-1])
            gt_angle = gt_bbox[..., -1:]

        gt_points = []
        gt_colors = []

        for i in range(model_result_dict['objects_count'][0]):
            pts = gt_pcl[i]
            # theta = float(gt_angle[i])
            # rot_matrix = np.array([
            #         [np.cos(theta), 0, -np.sin(theta)],
            #         [np.sin(theta), 0, np.cos(theta)],
            #         [0, 1, 0]
            # ])
            # pts = pts @ rot_matrix
            if self.use_full_bbox:
                pts = self._transform_points([pts], [gt_center[i]], [gt_scale[i]], [gt_angle[i]])
            else:
                pts = self._transform_points([pts], [gt_center[i]], [gt_scale[i]])
            gt_points.append(pts)
            color = np.tile(pt_colors[i], (pts.shape[0], 1))
            gt_colors.append(color)
        gt_points_combined = np.concatenate(gt_points, axis=0)
        gt_colors_combined = np.concatenate(gt_colors, axis=0)
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_points_combined)
        gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors_combined)
        
        return gt_pcd
    
    def reconstruct_gt_with_atlas(self, model_result_dict, pt_colors):
        self.model.train()
        gt_pcl = model_result_dict['objects_pcl'].clone().cpu().numpy() # （B, object_num, pts, 9）
        gt_pcl = gt_pcl[..., :3]
        atlas_encoder_features = self.model.get_atlas_encoder_output(torch.tensor(gt_pcl).to('cuda')) # (B, num_obj, dim)
        atlas_encoder_features = atlas_encoder_features.reshape(-1, atlas_encoder_features.shape[-1])
        atlas_decoder_output = self.model.atlas_net.module.decoder(atlas_encoder_features)

        gt_center = model_result_dict['objects_center'].cpu().numpy() # (B, object_num, 3)
        gt_center = gt_center.reshape(-1, gt_center.shape[-1])
        gt_scale = model_result_dict['objects_scale'].cpu().numpy() # (B, object_num, 1)
        gt_scale = gt_scale.reshape(-1, gt_scale.shape[-1])
        if self.use_full_bbox:
            gt_bbox = model_result_dict['objects_bbox'].cpu().numpy()
            gt_bbox = gt_bbox.reshape(-1, gt_bbox.shape[-1])
            gt_angle = gt_bbox[..., -1:]
        
        gt_points = []
        gt_colors = []
        for i in range(model_result_dict['objects_count'][0]):
            pts = atlas_decoder_output[i].squeeze(0).T.detach().cpu().numpy()
            if self.use_full_bbox:
                pts = self._transform_points([pts], [gt_center[i]], [gt_scale[i]], [gt_angle[i]])
            else:
                pts = self._transform_points([pts], [gt_center[i]], [gt_scale[i]])
            gt_points.append(pts)
            color = np.tile(pt_colors[i], (pts.shape[0], 1))
            gt_colors.append(color)
        gt_points_combined = np.concatenate(gt_points, axis=0)
        gt_colors_combined = np.concatenate(gt_colors, axis=0)
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_points_combined)
        gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors_combined)
        
        return gt_pcd
        
    def reconstruct_pred_with_atlas(self, model_result_dict, pt_colors):
        # self.model.load_pretrained_atlasnet()
        self.model.train()
        atlas_encoder_features = self.model.get_atlas_encoder_output(model_result_dict["objects_pcl"]) # (B, num_obj, dim)
        atlas_encoder_features = atlas_encoder_features.reshape(-1, atlas_encoder_features.shape[-1])
        atlas_decoder_output = self.model.atlas_net.module.decoder(atlas_encoder_features)

        bbox_pred = model_result_dict['reconstructed_boxes'].cpu().numpy() # (B, object_num, (cx, cy, cz, scale)
        bbox_pred = bbox_pred.reshape(-1, bbox_pred.shape[-1]) # (B*object_num, 4)
        pred_centers = bbox_pred[:, :3]
        pred_scales = bbox_pred[:, 3]
        if self.use_full_bbox:
            pred_thetas = bbox_pred[:, 4]
            
        points = []
        colors = []
        for i in range(model_result_dict['objects_count'][0]):
            pts = atlas_decoder_output[i].squeeze(0).T.detach().cpu().numpy()
            if self.use_full_bbox:
                pts = self._transform_points([pts], [pred_centers[i]], [pred_scales[i]], [pred_thetas[i]])
            else:
                pts = self._transform_points([pts], [pred_centers[i]], [pred_scales[i]])
            points.append(pts)
            color = np.tile(pt_colors[i], (pts.shape[0], 1))
            colors.append(color)
        points_combined = np.concatenate(points, axis=0)
        colors_combined = np.concatenate(colors, axis=0)
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(points_combined)
        pred_pcd.colors = o3d.utility.Vector3dVector(colors_combined)
        return pred_pcd
    
    def reconstruct_pred_with_gt_center_scale(self, model_result_dict, pt_colors, model_use_pretrained_atlasnet=False):
        self.model.train()
        # bbox_pred = model_result_dict['reconstructed_boxes'].cpu().numpy() # (B, object_num, (cx, cy, cz, scale)
        # bbox_pred = bbox_pred.reshape(-1, bbox_pred.shape[-1]) # (B*object_num, 4)
        # pred_centers = bbox_pred[:, :3]
        # pred_scales = bbox_pred[:, 3]
        # print("pred_scales: ", pred_scales)
        # print("pred_centers: ", pred_centers)
        gt_center = model_result_dict['objects_center'].cpu().numpy() # (B, object_num, 3)
        gt_center = gt_center.reshape(-1, gt_center.shape[-1])
        gt_scale = model_result_dict['objects_scale'].cpu().numpy() # (B, object_num, 1)
        gt_scale = gt_scale.reshape(-1, gt_scale.shape[-1])
        if self.use_full_bbox:
            # pred_thetas = bbox_pred[:, 4]
            print("gt_angle: ", model_result_dict["objects_bbox"].shape)
            gt_angle = model_result_dict["objects_bbox"].reshape(-1,model_result_dict["objects_bbox"].shape[-1])[:, -1:].cpu().numpy()
            
            
        pcl_pred_features = model_result_dict['reconstructed_shapes']
        features = pcl_pred_features.reshape(-1, pcl_pred_features.shape[-1])
        if model_use_pretrained_atlasnet:
            decoder_output = self.model.atlas_net.module.decoder(features)
        else:
            decoder_output = model_result_dict["objects_dec"]
        # print("decoder_output shape: ", decoder_output.shape)
        # print("pred_scales shape: ", pred_scales.shape)
        # print("pred_centers shape: ", pred_centers.shape)

        points = []
        colors = []
        for i in range(model_result_dict['objects_count'][0]):
            if model_use_pretrained_atlasnet:
                pts = decoder_output[i].squeeze(0).T.detach().cpu().numpy()
            else:
                pts = decoder_output[i].detach().cpu().numpy()
                
            if self.use_full_bbox:
                pts = self._transform_points([pts], [gt_center[i]], [gt_scale[i]], [gt_angle[i]])
            else:
                pts = self._transform_points([pts], [gt_center[i]], [gt_scale[i]])
                
            points.append(pts)
            color = np.tile(pt_colors[i], (pts.shape[0], 1))
            colors.append(color)
        points_combined = np.concatenate(points, axis=0)
        colors_combined = np.concatenate(colors, axis=0)
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(points_combined)
        pred_pcd.colors = o3d.utility.Vector3dVector(colors_combined)
        
        loss, loss_dict = self.model.reconstruction_loss(model_result_dict)
        print("Reconstruction loss: ", loss.item())
        print("Reconstruction loss dict: ", loss_dict)
        
        return pred_pcd
    
    def reconstruct_pred(self, model_result_dict, pt_colors, model_use_pretrained_atlasnet=False):
        self.model.train()
        bbox_pred = model_result_dict['reconstructed_boxes'].cpu().numpy() # (B, object_num, (cx, cy, cz, scale)
        bbox_pred = bbox_pred.reshape(-1, bbox_pred.shape[-1]) # (B*object_num, 4)
        pred_centers = bbox_pred[:, :3]
        pred_scales = bbox_pred[:, 3]
        if self.use_full_bbox:
            pred_thetas = bbox_pred[:, 4]
            
        pcl_pred_features = model_result_dict['reconstructed_shapes']
        features = pcl_pred_features.reshape(-1, pcl_pred_features.shape[-1])
        if model_use_pretrained_atlasnet:
            decoder_output = self.model.atlas_net.module.decoder(features)
        else:
            decoder_output = model_result_dict["objects_dec"]
        # print("decoder_output shape: ", decoder_output.shape)
        # print("pred_scales shape: ", pred_scales.shape)
        # print("pred_centers shape: ", pred_centers.shape)

        points = []
        colors = []
        for i in range(model_result_dict['objects_count'][0]):
            if model_use_pretrained_atlasnet:
                pts = decoder_output[i].squeeze(0).T.detach().cpu().numpy()
            else:
                pts = decoder_output[i].detach().cpu().numpy()
                
            if self.use_full_bbox:
                pts = self._transform_points([pts], [pred_centers[i]], [pred_scales[i]], [pred_thetas[i]])
            else:
                pts = self._transform_points([pts], [pred_centers[i]], [pred_scales[i]])
                
            points.append(pts)
            color = np.tile(pt_colors[i], (pts.shape[0], 1))
            colors.append(color)
        points_combined = np.concatenate(points, axis=0)
        colors_combined = np.concatenate(colors, axis=0)
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(points_combined)
        pred_pcd.colors = o3d.utility.Vector3dVector(colors_combined)
        
        loss, loss_dict = self.model.reconstruction_loss(model_result_dict)
        print("Reconstruction loss: ", loss.item())
        print("Reconstruction loss dict: ", loss_dict)
        
        return pred_pcd, features

    def eval_one_scene(self, scene_id, prefix='0'):
        scene_reconstruction_path = os.path.join(self.ply_saving_path, str(scene_id))
        scene_reconstruction_path = os.path.join(scene_reconstruction_path, prefix)
        if not os.path.exists(scene_reconstruction_path):
            os.makedirs(scene_reconstruction_path)
            
        scene_data = self.dataset.__getitem__(scene_id)
        data_dict = self.dataset.collate_fn([scene_data])
        print(data_dict["predicate_pcl_flag"])
        print(data_dict["edges"])
        data_dict = {key: value.to(self.device) for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
        
        max_obj = self.dataset.max_objs
        colors = np.random.rand(max_obj, 3)
        model_use_pretrained_atlasnet = self.model.use_pretrained_atlasnet
        with torch.no_grad():
            ### GT Reconstruction
            # self.model.atlas_net.load_state_dict(torch.load('/home/ubuntu/SceneGraphEdit/AtlasNetLocal/checkpoint/network.pth'))
            model_result_dict = self.model(data_dict)
            torch.save(model_result_dict, os.path.join(scene_reconstruction_path, f"model_result_dict_{scene_id}.pth"))
            gt_pcd = self.reconstruct_gt(model_result_dict, colors)
            gt_ply_file_path = os.path.join(scene_reconstruction_path, f"gt_{scene_id}.ply")
            o3d.io.write_point_cloud(gt_ply_file_path, gt_pcd)
            print(f"GT point cloud saved to {gt_ply_file_path}")
            
            
            ## Complete Reconstruction
            pred_pcd, features = self.reconstruct_pred(model_result_dict, colors, model_use_pretrained_atlasnet)
            features_file_path = os.path.join(scene_reconstruction_path, f"features_{scene_id}.npy")
            np.save(features_file_path, features.cpu().numpy())
            pred_ply_file_path = os.path.join(scene_reconstruction_path, f"pred_{scene_id}.ply")
            o3d.io.write_point_cloud(pred_ply_file_path, pred_pcd)
            print(f"Reconstruction saved to {pred_ply_file_path}")
            
            # ### Atlas Reconstruction
            # pred_pcd_atlas = self.reconstruct_pred_with_atlas(model_result_dict, colors)
            # pred_ply_file_path_atlas = os.path.join(scene_reconstruction_path, f"pred_atlas_{scene_id}.ply")
            # o3d.io.write_point_cloud(pred_ply_file_path_atlas, pred_pcd_atlas)
            # print(f"Pred Atlas reconstruction saved to {pred_ply_file_path_atlas}")
            
            # ### GT Center Scale Reconstruction
            # pred_pcd_atlas_gt_bbox = self.reconstruct_pred_with_gt_center_scale(model_result_dict, colors, model_use_pretrained_atlasnet)
            # pred_ply_file_path_gt_bbox = os.path.join(scene_reconstruction_path, f"pred_atlas_{scene_id}_with_gt_bbox.ply")
            # o3d.io.write_point_cloud(pred_ply_file_path_gt_bbox, pred_pcd_atlas_gt_bbox)
            # print(f"Pred with GT bbox reconstruction saved to {pred_ply_file_path_atlas}")
            
            # gt_pcd_atlas = self.reconstruct_gt_with_atlas(model_result_dict, colors)
            # gt_ply_file_path_atlas = os.path.join(scene_reconstruction_path, f"gt_atlas_{scene_id}.ply")
            # o3d.io.write_point_cloud(gt_ply_file_path_atlas, gt_pcd_atlas)
            # print(f"GT Atlas reconstruction saved to {gt_ply_file_path_atlas}")
            # input()
    
    def eval_metrics(self, scene_id):
        """
        Evaluate reconstruction quality using multiple metrics:
        1. 3D Box IoU
        """
        scene_data = self.dataset.__getitem__(scene_id)
        data_dict = self.dataset.collate_fn([scene_data])
        data_dict = {key: value.to(self.device) for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
        
        with torch.no_grad():
            model_result_dict = self.model(data_dict)
            gt_center = model_result_dict["objects_center"] # (B, object_num, 3)
            gt_scale = model_result_dict["objects_scale"] # (B, object_num, 1)
            gt_angle = model_result_dict["objects_bbox"][..., -1:] # B x N x 1

            pred_center = model_result_dict["reconstructed_boxes"][..., :3] # (B, object_num, 3)
            pred_scale = model_result_dict["reconstructed_boxes"][..., 3:4] #
            pred_angle = model_result_dict["reconstructed_boxes"][..., -1:] # (B, object_num, 1)
            
            obj_num = model_result_dict["objects_count"][0]
            # print("obj_num at scene {}: ".format(scene_id), obj_num)
            scale_diff = torch.abs(gt_scale - pred_scale)[:, :obj_num, :]
            center_diff = torch.norm(gt_center - pred_center, dim=-1)[:, :obj_num]
            
            ang_diff = torch.abs(torch.atan2(
                torch.sin(pred_angle - gt_angle),
                torch.cos(pred_angle - gt_angle)
            ))[:, :obj_num, :]
            
            mean_abs_scale_err = torch.mean(scale_diff)
            mean_abs_center_err = torch.mean(center_diff)
            mean_abs_angle_err = torch.mean(ang_diff)
            
            max_abs_scale_err = torch.max(scale_diff)
            max_abs_center_err = torch.max(center_diff)
            max_abs_angle_err = torch.max(ang_diff)
            
            metrics = {
                'mean_abs_scale_err': mean_abs_scale_err.item(),
                'mean_abs_center_err': mean_abs_center_err.item(),
                'max_abs_scale_err': max_abs_scale_err.item(),
                'max_abs_center_err': max_abs_center_err.item(),
                'mean_abs_angle_err': mean_abs_angle_err.item(),
                'max_abs_angle_err': max_abs_angle_err.item(),
            }
                        
            if not self.model.use_pretrained_atlasnet:
                cd = self.model.points_chamfer_loss(model_result_dict)
                metrics['chamfer_distance'] = cd.item()
            
            
            return metrics

    def _calculate_3d_box_giou(self, box1, box2):
        """Calculate IoU between axis-aligned 3D boxes"""
        def get_box_min_max(box):
            center = box[:3]
            size = np.abs(box[3:6])
            size = np.maximum(size, 1e-6)
            return center - size/2, center + size/2
        
        box1_min, box1_max = get_box_min_max(box1)
        box2_min, box2_max = get_box_min_max(box2)
        intersection_min = np.maximum(box1_min, box2_min)
        intersection_max = np.minimum(box1_max, box2_max)
        
        intersection_sizes = np.maximum(0.0, intersection_max - intersection_min)
        intersection = np.prod(intersection_sizes)
        
        vol1 = np.prod(np.abs(box1_max - box1_min))
        vol2 = np.prod(np.abs(box2_max - box2_min))
        
        union = vol1 + vol2 - intersection
        
        if union < 1e-6:
            return 0.0
            
        iou = intersection / union
        
        enclosing_vol = np.prod(np.maximum(box1_max, box2_max) - np.minimum(box1_min, box2_min))
        
        giou = iou - (enclosing_vol - (vol1 + vol2 - intersection)) / enclosing_vol
        return giou

    # def _transform_points_with_center_scale(self, points, centers, scales):

    #     transformed_points = []
    #     # print("debug")
    #     for i in range(len(centers)):
    #         pts = points[i]
    #         center = centers[i]
    #         scale = scales[i]
    #         pts = pts * scale
    #         pts = pts + center
    #         transformed_points.append(pts)
    #     return np.concatenate(transformed_points, axis=0)
            
    
    def _transform_points(self, points, centers, scales, thetas=None):
        """Transform points using box parameters"""
        transformed_points = []
        
        for i in range(len(centers)):
            pts = points[i]
            center = centers[i]
            scale = scales[i]
            if isinstance(pts, torch.Tensor):
                pts = pts.cpu().numpy()
                
            if thetas:
                theta = float(thetas[i])
                rot_matrix = np.array([
                    [np.cos(theta), 0, -np.sin(theta)],
                    [np.sin(theta), 0, np.cos(theta)],
                    [0, 1, 0]
                ]).T
                pts = pts @ rot_matrix
                
            pts = pts * scale
            pts = pts + center
            transformed_points.append(pts)
        
        return np.concatenate(transformed_points, axis=0)

    def draw_one_scene_graph(self, scene_id, prefix='0'):
        scene_reconstruction_path = os.path.join(self.ply_saving_path, str(scene_id))
        scene_reconstruction_path = os.path.join(scene_reconstruction_path, prefix)
        if not os.path.exists(scene_reconstruction_path):
            os.makedirs(scene_reconstruction_path)
            
        scene_data = self.dataset.__getitem__(scene_id)
        data_dict = self.dataset.collate_fn([scene_data])
        pcl_flags = data_dict["objects_cat"]
        print(pcl_flags)
        edges = data_dict["predicate_cat"]
        max_prob_indices = torch.argmax(edges, dim=-1)
        print("Edge indices with highest probabilities:", max_prob_indices)
        # print(edges.shape)
        class_dict = {
            27: "chair",
            154: "wall",
            139: "table",
            57: "floor",
            147: "towel",
            26: "ceiling",
            143: "toilet",
            148: "trash can",
            123: "shower",
            92: "towel rack",
        }
        relation_dict = {
            0: "none",
            15: "standing on",
            14: "attached to",
            17: "hanging on",
            2: "left",
            3: "right",
            9: "smaller than",
            8: "bigger than",
            6: "close by"
        }
        classes = [class_dict[int(i)] for i in pcl_flags[0]]
        relations = [relation_dict[int(i)] for i in max_prob_indices[0]]
        print("Classes: ", classes)
        print("Relations: ", relations)
        import matplotlib.pyplot as plt

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes (classes)
        for i, cls in enumerate(classes):
            G.add_node(i, label=(cls))
        print(data_dict["edges"][0])
        # Add edges (relations) if they are not "none"
        for i, rel in enumerate(relations):
            if rel != "none":
                src, dst = data_dict["edges"][0][i].tolist()
                G.add_edge(src, dst, label=rel)
        G.add_edge(1, 6, label="standing on")
        G.add_edge(8, 7, label="right")
        G.add_edge(2, 6, label="left")
        # G.add_edge(6, 4, label="attached to")
        G.remove_edge(0, 5)
        G.remove_edge(0, 3)
        G.remove_edge(3, 5)
        G.remove_edge(2, 5)

        # Draw the graph
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=1000, node_color="lightblue", font_size=10)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=8)

        # Save and show the graph
        graph_file_path = os.path.join(scene_reconstruction_path, f"scene_graph_{scene_id}.png")
        plt.savefig(graph_file_path)
        print(f"Scene graph saved to {graph_file_path}")
        plt.close()
        
    

if __name__ == "__main__":
    import json
    import networkx as nx
    # model_path = "/home/ubuntu/SceneGraphEdit/sgrec_pretrained_checkpoints/v14/pretrained_epoch_125000.pth"
    # dataset_path = "/home/ubuntu/SceneGraphEdit/data_out/datasets/OpenSG_3RScan/preprocessed_custom_atlas_feat"
    # config_path = "/home/ubuntu/SceneGraphEdit/open3dsg/config/config_v14.json"
    model_path = "/home/ubuntu/SceneGraphEdit/sgrec_finetuned_checkpoints/v20_ft/finetuned_epoch_90000.pth"
    dataset_path = "/home/ubuntu/SceneGraphEdit/data_out/datasets/OpenSG_3RScan/preprocessed_custom_atlas_feat"
    config_path = "/home/ubuntu/SceneGraphEdit/open3dsg/config/config_v20_ft.json"
    config = json.load(open(config_path, 'r'))
    hparams = config['hparams']

    evaluator = SGRecEvaluator(model_path=model_path,
                               dataset_path=dataset_path,
                               config={'hparams':hparams},
                               device='cuda:0',
                               swap_yz=False,
                               re_angle=True)
    evaluator.use_full_bbox = True
    # evaluator = SGRecEvaluator(model=model, dataset_path=dataset_path, config={'hparams':hparams})
    # evaluator = SGRecEvaluator(model=model, dataset=sgrec_dataset, config={'hparams':hparams})
    max_dataset_idx = len(evaluator.dataset.file_mapping)
    max_eval = 10
    np.random.seed(1124)
    # idxs = [3449]
    idxs = [5, 10, 100, 675, 1000, 1650, 1666, 2037, 2131, 2572, 3449]
    from collections import defaultdict
    metricss = defaultdict(float)
    for idx in idxs:
        # evaluator.draw_one_scene_graph(idx)
        metrics = evaluator.eval_metrics(idx)
        for key, value in metrics.items():
            print(f"{key}: {value}")
            metricss[key] += value
    
    for key, value in metricss.items():
        metricss[key] = value / len(idxs)
    print("Average metrics: ", metricss)
    
        # print(f"Metrics for scene {idx}: {metrics}")
    # for i in range(max_eval):
    #     random_sample_idx = np.random.randint(0, max_dataset_idx)
    #     evaluator.eval_one_scene(random_sample_idx)
    #     metrics = evaluator.eval_metrics(random_sample_idx)
    #     print(f"Metrics for scene {random_sample_idx}: {metrics}")
    
    
            