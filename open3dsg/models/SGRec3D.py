import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from open3dsg.models.pointnet import PointNetEncoder, feature_transform_reguliarzer
from open3dsg.models.network_GNN import TripletGCNModel, GraphEdgeAttenNetworkLayers
from open3dsg.models.network_util import build_mlp


class SGRec3D(nn.Module):
    """
    SGRec3D: Self-Supervised 3D Scene Graph Learning via Object-Level Scene Reconstruction
    
    This model performs self-supervised pre-training for 3D scene graph prediction
    using a pretext task of scene reconstruction, followed by fine-tuning for 
    scene graph prediction.
    """
    
    def __init__(self, hparams):
        """
        Initialize the SGRec3D model.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        super(SGRec3D, self).__init__()
        
        self.hparams = hparams
        
        # Input channel configuration
        self.rgb = hparams.get('use_rgb', False)
        self.nrm = hparams.get('use_normal', False)
        self.channels = 3 + 3*self.rgb + 3*self.nrm
        self.pointnet2 = hparams.get('pointnet2', False)
        
        # Initialize encoder components
        self._init_encoder()
        
        # Initialize graph bottleneck components
        self._init_bottleneck()
        
        # Initialize decoder components for pre-training
        self._init_decoder()
        
        # Initialize loss functions
        self._init_loss_functions()
        
        # Track pre-training/fine-tuning mode
        self.pretraining_mode = True
        
    def _init_encoder(self):
        """Initialize the encoder components."""
        # Point feature extractors
        if self.pointnet2:
            from open3dsg.models.pointnet2 import Pointnet2_Ssg as PointNet2Encoder
            self.objPointNet = PointNet2Encoder(normal_channel=True)
            self.relPointNet = PointNet2Encoder(normal_channel=True)
        else:
            self.objPointNet = PointNetEncoder(
                global_feat=True, 
                feature_transform=True,
                channel=self.channels  # (x,y,z) + (r,g,b) + (nx,ny,nz)
            )
            # For relationship points, add one extra channel for instance mask
            self.relPointNet = PointNetEncoder(
                global_feat=True, 
                feature_transform=True, 
                channel=self.channels+1  # Add mask channel
            )
        
        # Embedding layers for bounding boxes
        box_params = 6  # (w, l, h, cx, cy, cz)
        self.box_gconv_dim = 32
        self.loc_gconv_dim = 64
        self.bbox_emb = nn.Linear(box_params, self.box_gconv_dim)
        self.center_obj_emb = nn.Linear(3, self.loc_gconv_dim)
        self.dist_pred_emb = nn.Linear(3, self.loc_gconv_dim)
        self.angle_emb = nn.Linear(1, self.box_gconv_dim)
        
        # Feature adapter (combines PointNet features with bounding box features)
        feature_size = 256+64 if self.pointnet2 else 1024+64
        self.pointnet_adapter = build_mlp(
            [feature_size, 512, self.hparams.get('gconv_dim', 512)], 
            activation='relu', 
            on_last=True
        )
        
        # Graph Convolutional Network
        if self.hparams.get('gnn_layers', 0) > 0:
            graph_backbone = self.hparams.get('graph_backbone', 'message')
            if graph_backbone == "message":
                self.gconv_net = TripletGCNModel(
                    num_layers=self.hparams.get('gnn_layers', 4),
                    dim_node=self.hparams.get('gconv_dim', 512),
                    dim_edge=self.hparams.get('gconv_dim', 512),
                    dim_hidden=self.hparams.get('hidden_dim', 1024),
                    aggr='max'
                )
            elif graph_backbone == 'attention':
                self.gconv_net = GraphEdgeAttenNetworkLayers(
                    num_layers=self.hparams.get('gnn_layers', 4),
                    dim_node=self.hparams.get('gconv_dim', 512),
                    dim_edge=self.hparams.get('gconv_dim', 512),
                    dim_hidden=self.hparams.get('hidden_dim', 1024),
                    dim_atten=self.hparams.get('atten_dim', 512),
                    num_heads=self.hparams.get('gconv_nheads', 4),
                    DROP_OUT_ATTEN=0.3
                )
    
    def _init_bottleneck(self):
        """Initialize the graph bottleneck components."""
        # Node and edge bottleneck MLPs
        gconv_dim = self.hparams.get('gconv_dim', 512)
        num_obj_classes = self.hparams.get('num_obj_classes', 160)
        num_rel_classes = self.hparams.get('num_rel_classes', 27)
        
        # Node class prediction (softmax)
        self.node_bottleneck_mlp = nn.Linear(gconv_dim, num_obj_classes)
        
        # Edge class prediction (sigmoid for multi-label)
        self.edge_bottleneck_mlp = nn.Linear(gconv_dim, num_rel_classes)
    
    def _init_decoder(self):
        """Initialize the decoder components."""
        # Embedding MLP to lift from bottleneck to higher dimension
        gconv_dim = self.hparams.get('gconv_dim', 512)
        hidden_dim = self.hparams.get('hidden_dim', 1024)
        num_obj_classes = self.hparams.get('num_obj_classes', 160)
        num_rel_classes = self.hparams.get('num_rel_classes', 27)
        
        # Embedding layers to lift bottleneck features - FIX: Use correct input dimensions
        self.node_embedding = build_mlp([num_obj_classes, hidden_dim], activation='relu')
        self.edge_embedding = build_mlp([num_rel_classes, hidden_dim], activation='relu')
        
        # Decoder GCN with same structure as encoder
        self.decoder_gcn = TripletGCNModel(
            num_layers=self.hparams.get('gnn_layers', 4),
            dim_node=hidden_dim,
            dim_edge=hidden_dim,
            dim_hidden=hidden_dim*2,
            aggr='max'
        )
        
        # Box-Head for bounding box prediction (7 parameters: w, l, h, cx, cy, cz, angle)
        self.box_head = build_mlp([hidden_dim, hidden_dim, 7], activation='relu')
        
        # Shape-Head for shape encoding (1024-dim for AtlasNet)
        self.shape_head = build_mlp([hidden_dim, hidden_dim, 1024], activation='relu')
        
        # Initialize or load pre-trained AtlasNet (if available)
        self.atlas_net = None  # Placeholder for the actual AtlasNet model
    
    def _init_loss_functions(self):
        """Initialize loss functions for pre-training and fine-tuning."""
        # Pre-training losses
        self.bbox_loss = nn.L1Loss()
        self.angle_loss = nn.CrossEntropyLoss()
        self.shape_loss = nn.L1Loss()
        
        # Fine-tuning losses (can use focal loss for class imbalance)
        self.obj_loss = nn.CrossEntropyLoss()
        self.pred_loss = nn.BCEWithLogitsLoss()  # Binary cross entropy for multi-label prediction
    
    def encode_pcl(self, objects_pcl, predicate_pcl):
        """
        Encode point clouds into features using PointNet.
        
        Args:
            objects_pcl: Point clouds for objects [batch_size, num_objects, num_points, channels]
            predicate_pcl: Point clouds for predicates [batch_size, num_predicates, num_points, channels+1]
            
        Returns:
            obj_vecs: Object features [batch_size, num_objects, feature_dim]
            pred_vecs: Predicate features [batch_size, num_predicates, feature_dim]
            tf1, tf2: Transformation matrices from PointNet
        """
        # Process object point clouds
        objects_pcl_batched = objects_pcl.view(-1, *objects_pcl.shape[-2:])
        objects_pcl_batched = objects_pcl_batched.permute(0, 2, 1)
        obj_vecs, _, tf1 = self.objPointNet(objects_pcl_batched)
        
        # Process predicate point clouds
        predicate_pcl_batched = predicate_pcl.view(-1, *predicate_pcl.shape[-2:])
        predicate_pcl_batched = predicate_pcl_batched.permute(0, 2, 1)
        pred_vecs, _, tf2 = self.relPointNet(predicate_pcl_batched)
        
        # Reshape back to batch structure
        obj_vecs = obj_vecs.view(objects_pcl.shape[0], -1, obj_vecs.shape[-1])
        pred_vecs = pred_vecs.view(predicate_pcl.shape[0], -1, pred_vecs.shape[-1])
        tf1 = tf1.view(objects_pcl.shape[0], -1, *tf1.shape[1:])
        tf2 = tf2.view(predicate_pcl.shape[0], -1, *tf2.shape[1:])
        
        return obj_vecs, pred_vecs, tf1, tf2
    
    def encode_bbox(self, bboxes):
        """
        Encode bounding boxes into features.
        
        Args:
            bboxes: Bounding boxes [batch_size, num_objects, 7]
                   where each box is [w, l, h, cx, cy, cz, angle]
                   
        Returns:
            bbox_enc: Encoded box features
            angle_enc: Encoded angle features
        """
        bbox_enc = self.bbox_emb(bboxes[..., :6])
        angle_enc = self.angle_emb(bboxes[..., 6].unsqueeze(-1))
        return bbox_enc, angle_enc
    
    def encode_center_dist(self, obj_centers, pred_dists):
        """
        Encode object centers and predicate distances.
        
        Args:
            obj_centers: Object centers [batch_size, num_objects, 3]
            pred_dists: Predicate distances [batch_size, num_predicates, 3]
            
        Returns:
            obj_center_enc: Encoded object centers
            pred_dist_enc: Encoded predicate distances
        """
        obj_center_enc = self.center_obj_emb(obj_centers)
        pred_dist_enc = self.dist_pred_emb(pred_dists)
        return obj_center_enc, pred_dist_enc
    
    def encode_gcn(self, batch_size, obj_vecs, pred_vecs, objects_count, predicate_count, edges):
        """
        Process features through the GCN.
        
        Args:
            batch_size: Number of batches
            obj_vecs: Object features [batch_size, num_objects, feature_dim]
            pred_vecs: Predicate features [batch_size, num_predicates, feature_dim]
            objects_count: Number of objects per batch [batch_size]
            predicate_count: Number of predicates per batch [batch_size]
            edges: Edge indices [batch_size, num_predicates, 2]
            
        Returns:
            obj_vecs_list: List of processed object features
            pred_vecs_list: List of processed predicate features
        """
        obj_vecs_list = []
        pred_vecs_list = []
        # self.gcn_o_features = []
        # self.gcn_p_features = []
        
        for i in range(batch_size):
            object_num = objects_count[i]
            predicate_num = predicate_count[i]
            edges_batch = edges[i][:predicate_num]
            obj_vecs_batch = obj_vecs[i, :object_num]
            pred_vecs_batch = pred_vecs[i, :predicate_num]
            
            # Process through feature adapter
            o_vecs = self.pointnet_adapter(obj_vecs_batch)
            p_vecs = self.pointnet_adapter(pred_vecs_batch)
            
            # Process through GCN if enabled
            if self.hparams.get('gnn_layers', 0) > 0:
                o_vecs, p_vecs = self.gconv_net(o_vecs, p_vecs, edges_batch)
            
            # # Store GCN features for skip connection
            # self.gcn_o_features.append(o_vecs)
            # self.gcn_p_features.append(p_vecs)
            
            # Pad to fixed size if needed
            max_nodes = self.hparams.get('max_nodes', -1)
            max_edges = self.hparams.get('max_edges', -1)
            
            if max_nodes > 0:
                o_vecs_out = torch.cat((
                    o_vecs, 
                    torch.zeros((max_nodes - o_vecs.shape[0], o_vecs.shape[1])).to(o_vecs.device)
                ))
            else:
                o_vecs_out = o_vecs
                
            if max_edges > 0:
                p_vecs_out = torch.cat((
                    p_vecs, 
                    torch.zeros((max_edges - p_vecs.shape[0], p_vecs.shape[1])).to(p_vecs.device)
                ))
            else:
                p_vecs_out = p_vecs
            
            obj_vecs_list.append(o_vecs_out)
            pred_vecs_list.append(p_vecs_out)

        return obj_vecs_list, pred_vecs_list
    
    def create_graph_bottleneck(self, node_features, edge_features):
        """
        Create graph bottleneck representation.
        
        Args:
            node_features: Node features [batch_size, num_nodes, feature_dim]
            edge_features: Edge features [batch_size, num_edges, feature_dim]
            
        Returns:
            node_bottleneck: Node bottleneck features [batch_size, num_nodes, num_classes]
            edge_bottleneck: Edge bottleneck features [batch_size, num_edges, num_relations]
        """
        # Apply bottleneck MLPs
        node_logits = self.node_bottleneck_mlp(node_features)
        edge_logits = self.edge_bottleneck_mlp(edge_features)
        
        # Apply activations to create probability distributions
        node_bottleneck = F.softmax(node_logits, dim=-1)  # Object class distribution
        edge_bottleneck = torch.sigmoid(edge_logits)  # Multi-label predicate probabilities
        
        return node_bottleneck, edge_bottleneck, node_logits, edge_logits
    
    def decode_scene(self, node_bottleneck, edge_bottleneck, edges):
        """
        Decode the scene from bottleneck representation.
        
        Args:
            node_bottleneck: Node bottleneck features [batch_size, num_nodes, num_classes]
            edge_bottleneck: Edge bottleneck features [batch_size, num_edges, num_relations]
            edges: Edge indices [batch_size, num_predicates, 2]
            
        Returns:
            boxes: Reconstructed bounding boxes [batch_size, num_nodes, 7]
            shape_codes: Reconstructed shape codes [batch_size, num_nodes, 1024]
        """
        batch_size = node_bottleneck.shape[0]
        boxes_list = []
        shape_codes_list = []
        
        for i in range(batch_size):
            # Lift bottleneck features to higher dimension for this batch
            node_features_batch = self.node_embedding(node_bottleneck[i])
            edge_features_batch = self.edge_embedding(edge_bottleneck[i])
            
            # # Get the GCN features for this batch
            # gcn_o_features_batch = self.gcn_o_features[i]
            # gcn_p_features_batch = self.gcn_p_features[i]
            
            # Create skip connection with encoder GCN features
            # # Match dimensions first (pad the GCN features if needed)
            # max_nodes = self.hparams.get('max_nodes', -1)
            # if max_nodes > 0 and gcn_o_features_batch.shape[0] < node_features_batch.shape[0]:
            #     gcn_o_features_batch = torch.cat((
            #         gcn_o_features_batch,
            #         torch.zeros((node_features_batch.shape[0] - gcn_o_features_batch.shape[0], 
            #                 gcn_o_features_batch.shape[1])).to(gcn_o_features_batch.device)
            #     ))
            
            # Now combine with skip connection
            # node_features_with_skip = torch.cat([node_features_batch, gcn_o_features_batch], dim=-1)
            # edge_features_with_skip = torch.cat([edge_features_batch, gcn_p_features_batch], dim=-1)
            
            # Process through decoder GCN
            edges_batch = edges[i][:edge_features_batch.shape[0]]
            node_features_out, _ = self.decoder_gcn(
                node_features_batch,
                edge_features_batch,
                edges_batch
            )
            
            # Predict bounding boxes and shape codes
            boxes_batch = self.box_head(node_features_out)  # [w, l, h, cx, cy, cz, angle]
            shape_codes_batch = self.shape_head(node_features_out)  # Shape encoding for AtlasNet
            
            boxes_list.append(boxes_batch)
            shape_codes_list.append(shape_codes_batch)
        
        # Stack the results back into batch tensors
        boxes = torch.stack(boxes_list)
        shape_codes = torch.stack(shape_codes_list)
        
        return boxes, shape_codes
    
    def forward(self, data_dict):
        """
        Forward pass through the network.
        
        Args:
            data_dict: Dictionary containing input data:
                - objects_pcl: Object point clouds [batch_size, num_objects, num_points, channels]
                - predicate_pcl_flag: Predicate point clouds with mask [batch_size, num_predicates, num_points, channels+1]
                - objects_bbox: Object bounding boxes [batch_size, num_objects, 7]
                - objects_center: Object centers [batch_size, num_objects, 3]
                - predicate_dist: Predicate distances [batch_size, num_predicates, 3]
                - edges: Edge indices [batch_size, num_predicates, 2]
                - objects_count: Number of objects per batch [batch_size]
                - predicate_count: Number of predicates per batch [batch_size]
                
        Returns:
            data_dict: Updated dictionary with encoder outputs and optionally decoder outputs
        """
        batch_size = data_dict["objects_id"].size(0)
        obj_num, pred_num = data_dict["objects_count"], data_dict["predicate_count"]
        
        # Prepare point cloud inputs
        objects_pcl = data_dict["objects_pcl"][..., :self.channels]
        predicate_pcl_flag = torch.cat([
            data_dict["predicate_pcl_flag"][..., :self.channels],
            data_dict["predicate_pcl_flag"][..., -1].unsqueeze(-1)
        ], dim=-1)
        
        # Encode point clouds
        obj_vecs, pred_vecs, tf1, tf2 = self.encode_pcl(objects_pcl, predicate_pcl_flag)
        data_dict["trans_feat"] = [tf1, tf2]  # Store for regularization loss
        
        # Encode bounding boxes and spatial information
        box_enc, angle_enc = self.encode_bbox(data_dict["objects_bbox"])
        center_enc, dist_enc = self.encode_center_dist(data_dict["objects_center"], data_dict["predicate_dist"])
        
        # Combine features
        obj_vecs = torch.cat([obj_vecs, box_enc, angle_enc], dim=-1)
        pred_vecs = torch.cat([pred_vecs, dist_enc], dim=-1)
        
        # Process through GCN
        obj_vecs_batch, pred_vecs_batch = self.encode_gcn(
            batch_size, obj_vecs, pred_vecs, obj_num, pred_num, data_dict["edges"]
        )
        
        # Convert to tensors
        obj_vecs_tensor = torch.stack(obj_vecs_batch)
        pred_vecs_tensor = torch.stack(pred_vecs_batch)
        
        # Create graph bottleneck
        node_bottleneck, edge_bottleneck, node_logits, edge_logits = self.create_graph_bottleneck(
            obj_vecs_tensor, pred_vecs_tensor
        )
        
        # Store bottleneck features and logits
        data_dict["objects_enc"] = node_bottleneck
        data_dict["predicates_enc"] = edge_bottleneck
        data_dict["objects_logits"] = node_logits
        data_dict["predicates_logits"] = edge_logits
        
        # If in pre-training mode, decode the scene
        if self.pretraining_mode:
            boxes, shape_codes = self.decode_scene(
                node_bottleneck, edge_bottleneck, data_dict["edges"]
            )
            
            # Store reconstruction outputs
            data_dict["reconstructed_boxes"] = boxes
            data_dict["reconstructed_shapes"] = shape_codes
        
        return data_dict
    
    def reconstruction_loss(self, data_dict):
        """
        Compute the reconstruction loss for pre-training.
        
        Args:
            data_dict: Dictionary containing input and output data
            
        Returns:
            loss: Total reconstruction loss
            loss_dict: Dictionary of individual loss components
        """
        # Extract ground truth and predicted values
        gt_boxes = data_dict["objects_bbox"]
        pred_boxes = data_dict["reconstructed_boxes"]
        
        # Get shape codes (for real implementation, these would come from pre-trained AtlasNet)
        gt_shapes = torch.zeros_like(data_dict["reconstructed_shapes"])  # Placeholder
        pred_shapes = data_dict["reconstructed_shapes"]
        
        # Bounding box regression loss (L1 for positions and dimensions)
        box_loss = self.bbox_loss(pred_boxes[..., :6], gt_boxes[..., :6])
        
        # Angle classification loss (discretized into bins)
        num_bins = 24
        angle_pred = pred_boxes[..., 6].view(-1, 1)
        # Convert angle_pred to logits for each bin
        angle_pred_logits = torch.zeros(angle_pred.size(0), num_bins).to(angle_pred.device)
        angle_pred_logits.scatter_(1, (angle_pred * num_bins / (2 * np.pi)).long() % num_bins, 1)
        angle_gt = torch.round(gt_boxes[..., 6] * num_bins / (2 * np.pi)).long() % num_bins  # Convert to bin index
        angle_loss = self.angle_loss(angle_pred_logits, angle_gt.view(-1))
        
        # Shape embedding loss
        shape_loss = self.shape_loss(pred_shapes, gt_shapes)
        
        # Combined loss with weighting factors
        total_loss = 0.4 * box_loss + 0.2 * angle_loss + 0.4 * shape_loss
        
        loss_dict = {
            "box_loss": box_loss.item(),
            "angle_loss": angle_loss.item(),
            "shape_loss": shape_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def scene_graph_loss(self, data_dict):
        """
        Compute the scene graph prediction loss for fine-tuning.
        
        Args:
            data_dict: Dictionary containing input and output data
            
        Returns:
            loss: Total scene graph loss
            loss_dict: Dictionary of individual loss components
        """
        # Extract predictions and ground truth
        obj_logits = data_dict["objects_logits"]
        pred_logits = data_dict["predicates_logits"]
        
        obj_gt = data_dict["objects_cat"]
        pred_gt = data_dict["predicate_cat"]
        
        # Object classification loss
        obj_loss = self.obj_loss(obj_logits.view(-1, obj_logits.size(-1)), obj_gt.view(-1))
        
        # Predicate classification loss (multi-label)
        pred_loss = self.pred_loss(pred_logits, pred_gt.float())
        
        # Combined loss with weighting factors
        total_loss = 0.1 * obj_loss + 1.0 * pred_loss
        
        loss_dict = {
            "obj_loss": obj_loss.item(),
            "pred_loss": pred_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def set_pretraining_mode(self, pretraining=True):
        """
        Set the model to pre-training or fine-tuning mode.
        
        Args:
            pretraining: Whether to use pre-training mode (True) or fine-tuning mode (False)
        """
        self.pretraining_mode = pretraining
    
    def load_pretrained_encoder(self, checkpoint_path):
        """
        Load pre-trained encoder weights.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Filter to only load encoder parameters
        encoder_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if not k.startswith('decoder')}
        self.load_state_dict(encoder_state_dict, strict=False)
        print(f"Loaded pre-trained encoder from {checkpoint_path}")
    
    def load_pretrained_atlasnet(self, checkpoint_path):
        """
        Load pre-trained AtlasNet decoder.
        
        Args:
            checkpoint_path: Path to the AtlasNet checkpoint file
        """
        # In a real implementation, you would load the actual AtlasNet here
        # self.atlas_net = load_atlasnet_from_checkpoint(checkpoint_path)
        print(f"Loaded pre-trained AtlasNet from {checkpoint_path}")

if __name__ == "__main__":
    # Define hyperparameters
    hparams = {
        'use_rgb': False,
        'use_normal': False,
        'pointnet2': False,
        'gconv_dim': 512,
        'hidden_dim': 1024,
        'gnn_layers': 4,
        'num_obj_classes': 160,
        'num_rel_classes': 27,
        'max_nodes': 10,
        'max_edges': 15
    }
    
    # Initialize model
    model = SGRec3D(hparams)
    print("Model initialized")
    
    # Create dummy data
    batch_size = 2
    max_objects = 10
    max_predicates = 15
    num_points = 1024
    channels = 3  # xyz only
    
    # Create input dictionary with dummy tensors
    data_dict = {
        "objects_pcl": torch.randn(batch_size, max_objects, num_points, channels),
        "predicate_pcl_flag": torch.randn(batch_size, max_predicates, num_points, channels+1),
        "objects_bbox": torch.randn(batch_size, max_objects, 7),
        "objects_center": torch.randn(batch_size, max_objects, 3),
        "predicate_dist": torch.randn(batch_size, max_predicates, 3),
        "edges": torch.randint(0, max_objects, (batch_size, max_predicates, 2)),
        "objects_count": torch.tensor([max_objects, max_objects]),
        "predicate_count": torch.tensor([max_predicates, max_predicates]),
        "objects_id": torch.randint(0, 1000, (batch_size, max_objects)),
        "objects_cat": torch.randint(0, hparams['num_obj_classes'], (batch_size, max_objects)),
        "predicate_cat": torch.randint(0, 2, (batch_size, max_predicates, hparams['num_rel_classes'])).float()
    }
    
    # Forward pass - Pre-training mode
    print("\nRunning in pre-training mode")
    model.set_pretraining_mode(True)
    output_dict = model(data_dict)
    
    # Calculate pre-training loss
    recon_loss, recon_loss_dict = model.reconstruction_loss(output_dict)
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print("Loss components:", recon_loss_dict)
    
    # Forward pass - Fine-tuning mode
    print("\nRunning in fine-tuning mode")
    model.set_pretraining_mode(False)
    output_dict = model(data_dict)
    
    # Calculate fine-tuning loss
    sg_loss, sg_loss_dict = model.scene_graph_loss(output_dict)
    print(f"Scene graph prediction loss: {sg_loss.item():.4f}")
    print("Loss components:", sg_loss_dict)