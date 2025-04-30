import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from open3dsg.models.pointnet import PointNetEncoder, feature_transform_reguliarzer
from open3dsg.models.network_GNN import TripletGCNModel, GraphEdgeAttenNetworkLayers
from open3dsg.models.network_util import build_mlp
from open3dsg.models.spherical_harmonics_dec import SphericalHarmonicsDecoder
import sys
sys.path.append('.')
from AtlasNetLocal.model.model import EncoderDecoder
from AtlasNetLocal.auxiliary import argument_parser as argument_parser
from pytorch3d.loss import chamfer_distance

class SGRec3D(nn.Module):
    """
    SGRec3D: Self-Supervised 3D Scene Graph Learning via Object-Level Scene Reconstruction
    
    This model performs self-supervised pre-training for 3D scene graph prediction
    using a pretext task of scene reconstruction, followed by fine-tuning for 
    scene graph prediction.
    """
    
    def __init__(self, hparams, device='cuda:1', dtype=torch.float32):
        """
        Initialize the SGRec3D model.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        super(SGRec3D, self).__init__()
        
        self.hparams = hparams
        self.device = device
        print(f"Using device: {self.device}")
        
        # Input channel configuration
        self.rgb = hparams.get('use_rgb', False)
        self.nrm = hparams.get('use_normal', False)
        self.channels = 3 + 3*self.rgb + 3*self.nrm
        self.pointnet2 = hparams.get('pointnet2', False)
        self.shape_loss_type = hparams.get('shape_loss_type', 'feature')
        self.use_pretrained_atlasnet = self.hparams.get('use_pretrained_atlasnet', False)
        self.mask_padding = self.hparams.get('mask_padding', False)
        self.chamfer_reduction = hparams.get('chamfer_reduction', 'mean')
        self.use_implicit_decoder = self.hparams.get('use_implicit_decoder', False)
        
        if self.use_implicit_decoder:
            assert not self.use_pretrained_atlasnet, "Cannot use pre-trained AtlasNet with implicit decoder"
        
        # Initialize encoder components
        
        self._init_encoder()
        
        # Initialize graph bottleneck components
        self._init_bottleneck()
        self._use_softmax_in_bottleneck = hparams.get('use_softmax_in_bottleneck', True)
        
        # Track pre-training/fine-tuning mode
        self.pretraining_mode = self.hparams.get('pretraining_mode', True)
        
        if not self.pretraining_mode:
            self._init_sg_encoder()
            self._init_sg_bottleneck()
        
        # Initialize decoder components for pre-training
        self._init_decoder()
        
        # Initialize loss functions
        self._init_loss_functions()

        
    def _init_encoder(self):
        """Initialize the encoder components."""
        self.use_full_bbox = self.hparams.get('use_full_bbox', False) # if True, use full bbox (cx, cy, cz, scale, angle), else use (cx, cy, cz, scale)
        print(f"Using full bbox: {self.use_full_bbox}")
        self.point_net_adaptor_layers = self.hparams.get('point_net_adaptor_layers', 2)
        self.gconv_dim = self.hparams.get('gconv_dim', 512)
        
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
            ).to(self.device)
            # For relationship points, add one extra channel for instance mask
            self.relPointNet = PointNetEncoder(
                global_feat=True, 
                feature_transform=True, 
                channel=self.channels+1  # Add mask channel
            ).to(self.device)
        
        # Embedding layers for bounding boxes
        box_params = 4 # (cx, cy, cz, scale)
        self.box_gconv_dim = 8
        self.loc_gconv_dim = 8
        
        # self.bbox_emb = nn.Linear(box_params, self.box_gconv_dim).to(self.device)
        self.bbox_emb = build_mlp(
            [box_params, 256, 256, self.box_gconv_dim],
            activation='relu',
        ).to(self.device)
        
        if self.use_full_bbox:
            self.ang_emb = build_mlp(
                [box_params + 1, 64, 64, 1],
                activation='relu',
            ).to(self.device)
        
        self.dist_pred_emb = nn.Linear(3, self.loc_gconv_dim).to(self.device)
        
        # Feature adapter (combines PointNet features with bounding box features)
        self.point_net_features_size = 256 if self.pointnet2 else 1024
        obj_adaptor_feature_size = self.point_net_features_size + self.box_gconv_dim
        obj_adaptor_feature_size += 1 if self.use_full_bbox else 0
        
        pred_adaptor_feature_size = self.point_net_features_size + self.loc_gconv_dim
        
        self.pointnet_adapter_obj = build_mlp(
            [obj_adaptor_feature_size] + [self.gconv_dim] * self.point_net_adaptor_layers,
            activation='relu', 
            on_last=True
        ).to(self.device)
        
        self.pointnet_adapter_pred = build_mlp(
            [pred_adaptor_feature_size] + [self.gconv_dim] * self.point_net_adaptor_layers,
            activation='relu', 
            on_last=True
        ).to(self.device)
        
        # Graph Convolutional Network
        if self.hparams.get('gnn_layers', 0) > 0:
            graph_backbone = self.hparams.get('graph_backbone', 'message')
            if graph_backbone == "message":
                self.gconv_net = TripletGCNModel(
                    num_layers=self.hparams.get('gnn_layers', 4),
                    dim_node=self.gconv_dim,
                    dim_edge=self.gconv_dim,
                    dim_hidden=self.hparams.get('hidden_dim', 1024),
                    aggr='mean'
                ).to(self.device)
            elif graph_backbone == 'attention':
                self.gconv_net = GraphEdgeAttenNetworkLayers(
                    num_layers=self.hparams.get('gnn_layers', 4),
                    dim_node=self.hparams.get('gconv_dim', 512),
                    dim_edge=self.hparams.get('gconv_dim', 512),
                    dim_hidden=self.hparams.get('hidden_dim', 1024),
                    dim_atten=self.hparams.get('atten_dim', 512),
                    num_heads=self.hparams.get('gconv_nheads', 4),
                    DROP_OUT_ATTEN=0.3
                ).to(self.device)
                
    def _init_sg_encoder(self):
        # Given object cat, predicate cat, and predicate dist, create encoding through GCN
            # object_cat = data_dict["objects_cat"] # B x num_objects x num_object_cls
            # predicate_cat = data_dict["predicate_cat"] # B x num_predicates x num_predicate_cls
            # predicate_dist = data_dict["predicate_dist"] # B x num_predicates x 3
        
        num_obj_classes = self.hparams.get('num_obj_classes', 160)
        num_rel_classes = self.hparams.get('num_rel_classes', 27)
        self.obj_clss_embedding = build_mlp(
            [num_obj_classes, self.gconv_dim, self.gconv_dim, self.gconv_dim, self.gconv_dim],
            activation='relu',
        ).to(self.device)
        
        self.pred_clss_embedding = build_mlp(
            [num_rel_classes, self.gconv_dim, self.gconv_dim, self.gconv_dim, self.gconv_dim],
            activation='relu',
        ).to(self.device)
        
        self.sg_encoding_gconv = TripletGCNModel(
            num_layers=self.hparams.get('gnn_layers', 4) * 2,
            dim_node=self.gconv_dim,
            dim_edge=self.gconv_dim,
            dim_hidden=self.hparams.get('hidden_dim', 1024),
            aggr='mean'
        ).to(self.device)
    
    def _init_bottleneck(self):
        """Initialize the graph bottleneck components."""
        # Node and edge bottleneck MLPs
        gconv_dim = self.hparams.get('gconv_dim', 512)
        num_obj_classes = self.hparams.get('num_obj_classes', 160)
        num_rel_classes = self.hparams.get('num_rel_classes', 27)
        
        # Node class prediction (softmax)
        # self.node_bottleneck_mlp = nn.Linear(gconv_dim, num_obj_classes).to(self.device)
        self.node_bottleneck_mlp = build_mlp(
            [gconv_dim, gconv_dim, num_obj_classes],
            activation='relu',
        ).to(self.device)
        
        # Edge class prediction (sigmoid for multi-label)
        # self.edge_bottleneck_mlp = nn.Linear(gconv_dim, num_rel_classes).to(self.device)
        self.edge_bottleneck_mlp = build_mlp(
            [gconv_dim, gconv_dim, num_rel_classes],
            activation='relu',
        ).to(self.device)
    
    def _init_sg_bottleneck(self):
        """Initialize the graph bottleneck components."""
        # Node and edge bottleneck MLPs
        gconv_dim = self.hparams.get('gconv_dim', 512)
        num_obj_classes = self.hparams.get('num_obj_classes', 160)
        num_rel_classes = self.hparams.get('num_rel_classes', 27)
        
        # Node class prediction (softmax)
        # self.node_bottleneck_mlp = nn.Linear(gconv_dim, num_obj_classes).to(self.device)
        self.sg_node_bottleneck_mlp = build_mlp(
            [gconv_dim, gconv_dim, gconv_dim, gconv_dim, num_obj_classes],
            activation='relu',
        ).to(self.device)
        
        # Edge class prediction (sigmoid for multi-label)
        # self.edge_bottleneck_mlp = nn.Linear(gconv_dim, num_rel_classes).to(self.device)
        self.sg_edge_bottleneck_mlp = build_mlp(
            # [gconv_dim, gconv_dim, num_rel_classes],
            [gconv_dim, gconv_dim, gconv_dim, gconv_dim, num_rel_classes],
            activation='relu',
        ).to(self.device)
    
    def _init_decoder(self):
        """Initialize the decoder components."""
        # Embedding MLP to lift from bottleneck to higher dimension
        gconv_dim = self.hparams.get('gconv_dim', 512)
        hidden_dim = self.hparams.get('hidden_dim', 1024)
        num_obj_classes = self.hparams.get('num_obj_classes', 160)
        num_rel_classes = self.hparams.get('num_rel_classes', 27)
        self.use_pointnet_skip = self.hparams.get('use_pointnet_skip', True)
        
        # Embedding layers to lift bottleneck features - FIX: Use correct input dimensions
        self.node_embedding = build_mlp([num_obj_classes, hidden_dim], activation='relu').to(self.device)
        self.edge_embedding = build_mlp([num_rel_classes, hidden_dim], activation='relu').to(self.device)
        
        # Decoder GCN with same structure as encoder
        self.decoder_gcn = TripletGCNModel(
            num_layers=self.hparams.get('gnn_layers', 4),
            dim_node=hidden_dim,
            dim_edge=hidden_dim,
            dim_hidden=hidden_dim*2,
        ).to(self.device)
        
        # (5 parameters: cx, cy, cz, scale, angle) or (4 parameters: cx, cy, cz, scale)
        box_dim = 5 if self.use_full_bbox else 4
        self.box_head = build_mlp([hidden_dim, hidden_dim, hidden_dim, box_dim], activation='relu').to(self.device)
        
        # Shape-Head for shape encoding (1024-dim for AtlasNet)
        self.shape_head = build_mlp([hidden_dim, hidden_dim * 2, hidden_dim, 1024], activation='relu').to(self.device)
        
        # Initialize or load pre-trained AtlasNet (if available)
        # if not self.use_pretrained_atlasnet:
        if self.use_implicit_decoder:
            self.implicit_decoder = SphericalHarmonicsDecoder(
                latent_dim=1024, 
                max_degree=8, 
                num_points=1024
            ).to(self.device)
        else:
            self.load_pretrained_atlasnet()
        # pointnet features for skip connection
        self.pointnet_obj_features = None
        
        if self.use_pointnet_skip:
            self.skip_connection_adapter = build_mlp(
                [self.point_net_features_size + hidden_dim, hidden_dim, hidden_dim], 
                activation='relu', 
                on_last=True
            ).to(self.device)
        else:
            self.gcn_decode_adapter = build_mlp(
                [hidden_dim, hidden_dim, hidden_dim, hidden_dim],
                activation='relu',
                on_last=True
            ).to(self.device)
    
    def create_masks(self, data_dict):
        """Create masks for valid objects and predicates based on counts."""
        batch_size = data_dict["objects_pcl"].size(0)
        max_objects = data_dict["objects_pcl"].size(1)
        max_predicates = data_dict["predicate_pcl_flag"].size(1)
        
        # Create object mask
        obj_mask = torch.zeros((batch_size, max_objects), device=self.device, dtype=torch.bool)
        for i in range(batch_size):
            obj_mask[i, :data_dict["objects_count"][i]] = True
        
        # Create predicate mask
        pred_mask = torch.zeros((batch_size, max_predicates), device=self.device, dtype=torch.bool)
        for i in range(batch_size):
            pred_mask[i, :data_dict["predicate_count"][i]] = True
    
        return obj_mask, pred_mask
    
    def MSE_loss_with_mask(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
        # gt: (B x N x C)
        # pred: (B x N x C)
        # mask: (B x N)
        diff = (pred - gt) ** 2
        num_valid = mask.sum()
        assert num_valid > 0, "No valid elements in mask"
        loss = (diff.mean(dim=-1) * mask).sum() / num_valid
        return loss
        
    def _init_loss_functions(self):
        """Initialize loss functions for pre-training and fine-tuning."""
        # Pre-training losses
        if not self.mask_padding:
            self.bbox_loss = nn.MSELoss()
            self.scale_loss = nn.MSELoss()
            self.shape_loss = nn.MSELoss()
        else:
            self.bbox_loss = self.MSE_loss_with_mask
            self.scale_loss = self.MSE_loss_with_mask
            self.shape_loss = self.MSE_loss_with_mask
        
        # Fine-tuning losses (can use focal loss for class imbalance)
        if self._use_softmax_in_bottleneck:
            self.obj_loss = nn.CrossEntropyLoss()
            self.pred_loss = nn.BCEWithLogitsLoss()  # Binary cross entropy for multi-label prediction
        else: # MSE loss
            self.obj_loss = nn.MSELoss()
            self.pred_loss = nn.MSELoss()
    
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
        
        if self.use_pointnet_skip:
            self.pointnet_obj_features = obj_vecs
        
        return obj_vecs, pred_vecs, tf1, tf2
    
    def encode_bbox(self, bboxes):
        """
        Encode bounding boxes into features.
        
        Args:
            bboxes: Bounding boxes [batch_size, num_objects, 4], (cx, cy, cz, scale) or (cx, cy, cz, scale, angle)
                   
        Returns:
            bbox_enc: Encoded box features
        """
        if self.use_full_bbox:
            assert bboxes.shape[-1] == 5, "Bounding boxes must have 5 dimensions (cx, cy, cz, scale, angle)"
            bbox_enc = self.bbox_emb(bboxes[..., :4])
            angle_enc = self.ang_emb(bboxes)
            return bbox_enc, angle_enc
        else:
            bbox_enc = self.bbox_emb(bboxes)
            return bbox_enc
    
    def encode_center_dist(self, pred_dists):
        """
        Encode object centers and predicate distances.
        
        Args:
            pred_dists: Predicate distances [batch_size, num_predicates, 3]
            
        Returns:
            center_dist_enc: Encoded center and distance features
        """
        center_dist_enc = self.dist_pred_emb(pred_dists)
        return center_dist_enc
    
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
            o_vecs = self.pointnet_adapter_obj(obj_vecs_batch)
            p_vecs = self.pointnet_adapter_pred(pred_vecs_batch)
            
            # Process through GCN if enabled
            if self.hparams.get('gnn_layers', 0) > 0:
                o_vecs, p_vecs = self.gconv_net(o_vecs, p_vecs, edges_batch)
            
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
        node_logits = self.node_bottleneck_mlp(node_features) # [batch_size, num_nodes, num_classes]
        edge_logits = self.edge_bottleneck_mlp(edge_features) # [batch_size, num_edges, num_relations]
        
        # Apply activations to create probability distributions
        if self._use_softmax_in_bottleneck:
            node_bottleneck = F.softmax(node_logits, dim=-1)  # Object class distribution
            edge_bottleneck = torch.sigmoid(edge_logits)  # Multi-label predicate probabilities
        else:
            node_bottleneck = node_logits  # Object class distribution
            edge_bottleneck = edge_logits  # Multi-label predicate probabilities
        
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
                        
            # Process through decoder GCN
            edges_batch = edges[i][:edge_features_batch.shape[0]]
            node_features_out, _ = self.decoder_gcn(
                node_features_batch,
                edge_features_batch,
                edges_batch
            )
            
            if self.use_pointnet_skip:
                # Skip connection from PointNet features
                pointnet_features_batch = self.pointnet_obj_features[i]
                decode_features = torch.cat([node_features_out, pointnet_features_batch], dim=-1)
                node_features_out = self.skip_connection_adapter(decode_features)
            else:
                node_features_out = self.gcn_decode_adapter(node_features_out)

            # Predict bounding boxes and shape codes
            boxes_batch = self.box_head(node_features_out)  # [cx, cy, cz, scale, angle] or [cx, cy, cz, scale]
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
        if self.pretraining_mode:
            # use point cloud encoder
            self.ptcld_encoder_forward(data_dict)
        else:
            self.scene_graph_encoder_forward(data_dict)
        self.decoder_forward(data_dict)
        
        return data_dict
    
    def scene_graph_encoder_forward(self, data_dict):
        object_cat = data_dict["objects_cat"]  # B x num_objects x num_object_cls
        predicate_cat = data_dict["predicate_cat"]  # B x num_predicates x num_predicate_cls
        edges = data_dict["edges"]  # B x num_predicates x 2
        
        batch_size = object_cat.size(0)
        obj_num, pred_num = data_dict["objects_count"], data_dict["predicate_count"]
        
        # Process object and predicate categories through embedding layers
        obj_vecs_batch = []
        pred_vecs_batch = []
        
        for i in range(batch_size):
            object_num = obj_num[i]
            predicate_num = pred_num[i]
            
            # Get embeddings for this batch
            # print(f"object_cat shape: {object_cat.shape}, predicate_cat shape: {predicate_cat.shape}")
            obj_cat_batch = object_cat[i, :object_num]  # num_objects x num_object_cls
            pred_cat_batch = predicate_cat[i, :predicate_num]  # num_predicates x num_predicate_cls
            
            # print(f"obj_cat_batch shape: {obj_cat_batch.shape}, pred_cat_batch shape: {pred_cat_batch.shape}")
            # Create embeddings
            obj_embedding = self.obj_clss_embedding(obj_cat_batch)  # num_objects x gconv_dim
            pred_embedding = self.pred_clss_embedding(pred_cat_batch)  # num_predicates x gconv_dim
            
            # print(f"obj_embedding shape: {obj_embedding.shape}, pred_embedding shape: {pred_embedding.shape}")
                                    
            # Process through GCN
            edges_batch = edges[i][:predicate_num]
            o_vecs, p_vecs = self.sg_encoding_gconv(obj_embedding, pred_embedding, edges_batch)
            
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
            
            obj_vecs_batch.append(o_vecs_out)
            pred_vecs_batch.append(p_vecs_out)
        
        # Convert lists to tensors
        obj_vecs_tensor = torch.stack(obj_vecs_batch)  # B x max_nodes x gconv_dim
        pred_vecs_tensor = torch.stack(pred_vecs_batch)  # B x max_edges x gconv_dim
        
        # Create graph bottleneck using scene graph bottleneck MLPs
        node_logits = self.sg_node_bottleneck_mlp(obj_vecs_tensor)
        edge_logits = self.sg_edge_bottleneck_mlp(pred_vecs_tensor)
        
        # Apply activations to create probability distributions
        if self._use_softmax_in_bottleneck:
            node_bottleneck = F.softmax(node_logits, dim=-1)
            edge_bottleneck = torch.sigmoid(edge_logits)
        else:
            node_bottleneck = node_logits
            edge_bottleneck = edge_logits
        
        # Store bottleneck features and logits
        data_dict["objects_enc_from_sg"] = node_bottleneck
        data_dict["predicates_enc_from_sg"] = edge_bottleneck
        data_dict["objects_logits_from_sg"] = node_logits
        data_dict["predicates_logits_from_sg"] = edge_logits
        
        return data_dict
    
    def ptcld_encoder_forward(self, data_dict):
        batch_size = data_dict["objects_pcl"].size(0)
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
        # print(f"pred_vecs shape: {pred_vecs.shape}")
        # Encode bounding boxes and spatial information
        if self.use_full_bbox:
            # Use full bounding box representation (cx, cy, cz, scale, angle)
            # print("Shapes: ", data_dict["objects_center"].shape, data_dict["objects_scale"].shape, data_dict["objects_bbox"][...,-1:].shape)
            bbox = torch.concat([data_dict["objects_center"], data_dict["objects_scale"], data_dict["objects_bbox"][..., -1:]], dim=-1)
            # print("bbox shape: ", bbox.shape)
            box_enc, angle_enc = self.encode_bbox(bbox)
            obj_vecs = torch.cat([obj_vecs, box_enc, angle_enc], dim=-1)
        else:
            bbox = torch.concat([data_dict["objects_center"], data_dict["objects_scale"]], dim=-1)
            box_enc = self.encode_bbox(bbox)
            obj_vecs = torch.cat([obj_vecs, box_enc], dim=-1)

            
        center_dist_enc = self.encode_center_dist(data_dict["predicate_dist"])
        pred_vecs = torch.cat([pred_vecs, center_dist_enc], dim=-1)
        
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
        
        return data_dict
    
    def decoder_forward(self, data_dict):
        if self.pretraining_mode:
            node_bottleneck = data_dict["objects_enc"] # B x max_nodes x num_classes
            edge_bottleneck = data_dict["predicates_enc"] # B x max_edges x num_relations
        else:
            node_bottleneck = data_dict["objects_enc_from_sg"]
            edge_bottleneck = data_dict["predicates_enc_from_sg"]
        boxes, shape_codes = self.decode_scene(
                node_bottleneck, edge_bottleneck, data_dict["edges"]
            )
            
        data_dict["reconstructed_boxes"] = boxes
        data_dict["reconstructed_shapes"] = shape_codes
        if not self.use_pretrained_atlasnet:
            if self.use_implicit_decoder:
                data_dict["objects_dec"] = self.get_implicit_decoder_output(shape_codes)
            else:
                data_dict["objects_dec"] = self.get_atlas_decoder_output(shape_codes)
        
        return data_dict
        
    def box_angle_loss(self, data_dict):
        center_loss, scale_loss = self.center_scale_loss(data_dict)
        assert data_dict["objects_bbox"].shape[-1] == 7, "Bounding boxes must have 7 dimensions (cx, cy, cz, w, h, d, angle)"
        assert data_dict["reconstructed_boxes"].shape[-1] == 5, "Reconstructed boxes must have 5 dimensions (cx, cy, cz, scale, angle)"
        angle_gt = data_dict["objects_bbox"][..., -1:] # B x N x 1
        angle_pred = data_dict["reconstructed_boxes"][..., -1:] # B x N x 1
        if self.mask_padding:
            obj_mask, _ = self.create_masks(data_dict)
            angle_gt = angle_gt[obj_mask]
            angle_pred = angle_pred[obj_mask]
        
        angle_diff = torch.abs(torch.atan2(
            torch.sin(angle_pred - angle_gt),
            torch.cos(angle_pred - angle_gt)
        ))
        angle_loss = angle_diff.mean() / torch.pi
        assert not torch.isnan(angle_loss).any(), "NaN detected in angle_loss"
        return center_loss, scale_loss, angle_loss*8 # to scale the angle loss to be comparable to center and scale loss

    def center_scale_loss(self, data_dict):
        gt_center = data_dict["objects_center"]
        gt_scale = data_dict["objects_scale"]
        pred_center = data_dict["reconstructed_boxes"][..., :3]
        pred_scale = data_dict["reconstructed_boxes"][..., 3:4]
        # print("pred_center shape: ", pred_center.shape)
        # print("gt_center shape: ", gt_center.shape)
        # print("pred_scale shape: ", pred_scale.shape)
        # print("gt_scale shape: ", gt_scale.shape)
        # print("pred_center", pred_center)
        # print("gt_center", gt_center)

        if not self.mask_padding:
            center_loss = self.bbox_loss(pred_center, gt_center)
            scale_loss = self.scale_loss(pred_scale, gt_scale)
        else:
            obj_mask, _ = self.create_masks(data_dict)
            center_loss = self.bbox_loss(pred_center, gt_center, obj_mask)
            scale_loss = self.scale_loss(pred_scale, gt_scale, obj_mask)
        assert not torch.isnan(center_loss).any(), "NaN detected in center_loss"
        assert not torch.isnan(scale_loss).any(), "NaN detected in scale_loss"
        return center_loss, scale_loss
    
    def shape_feature_loss(self, data_dict):
        gt_pcl = data_dict["objects_pcl"]
        # self.atlas_net.eval()
        gt_feature = self.get_atlas_encoder_output(gt_pcl)
        # gt_feature = data_dict["atlas_feat"]
        pred_shapes = data_dict["reconstructed_shapes"] # B x N x 1024
        if self.mask_padding:
            obj_mask, _ = self.create_masks(data_dict)
            shape_loss = self.shape_loss(pred_shapes, gt_feature, obj_mask)
        else:
            shape_loss = self.shape_loss(pred_shapes, gt_feature)
        assert not torch.isnan(shape_loss).any(), "NaN detected in shape_loss"
        return shape_loss
    
    def points_chamfer_loss(self, data_dict):
        gt_points = data_dict["objects_pcl"]
        pred_points = data_dict["objects_dec"]
        batch, num_objects, num_points, channels = gt_points.shape
        if self.mask_padding:
            obj_mask, _ = self.create_masks(data_dict) # B x N
            gt_points = gt_points[obj_mask]
            pred_points = pred_points[obj_mask.reshape(-1)]
        
        chamfer_loss, _ = chamfer_distance(gt_points[..., :3].reshape(-1,num_points,3), pred_points, point_reduction=self.chamfer_reduction)
        
        return chamfer_loss

    def sg_logits_loss(self, data_dict, grad_accum_steps=1):
        # compute point cloud encoding
        with torch.no_grad():
            self.ptcld_encoder_forward(data_dict)
            target_obj_enc = data_dict["objects_enc"]
            target_pred_enc = data_dict["predicates_enc"]
        
        # B x num_objects x num_object_cls
        obj_enc_sg = data_dict["objects_enc_from_sg"]
        # B x num_predicates x num_predicate_cls
        pred_logits_sg = data_dict["predicates_logits_from_sg"]
        
        obj_loss = self.obj_loss(obj_enc_sg.view(-1, obj_enc_sg.size(-1)),
                                target_obj_enc.view(-1, target_obj_enc.size(-1)))
        pred_loss = self.pred_loss(pred_logits_sg.view(-1, pred_logits_sg.size(-1)),
                                    target_pred_enc.view(-1, target_pred_enc.size(-1)))

        sg_logits_loss = obj_loss + pred_loss
        loss_dict = {
            "obj_loss": obj_loss.item(),
            "pred_loss": pred_loss.item(),
            "sg_logits_loss": sg_logits_loss.item()
        }
        
        sg_logits_loss /= grad_accum_steps
        for key in loss_dict:
            loss_dict[key] /= grad_accum_steps
        return sg_logits_loss, loss_dict
    
    def reconstruction_loss(self, data_dict, grad_accum_steps=1):
        """
        Compute the reconstruction loss for pre-training.
        
        Args:
            data_dict: Dictionary containing input and output data
            
        Returns:
            loss: Total reconstruction loss
            loss_dict: Dictionary of individual loss components
        """
        if self.use_pretrained_atlasnet:
            shape_loss = self.shape_feature_loss(data_dict)
        else:
            shape_loss = self.points_chamfer_loss(data_dict)  * 10   
                 
        if self.use_full_bbox:
            center_loss, scale_loss, angle_loss = self.box_angle_loss(data_dict)
            total_loss = center_loss + scale_loss + angle_loss + shape_loss
        
            loss_dict = {
                "center_loss": center_loss.item(),
                "scale_loss": scale_loss.item(),
                "angle_loss": angle_loss.item(),
                "shape_loss": shape_loss.item(),
                "total_loss": total_loss.item()
            }
        else:
            center_loss, scale_loss = self.center_scale_loss(data_dict)
            # total_loss = 0.4 * center_loss + 0.3 * scale_loss + 0.3 * shape_loss
            total_loss = 0.2 * center_loss + 0.2 * scale_loss +  0.6 * shape_loss
            
            loss_dict = {
                "center_loss": center_loss.item(),
                "scale_loss": scale_loss.item(),
                "shape_loss": shape_loss.item(),
                "total_loss": total_loss.item()
            }
        
        total_loss /= grad_accum_steps
        for key in loss_dict:
            loss_dict[key] /= grad_accum_steps
        
        return total_loss, loss_dict
    
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
    
    def load_pretrained_atlasnet(self, checkpoint_path="/home/ubuntu/SceneGraphEdit/AtlasNetLocal/checkpoint/network.pth"):
        """
        Load pre-trained AtlasNet decoder and freeze its parameters.
        
        Args:
            checkpoint_path: Path to the AtlasNet checkpoint file
        """
        print(f"Loaded pre-trained AtlasNet from {checkpoint_path}")
        opt = argument_parser.parser()
        torch.cuda.set_device(opt.multi_gpu[0])
        # torch.cuda.set_device(1)
        opt['device'] = self.device
        # opt['number_points'] = 1500
        # opt['number_points_eval'] = 1000
        # opt['device'] = self.device
        self.atlas_net = EncoderDecoder(opt)
        # self.atlas_net = torch.nn.DataParallel(self.atlas_net, device_ids=[1])
        self.atlas_net = torch.nn.DataParallel(self.atlas_net, device_ids=opt.multi_gpu)
        
        # if self.use_pretrained_atlasnet:
        self.atlas_net.load_state_dict(torch.load(checkpoint_path))
        
        self.atlas_net.to(self.device)
        print("AtlasNet loaded to ", self.atlas_net.device_ids)
        print("AtlasNet parameters: ", sum(p.numel() for p in self.atlas_net.parameters()))
        # Freeze AtlasNet parameters
        for param in self.atlas_net.parameters():
            if self.use_pretrained_atlasnet:
                param.requires_grad = False
            else:
                param.requires_grad = True
               
    def get_atlas_encoder_output(self, x):
        with torch.no_grad():
            batch_size, num_objects, num_points, channels = x.shape
            x = x[..., :self.channels]
            x_reshaped = x.reshape(-1, num_points, self.channels)
            x_permuted = x_reshaped.permute(0, 2, 1) # [batch_size*num_objects, channels, num_points]
            
            # # normalize to fit in a unit cube
            # x_permuted_min = torch.min(x_permuted, dim=2, keepdim=True)[0]
            # x_permuted_max = torch.max(x_permuted, dim=2, keepdim=True)[0]
            # x_permuted_center = (x_permuted_min + x_permuted_max) / 2
            # x_permuted = x_permuted - x_permuted_center  # Center the point cloud
            # x_permuted_scale = torch.max(x_permuted_max - x_permuted_min, dim=2, keepdim=True)[0]
            # x_permuted = x_permuted / (x_permuted_scale + 1e-6)
            # print("x_scale: ", x_permuted_scale)
            # assert no nan
            # assert not torch.isnan(x_permuted).any(), "NaN detected in x_permuted"
                        
            encoded = self.atlas_net.module.encoder(x_permuted)
            encoded_dim = encoded.shape[1]
            encoded_reshaped = encoded.reshape(batch_size, num_objects, encoded_dim)
        
        return encoded_reshaped
    
    def get_implicit_decoder_output(self, x):
        batch_size, num_objects, encoded_dim = x.shape
        x_reshaped = x.reshape(-1, encoded_dim)
        decoded = self.implicit_decoder(x_reshaped)
        return decoded
    
    def get_atlas_decoder_output(self, x):
        if self.use_pretrained_atlasnet:
            with torch.no_grad():
                batch_size, num_objects, encoded_dim = x.shape
                x_reshaped = x.reshape(-1, encoded_dim)
                decoded = self.atlas_net.module.decoder(x_reshaped)
                decoded_reshaped = decoded.squeeze(1).permute(0,2,1)
        else:
            batch_size, num_objects, encoded_dim = x.shape
            x_reshaped = x.reshape(-1, encoded_dim)
            decoded = self.atlas_net.module.decoder(x_reshaped)
            decoded_reshaped = decoded.squeeze(1).permute(0,2,1)
        
        return decoded_reshaped
    

if __name__ == "__main__":
    # Define hyperparameters
    torch.cuda.empty_cache()
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
        'max_edges': 72,
        'use_pointnet_skip': False,
        'pretraining_mode': False,
        'use_pretrained_atlasnet': False,
        'use_full_bbox': True,
        'mask_padding': True,
        'use_implicit_decoder': True
    }
    device = 'cpu'
    # Initialize model
    model = SGRec3D(hparams=hparams, device=device)
    print("Model initialized")
    # print number of parameters, vram usage
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    print("VRAM usage:", sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024, "MB")
    
    # Create dummy data
    batch_size = 4
    max_objects = 10
    max_predicates = 72
    num_points = 1024
    channels = 3  # xyz only
    
    # Create input dictionary with dummy tensors
    object_counts = torch.tensor([torch.randint(2, max_objects+1, (1,)).item() for _ in range(batch_size)]).to(device)
    predicate_counts = torch.tensor([torch.randint(1, max_predicates+1, (1,)).item() for _ in range(batch_size)]).to(device)

    # Now create edges that only reference valid objects
    edges = torch.zeros((batch_size, max_predicates, 2), dtype=torch.long).to(device)
    for i in range(batch_size):
        # Generate random edge indices only up to the valid object count for this batch
        valid_objects = object_counts[i].item()
        edges[i, :predicate_counts[i]] = torch.randint(0, valid_objects, (predicate_counts[i], 2))
        
    data_dict = {
        "objects_pcl": torch.randn(batch_size, max_objects, num_points, channels).to(device),
        "predicate_pcl_flag": torch.randn(batch_size, max_predicates, num_points, channels+1).to(device),
        "objects_bbox": torch.randn(batch_size, max_objects, 7).to(device),
        "objects_center": torch.randn(batch_size, max_objects, 3).to(device),
        "objects_scale": torch.randn(batch_size, max_objects, 1).to(device),
        "predicate_dist": torch.randn(batch_size, max_predicates, 3).to(device),
        # "edges": torch.randint(0, max_objects, (batch_size, max_predicates, 2)).to(device),
        # "objects_count": torch.tensor([max_objects] * batch_size).to(device),
        # "predicate_count": torch.tensor([max_predicates] * batch_size).to(device),
        "edges": edges,
        "objects_count": object_counts,
        "predicate_count": predicate_counts,
        "object_dec": torch.randn(batch_size, max_objects, num_points, channels).to(device),
        "atlas_feat": torch.randn(batch_size, max_objects, 1024).to(device),
        # "objects_id": torch.randint(0, 1000, (batch_size, max_objects)).to(device),
        # scene graph prediction at fine-tuning
        "objects_cat": torch.randint(0, 2, (batch_size, max_objects, hparams['num_obj_classes'])).float().to(device),
        "predicate_cat": torch.randint(0, 2, (batch_size, max_predicates, hparams['num_rel_classes'])).float().to(device)
    }
    # forward pass - not pre-training
    print("Running in fine-tuning mode")
    model.pretraining_mode = False
    output_dict = model.scene_graph_encoder_forward(data_dict)
    sg_loss, sg_loss_dict = model.sg_logits_loss(data_dict)
    print(f"Scene graph prediction loss: {sg_loss.item():.4f}")
    print("Loss components:", sg_loss_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Perform one optimization step
    optimizer.zero_grad()
    sg_loss.backward()
    optimizer.step()
    
    
    # # Forward pass - Pre-training mode
    # print("\nRunning in pre-training mode")
    # model.pretraining_mode = True
    # # print(model)
    # output_dict = model(data_dict)
    
    # # Calculate pre-training loss
    # recon_loss, recon_loss_dict = model.reconstruction_loss(output_dict)
    # print(f"Reconstruction loss: {recon_loss.item():.4f}")
    # print("Loss components:", recon_loss_dict)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # # Perform one optimization step
    # optimizer.zero_grad()
    # recon_loss.backward()
    # optimizer.step()

    # print("Performed one optimization step")
    
    # Verify that gradients were computed and parameters updated
    print("Checking Gradient")
    for name, param in list(model.named_parameters()):
        if param.grad is None:
            print(f"{name}: No gradient!!")
    
    # for param in list(model.atlas_net.named_parameters()):
    #     if param[1].grad is not None:
    #         print(f"{param[0]}: Has gradient!!")
    # # Forward pass - Fine-tuning mode
    # print("\nRunning in fine-tuning mode")
    # model.set_pretraining_mode(False)
    # output_dict = model(data_dict)
    
    # # Calculate fine-tuning loss
    # sg_loss, sg_loss_dict = model.scene_graph_loss(output_dict)
    # print(f"Scene graph prediction loss: {sg_loss.item():.4f}")
    # print("Loss components:", sg_loss_dict)