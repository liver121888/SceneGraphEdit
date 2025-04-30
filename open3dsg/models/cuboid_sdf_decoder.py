import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImplicitCuboidDecoder(nn.Module):
    """
    Decoder that implicitly represents cuboid shapes.
    
    This decoder is specialized for furniture with flat surfaces (walls, tables, etc.)
    where spherical harmonics would struggle to create precise flat surfaces and sharp edges.
    It uses a signed distance function (SDF) approach to represent cuboids.
    """
    def __init__(self, latent_dim=256, num_points=1024, smooth_factor=0.01):
        """
        Initialize the implicit cuboid decoder.
        
        Args:
            latent_dim: Dimension of the input latent code
            num_points: Number of points to sample on the cuboid surface
            smooth_factor: Factor to control the smoothness of corners (0 = sharp, larger = smoother)
        """
        super(ImplicitCuboidDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.smooth_factor = smooth_factor
        
        # MLP to predict cuboid parameters from latent code
        self.params_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 9)  # 3 (dimensions) + 6 (rotation as 3×3 matrix with 6 DoF)
        )
        
        # Generate initial sampling points
        self.register_buffer('initial_samples', self._generate_initial_samples())
    
    def _generate_initial_samples(self):
        """
        Generate initial sampling points around a unit cube.
        These points will be transformed to create the final cuboid.
        
        Returns:
            points: Tensor of shape [num_points, 3]
        """
        # Create a distribution of points focused near the surface of a unit cube
        # We'll generate more points than needed and select the best ones
        num_extra_points = self.num_points * 8
        
        # Generate random points with higher density near the faces
        points = []
        
        # Points distributed on the 6 faces
        for axis in range(3):
            for sign in [-1, 1]:
                face_points = torch.rand(num_extra_points // 6, 3)
                face_points[:, axis] = sign * 0.5  # Place on the face
                points.append(face_points)
                
        points = torch.cat(points, dim=0)
        
        # Add random perturbations
        perturbations = (torch.rand_like(points) * 0.2) - 0.1
        points = points + perturbations
        
        # Add some random interior and exterior points
        interior_points = (torch.rand(num_extra_points // 4, 3) * 0.8) - 0.4
        exterior_points = (torch.rand(num_extra_points // 4, 3) * 0.8) - 0.4
        exterior_points = exterior_points / torch.norm(exterior_points, dim=1, keepdim=True) * 0.7
        
        all_points = torch.cat([points, interior_points, exterior_points], dim=0)
        
        # Ensure we have enough unique points
        if all_points.shape[0] < self.num_points:
            additional_points = torch.rand(self.num_points - all_points.shape[0], 3) * 2 - 1
            all_points = torch.cat([all_points, additional_points], dim=0)
        
        return all_points
    
    def _compute_sdf(self, points, dimensions):
        """
        Compute the signed distance function for a cuboid.
        
        Args:
            points: Tensor of shape [batch_size, num_points, 3]
            dimensions: Tensor of shape [batch_size, 3] containing half-dimensions
            
        Returns:
            sdf: Tensor of shape [batch_size, num_points] - distance to the surface
                 negative inside, positive outside, zero on the surface
        """
        # Calculate distance from point to closest face in each dimension
        distance_to_face = torch.abs(points) - dimensions.unsqueeze(1)
        
        # Outside the cuboid: Euclidean distance to the closest point on the cuboid
        outside_distance = torch.norm(
            torch.maximum(distance_to_face, torch.zeros_like(distance_to_face)), 
            dim=2
        )
        
        # Inside the cuboid: Negative of the distance to the closest face
        inside_distance = torch.minimum(
            torch.max(distance_to_face, dim=2)[0], 
            torch.zeros_like(distance_to_face[:, :, 0])
        )
        
        # Combine inside and outside distances
        sdf = inside_distance + outside_distance
        
        return sdf
    
    def _create_rotation_matrix(self, rot_params):
        """
        Create a valid rotation matrix from the predicted parameters.
        Uses a 6D representation of 3D rotations for stable optimization.
        
        Args:
            rot_params: Tensor of shape [batch_size, 6]
            
        Returns:
            rotation_matrix: Tensor of shape [batch_size, 3, 3]
        """
        batch_size = rot_params.shape[0]
        
        # Create two vectors from the parameters
        a1, a2 = rot_params[:, :3], rot_params[:, 3:6]
        
        # Normalize the first vector
        b1 = F.normalize(a1, dim=1)
        
        # Make the second vector orthogonal to the first
        b2 = a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=1)
        
        # The third vector is the cross product of the first two
        b3 = torch.cross(b1, b2, dim=1)
        
        # Stack to form rotation matrix
        matrix = torch.stack([b1, b2, b3], dim=2)
        
        return matrix
    
    def _sample_surface_points(self, sdf, points, dimensions, smooth_factor, num_points):
        """
        Sample points on or near the surface of the cuboid.
        
        Args:
            sdf: Tensor of shape [batch_size, num_sample_points] - SDF values
            points: Tensor of shape [batch_size, num_sample_points, 3] - Candidate points
            dimensions: Tensor of shape [batch_size, 3] - Cuboid dimensions
            smooth_factor: Smoothing factor for the SDF
            num_points: Number of points to sample
            
        Returns:
            surface_points: Tensor of shape [batch_size, num_points, 3]
        """
        batch_size = sdf.shape[0]
        surface_points = []
        
        # Select points close to the surface
        surface_dist = torch.abs(sdf - smooth_factor)
        
        for b in range(batch_size):
            # Sort by distance to the surface
            _, indices = torch.sort(surface_dist[b])
            
            # Keep the top num_points
            selected_indices = indices[:num_points]
            selected_points = points[b, selected_indices]
            
            surface_points.append(selected_points)
        
        return torch.stack(surface_points)
    
    def forward(self, latent_code):
        """
        Decode the latent code into a 3D point cloud representing a cuboid.
        
        Args:
            latent_code: Tensor of shape [batch_size, latent_dim]
            
        Returns:
            point_cloud: Tensor of shape [batch_size, num_points, 3]
        """
        batch_size = latent_code.shape[0]
        
        # Predict cuboid parameters
        params = self.params_mlp(latent_code)
        
        # Extract parameters
        dimensions = F.softplus(params[:, :3])  # Ensure positive dimensions
        rot_params = params[:, 3:9]  # Rotation parameters
        
        # Create rotation matrix
        rotation_matrix = self._create_rotation_matrix(rot_params)
        
        # Get the initial sample points
        initial_samples = self.initial_samples
        
        # Expand to batch dimension
        points = initial_samples.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute SDF for these points
        sdf = self._compute_sdf(points, dimensions.unsqueeze(1))
        
        # Sample points near the surface
        surface_points = self._sample_surface_points(
            sdf, points, dimensions, self.smooth_factor, self.num_points
        )
        
        # Apply the rotation matrix
        rotated_points = torch.bmm(surface_points, rotation_matrix)
        
        return rotated_points
