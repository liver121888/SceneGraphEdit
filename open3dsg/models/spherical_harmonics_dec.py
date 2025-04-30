import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SphericalHarmonicsDecoder(nn.Module):
    """
    Decoder that uses spherical harmonics to generate 3D shapes from latent codes.
    This replaces AtlasNet in the SGRec3D architecture.
    """
    def __init__(self, latent_dim=1024, max_degree=4, num_points=1024):
        """
        Initialize the spherical harmonics decoder.
        
        Args:
            latent_dim: Dimension of the input latent code
            max_degree: Maximum degree of spherical harmonics
            num_points: Number of points to sample on the sphere
        """
        super(SphericalHarmonicsDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.max_degree = max_degree
        self.num_points = num_points
        
        # Number of spherical harmonics coefficients: (max_degree+1)^2
        # For max_degree=4, this gives 25 coefficients
        num_coeffs = (max_degree + 1) ** 2
        
        # For each x, y, z coordinate, we need to predict num_coeffs
        # So total coefficients to predict is 3 * num_coeffs
        self.total_coeffs = 3 * num_coeffs
        
        # MLP to predict the spherical harmonics coefficients from the latent code
        self.coefficient_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.total_coeffs)
        )
        
        # Pre-compute the initial sphere points for faster inference
        self.register_buffer('sphere_points', self._generate_sphere_points(num_points))
        
    def _generate_sphere_points(self, num_points):
        """
        Generate uniformly distributed points on a unit sphere.
        We use Fibonacci sphere algorithm for uniform distribution.
        
        Args:
            num_points: Number of points to generate
            
        Returns:
            points: Tensor of shape [num_points, 3] containing the points on the unit sphere
        """
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # Golden angle in radians
        
        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # Golden angle increment
            
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            
            points.append([x, y, z])
            
        return torch.tensor(points, dtype=torch.float32)
    
    def _evaluate_spherical_harmonics(self, points, coeffs):
        """
        Evaluate the spherical harmonics function at the given points.
        
        Args:
            points: Tensor of shape [batch_size, num_points, 3] containing the points on the unit sphere
            coeffs: Tensor of shape [batch_size, (max_degree+1)^2 * 3] containing the spherical harmonics coefficients
            
        Returns:
            deformed_points: Tensor of shape [batch_size, num_points, 3] containing the deformed points
        """
        batch_size = coeffs.shape[0]
        num_points = points.shape[0]
        num_coeffs = (self.max_degree + 1) ** 2
        
        # Reshape coefficients to [batch_size, 3, num_coeffs]
        coeffs = coeffs.view(batch_size, 3, num_coeffs)
        
        # Convert points from Cartesian to spherical coordinates (r, theta, phi)
        # Note: r = 1 for all points since we're on a unit sphere
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.acos(z / r)
        phi = torch.atan2(y, x)
        
        # Evaluate spherical harmonics for each point
        Y = []
        
        for l in range(self.max_degree + 1):
            for m in range(-l, l + 1):
                # Evaluate spherical harmonic Y_l^m at (theta, phi)
                # This is a simplified approximation - a full implementation would use
                # associated Legendre polynomials and complex exponentials
                if m == 0:
                    # For m=0, the spherical harmonic is just the Legendre polynomial
                    Y_lm = torch.ones_like(theta)  # Simplified
                else:
                    # For m!=0, we include angular dependencies
                    Y_lm = torch.sin(theta * l) * torch.cos(phi * m)  # Simplified
                
                Y.append(Y_lm)
        
        Y = torch.stack(Y, dim=1)  # [num_points, num_coeffs]
        
        # Expand Y to match batch size
        Y = Y.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_points, num_coeffs]
        
        # Apply coefficients to get offsets in x, y, z
        # [batch_size, num_points, 3]
        offsets = torch.bmm(Y, coeffs.transpose(1, 2))
        
        # Apply offsets to the original points on the unit sphere
        deformed_points = points.unsqueeze(0).expand(batch_size, -1, -1) + offsets
        
        return deformed_points
    
    def forward(self, latent_code):
        """
        Decode the latent code into a 3D point cloud.
        
        Args:
            latent_code: Tensor of shape [batch_size, latent_dim] containing the latent code
            
        Returns:
            point_cloud: Tensor of shape [batch_size, num_points, 3] containing the generated point cloud
        """
        batch_size = latent_code.shape[0]
        
        # Predict spherical harmonics coefficients
        coeffs = self.coefficient_mlp(latent_code)  # [batch_size, 3 * (max_degree+1)^2]
        
        # Apply spherical harmonics to deform the sphere
        sphere_points = self.sphere_points.to(latent_code.device)
        deformed_points = self._evaluate_spherical_harmonics(sphere_points, coeffs)
        
        return deformed_points

if __name__ == "__main__":
    # Example usage
    latent_dim = 1024
    max_degree = 4
    num_points = 1500
    
    decoder = SphericalHarmonicsDecoder(latent_dim, max_degree, num_points)
    
    # Generate a random latent code
    latent_code = torch.randn(8, latent_dim)  # Batch size of 8
    
    # Decode the latent code into a point cloud
    point_cloud = decoder(latent_code)
    
    print("Generated point cloud shape:", point_cloud.shape)  # Should be [8, 1024, 3]