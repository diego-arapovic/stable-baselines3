import flax.linen as fnn
import jax.numpy as jnp
import distrax

class SpatialSoftmax(fnn.Module):
    """
    Converts HxWxC feature maps into C*2 (x,y) coordinates.
    Used widely in Deep RL (Levine et al.) to preserve spatial geometry.
    """
    height: int
    width: int
    temperature: float = 1.0 # Controls sharpness of the softmax

    def setup(self):
        # Create coordinate grids
        # y_coords: [[0, 0...], [1, 1...]] normalized to [-1, 1]
        pos_y, pos_x = jnp.meshgrid(
            jnp.linspace(-1., 1., self.height),
            jnp.linspace(-1., 1., self.width),
            indexing='ij'
        )
        self.pos_x = pos_x.reshape(-1)
        self.pos_y = pos_y.reshape(-1)

    def __call__(self, x):
        # x shape: (Batch, H, W, C)
        B, H, W, C = x.shape
        
        # Flatten spatial dims: (B, H*W, C)
        flat_x = x.reshape(B, H * W, C)
        
        # Softmax over spatial dimensions
        softmax_attention = fnn.softmax(flat_x / self.temperature, axis=1)
        
        # Calculate expected coordinates
        # (B, H*W, C) * (H*W, 1) -> sum -> (B, C)
        expected_x = jnp.sum(softmax_attention * self.pos_x[:, None], axis=1)
        expected_y = jnp.sum(softmax_attention * self.pos_y[:, None], axis=1)
        
        # Stack coordinates: (B, C, 2) -> Flatten to (B, C*2)
        keypoints = jnp.stack([expected_x, expected_y], axis=-1)
        return keypoints.reshape(B, -1)
