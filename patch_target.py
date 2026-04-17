import torch
import torch.nn as nn
from typing import Optional

def add_generator():
    with open("src/target_generators.py", "a") as f:
        f.write('''
class V19LabelConditionedTextureGenerator(BaseTargetGenerator):
    """
    V19 Stochastic Semantic Decoupling: Merges geometric label-priors with
    texture-preserving latent space.
    """
    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C, D, H, W = images.shape
        device = images.device
        dtype = images.dtype

        # Step A: Base v18_6 Background Synthesis
        mask = images > 0.01
        
        y = images.clone()
        
        # Subsample to compute K=8 quantile edges
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()
            
            # Generate random base targets
            mu_base = _shared_rand((B, 8), device=device, dtype=torch.float32)
            alpha_base = _shared_rand((B, 8), device=device, dtype=torch.float32) * 1.5 + 0.5
            
            q_edges = torch.linspace(0, 1, 9, device=device)
            
            # Bucketize
            c_i = torch.bucketize(images_f, q_edges) - 1
            c_i = torch.clamp(c_i, 0, 7)
            
            mu_c = mu_base.view(B, 8, 1, 1, 1).gather(1, c_i)
            alpha_c = alpha_base.view(B, 8, 1, 1, 1).gather(1, c_i)
            q_c_minus_1 = q_edges[:-1].view(1, 8, 1, 1, 1).gather(1, c_i)
            
            y_base = mu_c + alpha_c * (images_f - q_c_minus_1)
            y = torch.where(mask, y_base.to(dtype), y)

        # Step B: Stochastic Semantic Decoupling
        if labels is not None:
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
                y_f = y.float()
                images_f = images.float()
                
                for c in [1, 2, 3]:
                    # DDP-safe boolean decoupling mask per batch item
                    decouple = _shared_rand((B, 1, 1, 1, 1), device=device, dtype=torch.float32) > 0.5
                    
                    mu_path = _shared_rand((B, 1, 1, 1, 1), device=device, dtype=torch.float32)
                    alpha_path = _shared_rand((B, 1, 1, 1, 1), device=device, dtype=torch.float32) * 1.5 + 0.5
                    
                    class_mask = (labels == c)
                    
                    # Calculate mean intensity of class voxels per batch item
                    # Use safe division
                    class_sum = (images_f * class_mask).sum(dim=(1, 2, 3, 4), keepdim=True)
                    class_count = class_mask.sum(dim=(1, 2, 3, 4), keepdim=True)
                    class_count_safe = torch.clamp(class_count, min=1.0)
                    mean_c = class_sum / class_count_safe
                    
                    y_override = mu_path + alpha_path * (images_f - mean_c)
                    
                    # Apply override conditionally
                    valid_override = class_mask & decouple & (class_count > 0)
                    y_f = torch.where(valid_override, y_override, y_f)
                    
                y = y_f.to(dtype)
                
        # Step C: Masking & Clamping
        y = torch.clamp(y, 0.0, 1.0)
        y = torch.where(mask, y, torch.zeros_like(y))
        
        target_hist = hist_module(y)
        return target_hist, y, y
''')
add_generator()
