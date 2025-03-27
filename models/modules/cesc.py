import torch.nn as nn
import torch.nn.functional as F


class ContextEnhancedGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x, global_context):
        x_normed = self.group_norm(x)
        return x_normed + global_context.expand_as(x)


class CESC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.sparse_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        # self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.ce_gn = ContextEnhancedGroupNorm(out_channels)

    def forward(self, x, mask, global_context):
        # Apply binary mask (simulate sparse convolution)
        x_sparse = x * mask

        # Sparse conv (simulated)
        feat = self.sparse_conv(x_sparse)

        # # Step 2: Global context
        # global_context = F.adaptive_avg_pool2d(x, (1, 1))
        # global_context = self.pointwise_conv(global_context)

        # Step 3: CE-GN normalization
        out = self.ce_gn(feat, global_context)
        return out
