backbone = dict(
    type="ResNet",
    depth=18,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type="BN", requires_grad=True),
    norm_eval=True,
    style="pytorch",
    init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet18"),
)

neck = dict(type="FPN", in_channels=[64, 128, 256, 512], out_channels=256, num_outs=5)
