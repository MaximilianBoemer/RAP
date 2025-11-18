import torch
import torch.nn as nn
from .encoder import BEVFormerEncoder
# from mmdet.models.layers.positional_encoding import LearnedPositionalEncoding
from .decoder import CustomMSDeformableAttention
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D


# Added here for compatibility
class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Defaults to 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Defaults to 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_feats: int,
        row_num_embed: int = 50,
        col_num_embed: int = 50,
        init_cfg = dict(type='Uniform', layer='Embedding')
    ) -> None:
        # super().__init__(init_cfg=init_cfg) # REMOVED AS CHANGED TO nn.Module
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos

    def __repr__(self) -> str:
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str


class Bev_refiner(nn.Module):
    def __init__(self, config,bev_h,bev_w,proposal_query):
        super().__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pose_dim=3

        d_model = config.tf_d_model
        d_ffn = config.tf_d_ffn

        self.proposal_query=proposal_query

        if self.proposal_query:
            num_points=config.num_points_in_pillar*4
           # self.in_proj = nn.Linear(self.pose_dim, d_model)
        else:
            num_points=8
           # self.in_proj = nn.Embedding(self.bev_h*self.bev_w, d_model)

        self.positional_encoding = LearnedPositionalEncoding(num_feats=d_model // 2, row_num_embed=self.bev_h,
                                                             col_num_embed=self.bev_w)


        _num_levels_ = 1
        num_cams = 4

        num_layers = config.num_bev_layers

        half_length = config.half_length
        half_width = config.half_width
        rear_axle_to_center = config.rear_axle_to_center
        lidar_height=config.lidar_height

        self.bev_decoder = BEVFormerEncoder(
            bev_w=self.bev_w,
            bev_h=self.bev_h,
            num_layers=num_layers,
            pc_range=config.point_cloud_range,
            num_points_in_pillar=config.num_points_in_pillar,
            half_length=half_length,
            half_width=half_width,
            rear_axle_to_center=rear_axle_to_center,
            lidar_height=lidar_height,
            return_intermediate=False,
            transformerlayers=dict(
                type='BEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TemporalSelfAttention',
                        embed_dims=d_model,
                        num_levels=1,
                        dropout=config.tf_dropout,
                        proposal_query=proposal_query,
                        config=config
                       ),
                    dict(
                        type='SpatialCrossAttention',
                        num_cams=num_cams,
                        pc_range=config.point_cloud_range,
                        dropout=config.tf_dropout,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=d_model,
                            num_points=num_points,
                            num_levels=_num_levels_),
                        embed_dims=d_model,
                    )
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=d_model,
                    feedforward_channels=config.tf_d_ffn,
                    num_fcs=2,
                    ffn_drop=config.tf_dropout,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                feedforward_channels=d_ffn,
                ffn_dropout=config.tf_dropout,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm')),
        )

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    def forward(self, pose,prev_bev,image_feature):
        img=image_feature[0]
        batch_size = img.shape[2]
        bev_queries = prev_bev  # self.in_proj(pose)

        if self.proposal_query:
            pose=pose.reshape(batch_size, -1, self.pose_dim)
            ref_2d =pose.detach()
        else:
            ref_2d= None
            # bev_queries= self.in_proj.weight[None].repeat(batch_size, 1, 1)

        bev_mask = torch.zeros((batch_size, self.bev_h, self.bev_w), device=img.device).to(img.dtype)
        bev_pos = self.positional_encoding(bev_mask).to(img.dtype)  # 256,100,100
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1)  # len,bs,256

        feat_flatten, spatial_shapes, level_start_index, kwargs = image_feature

        #kwargs['img_metas']["prev_bev"] = prev_bev

        bev_feature = self.bev_decoder(
            bev_queries.permute(1, 0, 2),
            feat_flatten,
            feat_flatten,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            bev_pos=bev_pos.permute(1, 0, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            ref_2d=ref_2d,
            # prev_bev=keyval,
            # shift=shift,
            **kwargs
        )

        return bev_feature