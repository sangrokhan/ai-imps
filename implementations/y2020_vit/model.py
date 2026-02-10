import torch
import torch.nn as nn
from core.base_model import BaseModel
from core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("ViT")
class VisionTransformer(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.img_size = config.get("img_size", 32)
        self.patch_size = config.get("patch_size", 4)
        self.in_channels = config.get("in_channels", 3)
        self.num_classes = config.get("num_classes", 100)
        self.embed_dim = config.get("embed_dim", 256)
        self.depth = config.get("depth", 6)
        self.num_heads = config.get("num_heads", 8)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.drop_rate = config.get("drop_rate", 0.1)
        
        # Patch Embedding
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            self.in_channels, self.embed_dim, 
            kernel_size=self.patch_size, stride=self.patch_size
        )
        
        # Class Token & Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_rate)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.embed_dim * self.mlp_ratio),
            dropout=self.drop_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)
        self.norm = nn.LayerNorm(self.embed_dim)
        
        # Head
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        
        # Init weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights_layer)

    def _init_weights_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, C, H, W]
        B = x.shape[0]
        
        # Patch Embedding: [B, C, H, W] -> [B, E, H/P, W/P] -> [B, E, N] -> [B, N, E]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Positional Embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer Encoder
        x = self.blocks(x)
        x = self.norm(x)
        
        # Classification (use CLS token)
        cls_out = x[:, 0]
        return self.head(cls_out)

    def compute_loss(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)
