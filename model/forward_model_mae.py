import torch
import torch.nn as nn
import timm

class MAE_Translation(nn.Module):
    def __init__(self, pretrained_mae, embed_dim=768, out_dim=3, freeze_encoder=False, frozen_layer_num = 6):
        super().__init__()
        self.encoder = pretrained_mae.vit  # ViTMAEForPreTraining encoder

        for i, layer in enumerate(self.encoder.encoder.layer):
            freeze = i < frozen_layer_num
            for p in layer.parameters():
                p.requires_grad = not freeze  # True only for trainable layers

        # if freeze_encoder:
        #     for p in self.encoder.parameters():
        #         p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_dim)
        )

    def forward(self, imgs):
        imgs = imgs.squeeze(1).contiguous()
        # import pdb; pdb.set_trace()
        outputs = self.encoder(imgs)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.head(pooled)

    # def forward(self, imgs):
    #     if imgs.dim() == 5:
    #         x = imgs.squeeze(1)
    #     else:
    #         x = imgs
    #     assert x.dim() == 4, f"Expected [B, 3, H, W], got {x.shape}"
    #     x = x.contiguous()

    #     x = self.encoder.embeddings(x)
    #     if isinstance(x, tuple):
    #         x = x[0]

    #     for i, layer in enumerate(self.encoder.encoder.layer):
    #         #import pdb; pdb.set_trace()
    #         if i < self.N:
    #             x = layer(x)
    #         else:
    #             x = layer(x)

    #     x = self.encoder.layernorm(x)
    #     pooled = x.mean(dim=1)
    #     return self.head(pooled)

    # def forward(self, imgs):
    #     # imgs: B, 3, H, W
    #     outputs = self.encoder(imgs.squeeze())
    #     last_hidden = outputs.last_hidden_state  # (B, N, D)
    #     pooled = last_hidden.mean(dim=1) # mean pooling
    #     # x = x[:, 0, :] # cls
    #     return self.head(pooled)

class MAE_Translation_gelu(nn.Module):
    def __init__(self, pretrained_mae, embed_dim=768, out_dim=3, freeze_encoder=False, frozen_layer_num=6):
        super().__init__()
        self.encoder = pretrained_mae.vit  # ViTMAEForPreTraining encoder

        for i, layer in enumerate(self.encoder.encoder.layer):
            freeze = i < frozen_layer_num
            for p in layer.parameters():
                p.requires_grad = not freeze  # True only for trainable layers

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, out_dim)
        )

    def forward(self, imgs):
        imgs = imgs.squeeze(1).contiguous()
        # import pdb; pdb.set_trace()
        outputs = self.encoder(imgs)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        global_features = torch.mean(patch_tokens, dim=1)

        normalized_features = self.norm(global_features)
        action_pred = self.head(normalized_features)
        return action_pred