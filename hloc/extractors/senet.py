import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path

from ..utils.base_model import BaseModel

import third_party.SENet.core.checkpoint as checkpoint
from third_party.SENet.model.SENet_model import SENet as OriginalSENet


class SENet(BaseModel):

    checkpoint_urls = {"SENet_R50_con": "https://data.ciirc.cvut.cz/public/projects/2020ARTwin/models/senet_weights/SENet_R50_con.pyth"}

    def _init(self, config, device=None):
        if config["model_name"] not in self.checkpoint_urls:
            raise ValueError(
                f'{config["model_name"]} not in {self.checkpoint_urls.keys()}.'
            )    
        checkpoint_path = Path("/app/third_party/SENet/weights") / (config["model_name"] + ".pyth")
        if not checkpoint_path.exists():
            checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            url = self.checkpoint_urls[config["model_name"]]
            torch.hub.download_url_to_file(url, checkpoint_path)

        self.model = OriginalSENet(config['resnet_size'])
        self.device = device
        self.config = config
        self.scale_list = config['scale_list']
        checkpoint.load_checkpoint(str(checkpoint_path), self.model)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def _forward(self, data):
        image = data['image']
        img_feats = [[] for _ in self.scale_list]

        for idx, scale in enumerate(self.scale_list):
            _, _, height, width = image.shape
            new_height = int(height * scale)
            new_width = int(width * scale)
            image_tensor = F.interpolate(image, size=(new_height, new_width), mode='bilinear', align_corners=False)

            with torch.no_grad():
                desc = self.model(image_tensor)

            if len(desc.shape) == 1:
                desc.unsqueeze_(0)
            desc = F.normalize(desc, p=2, dim=1)
            img_feats[idx].append(desc.detach().cpu())

        # Aggregate across scales
        for idx in range(len(img_feats)):
            img_feats[idx] = torch.cat(img_feats[idx], dim=0)
            if len(img_feats[idx].shape) == 1:
                img_feats[idx].unsqueeze_(0)

        img_feats_agg = F.normalize(
            torch.mean(
                torch.cat([img_feat.unsqueeze(0) for img_feat in img_feats], dim=0),
                dim=0,
            ),
            p=2,
            dim=1,
        )
        return {"global_descriptor": img_feats_agg}

