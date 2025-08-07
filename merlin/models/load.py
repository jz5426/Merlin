import os

import torch
from torch import nn

from merlin.models.build import MerlinArchitecture
from merlin.utils import download_file


class Merlin(nn.Module):
    def __init__(self, ImageEmbedding: bool = False):
        super(Merlin, self).__init__()
        assert ImageEmbedding == False

        self.ImageEmbedding = ImageEmbedding
        self.checkpoint_parent_path = '/cluster/projects/mcintoshgroup/publicData/'
        self.local_dir = os.path.join(self.checkpoint_parent_path, "merlin_checkpoint")
        self.checkpoint_name = (
            "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"
        )
        self.repo_id = "stanfordmimi/Merlin"
        self.model = self._load_model()

    """
    Load the Merlin model with the initialized weights
    """

    def _load_model(self):
        model = MerlinArchitecture(ImageEmbedding=self.ImageEmbedding)
        state_dict = torch.load(os.path.join(self.local_dir, self.checkpoint_name))
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        return model

    def forward(self, *input):
        return self.model(*input)
