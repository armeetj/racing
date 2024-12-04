from typing import Tuple
from ultralytics import YOLO
import torch
from torch import nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


DINO_EMBEDDING_DIM = 384
DINO_PATCH_DIM = 14


class DINOVisionModel(nn.Module):
    def __init__(
        self,
        img_dim: Tuple[int, int],
        hidden_features: int = 1000,
        freeze_dino: bool = True,
    ):
        super().__init__()

        # metadata
        self.id = "dinovision-v0.0"

        # properties
        self.img_dim = img_dim
        self.freeze_dino = freeze_dino

        # DINOv2
        self.dino: nn.Module = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14_reg"
        )  # type: ignore
        if self.freeze_dino:
            self.dino.eval()
        # hook
        self.hook_handle = self.dino.patch_embed.register_forward_hook(self.hook)
        self.patch_embeddings = None

    def state_dict(self, *args, **kwargs):
        original_state_dict = super().state_dict(*args, **kwargs)
        filtered_state_dict = {
            k: v for k, v in original_state_dict.items() if "dino" not in k
        }
        return filtered_state_dict

    def hook(self, _, __, output):
        self.patch_embeddings = output

    def forward(self, img):
        if img.dim() == 3:
            # add batch dim (C, H, W) -> (1, C, H, W)
            img = img.unsqueeze(0)
        # assert img.shape[1:] == (
        #     3,
        #     *self.img_dim,
        # ), "Image shape must be (N, C, H, W) with C=3 and matching img_dim"

        N, C, H, W = img.shape

        # image processing
        self.dino(img)
        patches = self.patch_embeddings
        assert patches is not None, "patch embeddings must exist"
        N, num_patches, embedding_dim = patches.shape
        dim_height = self.img_dim[0] // DINO_PATCH_DIM
        dim_width = self.img_dim[1] // DINO_PATCH_DIM
        assert (
            dim_height * dim_width == num_patches
        ), f"Number of reshaped patches {dim_height}x{dim_width}={dim_height * dim_width} does not match actual number of patches: {num_patches}"
        patches = patches.reshape((N, dim_height, dim_width, embedding_dim)).permute(
            0, 3, 1, 2
        )

        return patches


class YOLOVisionModel(nn.Module):
    def __init__(
        self,
        img_dim: Tuple[int, int],
        out_features: int,
        freeze_yolo: bool = False,
    ):
        super().__init__()

        raise NotImplementedError("Not fully implemented.")

        # metadata
        self.id = "yolovision-v0.0"

        # properties
        self.img_dim = img_dim
        self.out_features = out_features

        # nn
        self.yolo = YOLO("weights/yolo11n-seg.pt", verbose=False)
        if self.freeze_yolo:
            for param in self.yolo.parameters():
                param.requires_grad = False
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, out_features)

    def forward(self, img):
        if img.dim() == 3:
            # add batch dim (C, H, W) -> (1, C, H, W)
            img = img.unsqueeze(0)
        assert img.shape[1:] == (
            3,
            *self.img_dim,
        ), "Image shape must be (N, C, H, W) with C=3 and matching img_dim"

        N, C, H, W = img.shape

        # image processing
        yolo_res = self.yolo(img)

        combined_mask = torch.zeros(self.shape)
        for res in yolo_res:
            mask = res.masks
            if mask is None:
                continue

        i = self.cnn(seg_mask)

        # feature vector
        x = torch.flatten(i)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


# debugging purposes
if __name__ == "__main__":
    dims = (126, 126)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print("-------------------------------------------------------------------")
    model = DINOVisionModel(dims, 10).to(device)
    x = torch.rand(1, 3, *dims).to(device)
    out = model(x)
    print(out.shape)
