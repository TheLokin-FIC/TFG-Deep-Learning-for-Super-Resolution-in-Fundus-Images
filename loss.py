import torch.nn.functional as F

from torch import nn
from torchvision.models import vgg19


class ContentLoss(nn.Module):
    """Where VGG5.4 represents the feature map of 34th layer in pretrained VGG19 model.
    '"Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    <https://arxiv.org/pdf/1609.04802.pdf>'. A loss defined on feature maps of higher level
    features from deeper network layers with more potential to focus on the content of the
    images. We refer to this network as SRGAN in the following.
    """

    def __init__(self, feature_layer=36):
        """Constructing characteristic loss function of VGG network. For VGG5.4 layer.
        For VGG2.2 use 9th layer.
        Args:
            feature_layer (int): How many layers in VGG19. (Default:36).
        Notes:
            features(
              (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): ReLU(inplace=True)
              (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (3): ReLU(inplace=True)
              (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (6): ReLU(inplace=True)
              (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (8): ReLU(inplace=True)
              (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (11): ReLU(inplace=True)
              (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (13): ReLU(inplace=True)
              (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (15): ReLU(inplace=True)
              (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (17): ReLU(inplace=True)
              (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (20): ReLU(inplace=True)
              (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (22): ReLU(inplace=True)
              (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (24): ReLU(inplace=True)
              (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (26): ReLU(inplace=True)
              (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (29): ReLU(inplace=True)
              (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (31): ReLU(inplace=True)
              (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (33): ReLU(inplace=True)
              (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (35): ReLU(inplace=True)
            )
        """

        super(ContentLoss, self).__init__()

        model = vgg19(pretrained=True)
        self.features = nn.Sequential(
            *list(model.features.children())[:feature_layer]).eval()

        # Freeze parameters (don't train)
        for _, param in self.features.named_parameters():
            param.requires_grad = False

    def forward(self, input, target):
        # We then define the VGG loss as the euclidean distance between the feature representations
        # of a reconstructed image G(LR) and the reference image I(HR)

        return F.mse_loss(self.features(input), self.features(target))
