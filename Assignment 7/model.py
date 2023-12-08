import torch
import torch.nn as nn
import torchvision

# module to normalize output (from last session)
class NormLayer(nn.Module):
    """ Layer that computes embedding normalization """
    def __init__(self, l=2):
        """ Layer initializer """
        assert l in [1, 2]
        super().__init__()
        self.l = l
        return

    def forward(self, x):
        """ Normalizing embeddings x. The shape of x is (B,D) """
        x_normalized = x / torch.norm(x, p=self.l, dim=-1, keepdim=True)
        return x_normalized

# model fror Siamese Model based on resnet18
class AdpSiameseModel(nn.Module):
    def __init__(self, emb_dim=32, pretrained = True, fine_tune = False): # pretrain and fine tune describe if resnet18 is trained itselfe, and if it uses random weights
        super().__init__()
        self.emb_dim = emb_dim

        if pretrained:
          self.resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT) # newest weights for resnet18
        else:
          self.resnet18 = torchvision.models.resnet18()

        if fine_tune:
          for param in self.resnet18.parameters():
            param.requires_grad = False

        # fully conected layers in the end. (512 as hidden dimension)
        fc_layers = nn.Sequential(  
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features= 512, out_features = self.emb_dim)
        )

        self.resnet18.fc = fc_layers

        self.norm = NormLayer()

        return

    def forward_one(self, x):
        # forward through resnet18 and linear layer and normalization
        x = self.resnet18(x)
        x_emb_norm = self.norm(x)
        return x_emb_norm

    def forward(self, anchor, positive, negative):
        # forwarding triplet through resnet and linear layer
        imgs = torch.concat([anchor, positive, negative], dim=0)
        embs = self.forward_one(imgs)
        anchor_emb, positive_emb, negative_emb = torch.chunk(embs, 3, dim=0)

        return anchor_emb, positive_emb, negative_emb