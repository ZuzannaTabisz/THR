
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def mish(input):
    return input * torch.tanh(F.softplus(input))

class SoftCrossEntropyLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction
    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        loss = -(targets * lsm).sum(-1)
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

class MayoSoftCrossEntropyLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction
    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)
        weight = self.weight * inputs.shape[0]
        weight = 1. / weight
        if self.weight is not None:
            lsm = lsm * weight.unsqueeze(0)
        loss = -(targets * lsm).sum() / 2.
        return loss

class MSDPRegHead(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(MSDPRegHead, self).__init__()
        self.dropout1 = nn.Dropout(0.1)  # CFG.fc_dropout not defined, using default
        self.dropout2 = nn.Dropout(0.1)
        self.out = nn.Linear(512, out_feat)
        self.out_hidden = nn.Linear(in_feat, 512)
        torch.nn.init.normal_(self.out.weight, std=0.02)
        torch.nn.init.normal_(self.out_hidden.weight, std=0.02)
    def forward(self, x):
        x = torch.mean(
            torch.stack([mish(self.out_hidden(self.dropout1(x))) for _ in range(5)], dim=0),
            dim=0,
        )
        x = torch.mean(
            torch.stack([self.out(self.dropout2(x)) for _ in range(5)], dim=0),
            dim=0,
        )
        return x

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = x.shape[1]
        eij = torch.mm(x.contiguous().view(-1, feature_dim), self.weight).view(-1, step_dim)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        if mask is not None:
            a = a * mask
        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=False, dropout=0.1):
        super().__init__()
        if cfg.model == 'coat_lite_medium':
            self.model = timm.create_model('coat_lite_medium', pretrained=False)
        else:
            self.model = timm.create_model(cfg.model, pretrained=pretrained)
        self.model_reformat = cfg.model_reformat
        feat_dim = self.model.get_classifier().in_features
        self.norm = nn.LayerNorm(feat_dim)
        self.att = Attention(feat_dim, cfg.num_instance, bias=False)
        self.fc = nn.Sequential(nn.Linear(feat_dim, len(cfg.target_cols)))
        self.model.reset_classifier(num_classes=0, global_pool="avg")
    def forward(self, images):
        b, n, c, w, h = images.shape
        x = images.view(b * n, c, w, h)
        x = self.model.forward_features(x) # For Swin, shape is (b*n, H, W, C), e.g., (64, 12, 12, 1536)
        # Flatten the spatial dimensions (H, W) into a single sequence of tokens
        x = x.flatten(1, 2)                # Shape: (b*n, H*W, C) -> (64, 144, 1536)
        x = x.mean(dim=1)                  # Manually perform global average pooling. Shape: (b*n, C) -> (64, 1536)
        x = x.view(b, n, -1)               # Reshape to (b, n, feature_dim)
        x = self.norm(x)
        x = self.att(x)
        output = self.fc(x)
        return output
