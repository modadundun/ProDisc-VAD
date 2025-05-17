
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import torch.nn.init as torch_init
import numpy as np 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class PrototypeInteractionLayer(nn.Module):
    def __init__(self, feature_dim=512, num_prototypes=5, dropout_rate=0.1):
        super().__init__()
        self.normal_prototypes_keys = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.normal_prototypes_values = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.scale = feature_dim ** -0.5
        self.norm = nn.LayerNorm(feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        
        torch_init.normal_(self.normal_prototypes_keys, std=0.02)
        torch_init.normal_(self.normal_prototypes_values, std=0.02)

    def forward(self, features):
        B, T, D = features.shape
        N = self.normal_prototypes_keys.shape[0]
    
        attn_logits = torch.matmul(features, self.normal_prototypes_keys.t()) * self.scale 
        attn_weights = F.softmax(attn_logits, dim=-1) 
        normality_context = torch.matmul(attn_weights, self.normal_prototypes_values)  
        combined_features = 10*features + 5*normality_context
        normed_features = self.norm(combined_features)
        projected_features = self.output_proj(normed_features) 
        activated_features = self.activation(projected_features) 
        output_features = self.dropout(activated_features)
        return output_features

class PseudoInstanceDiscriminativeEnhancement(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, element_scores, seq_len):
        batch_size, max_seq_len, feature_dim = features.shape
        pseudo_labels = torch.zeros(batch_size, max_seq_len, device=features.device) 
        for i in range(batch_size):
            valid_len = seq_len[i].item()
            if valid_len > 0:
                valid_scores = element_scores[i, :valid_len, 0] 
                max_idx = torch.argmax(valid_scores)
                min_idx = torch.argmin(valid_scores)
                pseudo_labels[i, max_idx] = 1
                pseudo_labels[i, min_idx] = -1

        normalized_features = F.normalize(features, p=2, dim=-1) 
        sim_matrix = torch.matmul(normalized_features, normalized_features.transpose(-2, -1)) / self.temperature
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=-1, keepdim=True) + 1e-8) 
        
        mask = pseudo_labels.unsqueeze(1) * pseudo_labels.unsqueeze(2) 
        mask_positive_pairs = (mask > 0).float() 

        loss = - (mask_positive_pairs * log_prob).sum() / (mask_positive_pairs.sum() + 1e-8)
        return loss


class ProDisc_VAD(nn.Module):
    def __init__(self, feature_size=512):
        super().__init__()
        self.prototype_layer = PrototypeInteractionLayer(feature_dim=feature_size)
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1)  
        )
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5) 
        self.pide = PseudoInstanceDiscriminativeEnhancement()
        self.apply(weights_init)

    def forward(self, x, seq_len=None, is_training=True):
        processed_features = self.prototype_layer(x) 
        visual_features = processed_features 
        raw_logits = self.classifier(visual_features) 
        element_scores = self.sigmoid(raw_logits) 
        if is_training:
            if seq_len is None:
                seq_len = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
            pide_loss = self.pide(visual_features, element_scores, seq_len)
            return raw_logits, element_scores, visual_features, pide_loss
        else:
            return raw_logits, element_scores, visual_features



