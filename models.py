# models.py

import torch
import torch.nn as nn
from torchvision import models, transforms

class ACTPolicy(nn.Module):
    def __init__(self, input_image_size=(224, 224), num_joints=6, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1, max_sequence_length=100):
        super(ACTPolicy, self).__init__()
        # Image Encoder (Pretrained ResNet18)
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the classification layer
        self.image_encoder = nn.Sequential(*modules)  # Output: [batch, 512, 1, 1]
        self.image_feature_dim = 512

        # Robot Status Encoder
        self.robot_encoder = nn.Sequential(
            nn.Linear(num_joints + 3, hidden_dim),  # Assuming 3 for target position
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.robot_feature_dim = hidden_dim

        # Total feature dimension
        self.total_feature_dim = self.image_feature_dim + self.robot_feature_dim

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(self.total_feature_dim, dropout, max_len=max_sequence_length)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.total_feature_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Layer
        self.output_layer = nn.Linear(self.total_feature_dim, 3)  # Predicting [x, y, z] target position

        # Normalize Images
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, images, qposes):
        """
        images: [sequence_length, batch_size, 3, H, W]
        qposes: [sequence_length, batch_size, num_joints + 3]
        """
        sequence_length, batch_size = images.shape[0], images.shape[1]

        # Encode Images
        img_feats = []
        for t in range(sequence_length):
            img = images[t]  # [batch_size, 3, H, W]
            img = torch.stack([self.normalize(img[i]) for i in range(batch_size)])
            img_feat = self.image_encoder(img)  # [batch_size, 512, 1, 1]
            img_feat = img_feat.view(batch_size, -1)  # [batch_size, 512]
            img_feats.append(img_feat)
        img_feats = torch.stack(img_feats)  # [sequence_length, batch_size, 512]

        # Encode Robot Statuses
        robot_feats = []
        for t in range(sequence_length):
            robot_feat = self.robot_encoder(qposes[t])  # [batch_size, hidden_dim]
            robot_feats.append(robot_feat)
        robot_feats = torch.stack(robot_feats)  # [sequence_length, batch_size, hidden_dim]

        # Concatenate Features
        combined_feats = torch.cat((robot_feats, img_feats), dim=2)  # [sequence_length, batch_size, total_feature_dim]

        # Apply Positional Encoding
        combined_feats = self.positional_encoding(combined_feats)  # [sequence_length, batch_size, total_feature_dim]

        # Pass through Transformer Encoder
        transformer_out = self.transformer_encoder(combined_feats)  # [sequence_length, batch_size, total_feature_dim]

        # Predict Next Target Positions for each timestep
        next_targets = self.output_layer(transformer_out)  # [sequence_length, batch_size, 3]

        return next_targets

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings once in log space
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [sequence_length, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)