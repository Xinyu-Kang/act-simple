import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

class ACTPolicy(nn.Module):
    def __init__(self, input_image_size=(224, 224), num_joints=6, hidden_dim=256, latent_dim=64, num_heads=8, num_layers=4, dropout=0.1, max_sequence_length=100):
        super(ACTPolicy, self).__init__()
        # Image Encoder (Pretrained ResNet18)
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the classification layer
        self.image_encoder = nn.Sequential(*modules)  # Output: [batch, 512, 1, 1]
        self.image_feature_dim = 512

        # Robot Status Encoder
        self.robot_encoder = nn.Sequential(
            nn.Linear(num_joints + 3, hidden_dim),
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

        # CVAE Components
        self.latent_dim = latent_dim
        # Prior Network
        self.prior_mean = nn.Linear(self.total_feature_dim, latent_dim)
        self.prior_logvar = nn.Linear(self.total_feature_dim, latent_dim)
        # Posterior Network
        self.posterior_mean = nn.Linear(self.total_feature_dim + 3, latent_dim)
        self.posterior_logvar = nn.Linear(self.total_feature_dim + 3, latent_dim)

        # Decoder (Transformer Decoder)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.total_feature_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output Layer
        self.output_layer = nn.Linear(self.total_feature_dim, 3)  # Predicting [x, y, z] target position

        # Normalize Images
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, images, qposes, actions=None):
        """
        images: [sequence_length, batch_size, 3, H, W]
        qposes: [sequence_length, batch_size, num_joints + 3]
        actions: [sequence_length, batch_size, 3] (only during training)
        """
        sequence_length, batch_size = images.shape[0], images.shape[1]

        # Encode Images and Robot States
        combined_feats = self.encode_inputs(images, qposes)

        # Apply Positional Encoding
        combined_feats = self.positional_encoding(combined_feats)

        # Encode the sequence
        memory = self.transformer_encoder(combined_feats)  # [sequence_length, batch_size, total_feature_dim]

        if self.training and actions is not None:
            # Compute posterior parameters
            posterior_input = torch.cat([memory, actions], dim=2)
            z_mean_post = self.posterior_mean(posterior_input)  # [sequence_length, batch_size, latent_dim]
            z_logvar_post = self.posterior_logvar(posterior_input)

            # Sample z from posterior
            z_post = self.reparameterize(z_mean_post, z_logvar_post)

            # Compute prior parameters
            z_mean_prior = self.prior_mean(memory)
            z_logvar_prior = self.prior_logvar(memory)

            # Compute KL divergence
            kl_loss = self.kl_divergence(z_mean_post, z_logvar_post, z_mean_prior, z_logvar_prior)

            # Decode
            z = z_post
        else:
            # Inference mode: sample from prior
            z_mean_prior = self.prior_mean(memory)
            z_logvar_prior = self.prior_logvar(memory)
            z_prior = self.reparameterize(z_mean_prior, z_logvar_prior)
            z = z_prior
            kl_loss = None

        # Prepare target sequence for the decoder
        if actions is not None:
            tgt = torch.cat([memory, actions], dim=2)
        else:
            # Use zeros as placeholder during inference
            tgt = torch.zeros_like(memory)

        # Decode
        decoder_output = self.transformer_decoder(tgt, memory)  # [sequence_length, batch_size, total_feature_dim]

        # Output Layer
        outputs = self.output_layer(decoder_output)  # [sequence_length, batch_size, 3]

        return outputs, kl_loss

    def encode_inputs(self, images, qposes):
        """
        Encodes images and robot states into combined features.
        """
        sequence_length, batch_size = images.shape[0], images.shape[1]
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
        return combined_feats

    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick to sample from N(mean, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def kl_divergence(self, mean_post, logvar_post, mean_prior, logvar_prior):
        """
        Computes the KL divergence between the posterior and prior distributions.
        """
        kl = 0.5 * (logvar_prior - logvar_post + (torch.exp(logvar_post) + (mean_post - mean_prior) ** 2) / torch.exp(logvar_prior) - 1)
        return kl.sum()

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