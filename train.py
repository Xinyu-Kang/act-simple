import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import FollowCubeDataset
from models import ACTPolicy
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train ACTPolicy with CVAE to follow a cube.')
    parser.add_argument('--dataset_path', type=str, default='datasets/follow_cube.hdf5', help='Path to the dataset file.')
    parser.add_argument('--sequence_length', type=int, default=5, help='Sequence length for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--save_path', type=str, default='act_policy_cvae.pth', help='Path to save the trained model.')
    parser.add_argument('--latent_dim', type=int, default=64, help='Dimension of latent space in CVAE.')
    parser.add_argument('--kl_weight', type=float, default=0.01, help='Weight for KL divergence in loss.')
    parser.add_argument('--num_joints', type=int, default=6, help='Number of robot joints.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of Transformer encoder/decoder layers.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    args = parser.parse_args()

    # Define image transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create dataset and dataloader
    dataset = FollowCubeDataset(dataset_path=args.dataset_path, sequence_length=args.sequence_length, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize model
    model = ACTPolicy(
        num_joints=args.num_joints,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_sequence_length=args.sequence_length
    )

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    model.train()
    kl_weight = args.kl_weight
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        for batch_idx, (imgs, qposes, actions) in enumerate(dataloader):
            # imgs: [batch_size, sequence_length, C, H, W]
            # qposes: [batch_size, sequence_length, num_joints + 3]
            # actions: [batch_size, sequence_length, 3]

            # Move data to device
            imgs = imgs.to(device)
            qposes = qposes.to(device)
            actions = actions.to(device)

            # Permute to match expected input shape: [sequence_length, batch_size, ...]
            imgs = imgs.permute(1, 0, 2, 3, 4)
            qposes = qposes.permute(1, 0, 2)
            actions = actions.permute(1, 0, 2)

            optimizer.zero_grad()

            # Forward pass
            predicted_actions, kl_loss = model(imgs, qposes, actions)

            # Compute reconstruction loss (MSE)
            recon_loss = criterion(predicted_actions, actions)

            # Total loss
            if kl_loss is not None:
                total_loss_batch = recon_loss + kl_weight * kl_loss
            else:
                total_loss_batch = recon_loss

            # Backward pass and optimization
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            if kl_loss is not None:
                total_kl_loss += kl_loss.item()
            else:
                total_kl_loss += 0.0

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx}/{len(dataloader)}], '
                      f'Total Loss: {total_loss_batch.item():.4f}, Recon Loss: {recon_loss.item():.4f}, '
                      f'KL Loss: {kl_loss.item() if kl_loss is not None else 0.0:.4f}')

        avg_total_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{args.epochs}], Average Total Loss: {avg_total_loss:.4f}, '
              f'Average Recon Loss: {avg_recon_loss:.4f}, Average KL Loss: {avg_kl_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), args.save_path)
    print(f'Model saved to {args.save_path}')

if __name__ == '__main__':
    main()