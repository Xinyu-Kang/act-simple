import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import FollowCubeDataset
from models import ACTPolicy
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train ACTPolicy to follow a cube.')
    parser.add_argument('--dataset_path', type=str, default='datasets/follow_cube.hdf5', help='Path to the dataset file.')
    parser.add_argument('--sequence_length', type=int, default=5, help='Sequence length for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--save_path', type=str, default='act_policy.pth', help='Path to save the trained model.')
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
        num_joints=6,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
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
    for epoch in range(args.epochs):
        total_loss = 0.0
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
            predicted_actions = model(imgs, qposes)

            # Compute loss
            loss = criterion(predicted_actions, actions)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), args.save_path)
    print(f'Model saved to {args.save_path}')

if __name__ == '__main__':
    main()