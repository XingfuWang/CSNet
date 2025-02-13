# Author: Xingfu Wang at University of Chinese Academy of Sciences
# Contact Email: wangxingfu21[AT]mails[DOT]ucas[DOT]ac[DOT]cn

import torch
import torch.nn as nn


class CholeskyOperations:
    """
    Cholesky Operations: Encapsulates operations required for projecting Gram matrices
    to the Cholesky manifold, a differentiable embedding from the symmetric positive
    definite (SPD) manifold to a lower-dimensional tangent space for efficient computation.
    """

    @staticmethod
    def compute_gram_matrix(Xt, input_window_samples):
        """
        Compute the Gram matrix (a special case of inner product matrix).
        Used to capture linear dependencies in the input space.

        Args:
            Xt (torch.Tensor): Temporal features of shape (batch_size, channels, time).
            input_window_samples (int): The number of time samples in each input window.

        Returns:
            Xb (torch.Tensor): Batch of Gram matrices of shape (batch_size, channels, channels).
        """
        Xb = torch.bmm(Xt, Xt.transpose(1, 2)) / (input_window_samples - 1)
        return Xb

    @staticmethod
    def normalize_gram_matrix(Xb, epsilon=1e-8):
        """
        Normalize the Gram matrix into a correlation matrix to ensure numerical stability.

        Args:
            Xb (torch.Tensor): Batch of Gram matrices.
            epsilon (float): Small value added to avoid division by zero.

        Returns:
            Xb (torch.Tensor): Normalized Gram matrix (correlation matrix).
        """
        diag = torch.sqrt(torch.diagonal(Xb, dim1=1, dim2=2))
        diag_matrix = diag.unsqueeze(2) * diag.unsqueeze(1)
        diag_matrix = diag_matrix + epsilon  # Stabilize diagonal elements
        Xb = Xb / diag_matrix
        return Xb

    @staticmethod
    def cholesky_decomposition(Xb, epsilon=1e-3):
        """
        Perform Cholesky decomposition on the normalized Gram matrix.
        This maps SPD matrices to triangular matrices, ensuring a
        smooth diffeomorphic mapping to the Cholesky manifold.

        Args:
            Xb (torch.Tensor): Normalized Gram matrix.
            epsilon (float): Regularization term for numerical stability.

        Returns:
            L (torch.Tensor): Lower triangular matrix from Cholesky decomposition.
        """
        try:
            L = torch.linalg.cholesky(Xb)
        except torch._C._LinAlgError:
            # Enforce symmetry in case of numerical instability
            Xb = 0.5 * (Xb + Xb.transpose(1, 2))
            Xb += epsilon * torch.eye(Xb.size(-1), device=Xb.device).expand_as(Xb)
            L = torch.linalg.cholesky(Xb)
        return L

    @staticmethod
    def extract_tangent_space(L):
        """
        Extract a tangent space representation from the Cholesky factor L.
        This involves vectorizing both the diagonal and off-diagonal elements
        to create a coordinate system suitable for optimization on the tangent space.

        Returns:
            tangent_space_representation (torch.Tensor): Flattened tensor with log-diagonal
            elements and off-diagonal elements concatenated.
        """
        batch_size, n_features = L.shape[0], L.shape[-1]
        d = L.new_zeros(batch_size, n_features)  # Diagonal elements
        l = L.new_zeros(batch_size, n_features * (n_features - 1) // 2)  # Off-diagonal elements

        for i in range(batch_size):
            d[i] = L[i].diag()
            l[i] = torch.cat([L[i][j: j + 1, :j] for j in range(1, n_features)], dim=1)[0]

        tangent_space_representation = torch.cat((d.log(), l), dim=1)
        return tangent_space_representation

class CSNet(nn.Module):
    """
    CSNet: Cholesky Space-based model with multi-branch spatial and temporal convolutions for brain-computer interfaces.
    This architecture performs best on motor imagery and emotion recognition tasks, CSNet_ST is recommended for ERN task.
    Please refer to the paper "Cholesky Space for Brain-Computer Interfaces" for more details.

    Args:
        n_chans (int): Channel number of the input EEG data.
        n_class (int): Class number of the classification task.
        spatial_expansion (int): Controls the spatial expansion of the spatial feature map.
        spatial_merge (int): Controls the spatial shrinkage of the spatial feature map.
        filters (list): Temporal convolution kernel sizes.
        temporal_expansion (int): Controls the temporal depth of the temporal convolution.
    """

    def __init__(self, n_chans: int, n_class: int = 4, spatial_expansion: int = 240, spatial_merge: int = 32, filters: list = None,
                 temporal_expansion: int = 2):
        super().__init__()
        # Set default temporal convolution kernel sizes if not specified
        filters = filters or [41, 51, 61]

        # Assign base size for spatial feature extraction based on input channels
        base_size = 2 if n_chans <= 64 else (3 if n_chans <= 128 else 4)

        # Feature dimensions based on spatial expansion
        feature_dim = [(spatial_expansion // (n_chans - base_size ** i + 1), base_size ** i) for i in range(2, 6) if base_size ** i < n_chans]
        feature_dim.append((spatial_expansion, n_chans))
        # Spatial convolution layers
        self.spatial_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, dim[0], kernel_size=(dim[1], 1))) for dim in feature_dim]
        )
        self.spatial_convs.append(nn.Sequential(
            nn.Conv2d(1, spatial_merge, kernel_size=(sum(dim[0] * (n_chans - dim[1] + 1) for dim in feature_dim), 1)),
            nn.BatchNorm2d(spatial_merge)
        ))

        # Temporal convolution layers
        self.temporal_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(spatial_merge, temporal_expansion * spatial_merge, kernel_size=(1, size), padding=(0, size // 2), groups=spatial_merge),
            nn.BatchNorm2d(temporal_expansion * spatial_merge)) for size in filters
        ])

        # Fully connected layer for classification
        self.fc = nn.Linear((len(filters) * temporal_expansion * spatial_merge + 1) * (len(filters) * temporal_expansion * spatial_merge) // 2,
                            n_class)

    def forward(self, input):
        batch_size, _, input_window_samples = input.shape
        input = input.unsqueeze(1)

        # Spatial feature extraction
        spatial_features = [conv(input).reshape(batch_size, -1, input_window_samples) for conv in self.spatial_convs[:-1]]
        Xs = torch.cat(spatial_features, 1).unsqueeze(1)
        Xs = self.spatial_convs[-1](Xs)

        # Temporal feature extraction
        Xt = torch.stack([conv(Xs) for conv in self.temporal_convs], dim=1)
        Xt = Xt.reshape(batch_size, -1, Xt.shape[-1])  # [batch_size, features, time]

        # Riemannian manifold embedding
        Xb = torch.bmm(Xt, Xt.transpose(1, 2)) / (input_window_samples - 1)

        # Cholesky decomposition
        L = CholeskyOperations.cholesky_decomposition(Xb)

        # Tangent space mapping
        Xb = CholeskyOperations.extract_tangent_space(L)

        # Classification
        Xm = self.fc(Xb)
        return Xm


class CSNet_ST(nn.Module):
    """
    Use simple spatial and temporal convolutions from CSNet, please refer to the original paper for more details.

    Args:
        n_chans (int): Channel number of the input EEG data.
        n_class (int): Class number of the classification task.
        spatial_expansion (int): Controls the spatial expansion of the spatial feature map.
        spatial_merge (int): Controls the spatial shrinkage of the spatial feature map.
        filters (list): Temporal convolution kernel sizes.
        temporal_expansion (int): Controls the temporal depth of the temporal convolution.
    """

    def __init__(self, n_chans: int, n_class: int = 4, spatial_expansion: int = 240, spatial_merge: int = 32, filters: list = None,
                 temporal_expansion: int = 2):
        super().__init__()
        # Set default temporal convolution kernel sizes if not specified
        filters = filters or [61]

        # Assign base size for spatial feature extraction based on input channels
        base_size = 2 if n_chans <= 64 else (3 if n_chans <= 128 else 4)

        # Feature dimensions based on spatial expansion
        feature_dim = [(spatial_expansion // (n_chans - base_size ** i + 1), base_size ** i) for i in range(2, 2) if base_size ** i < n_chans]
        feature_dim.append((spatial_expansion, n_chans))
        # Spatial convolution layers
        self.spatial_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, dim[0], kernel_size=(dim[1], 1))) for dim in feature_dim]
        )
        self.spatial_convs.append(nn.Sequential(
            nn.Conv2d(1, spatial_merge, kernel_size=(sum(dim[0] * (n_chans - dim[1] + 1) for dim in feature_dim), 1)),
            nn.BatchNorm2d(spatial_merge)
        ))

        # Temporal convolution layers
        self.temporal_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(spatial_merge, temporal_expansion * spatial_merge, kernel_size=(1, size), padding=(0, size // 2), groups=spatial_merge),
            nn.BatchNorm2d(temporal_expansion * spatial_merge)) for size in filters
        ])

        # Fully connected layer for classification
        self.fc = nn.Linear((len(filters) * temporal_expansion * spatial_merge + 1) * (len(filters) * temporal_expansion * spatial_merge) // 2,
                            n_class)

    def forward(self, input):
        batch_size, _, input_window_samples = input.shape
        input = input.unsqueeze(1)

        # Spatial feature extraction
        spatial_features = [conv(input).reshape(batch_size, -1, input_window_samples) for conv in self.spatial_convs[:-1]]
        Xs = torch.cat(spatial_features, 1).unsqueeze(1)
        Xs = self.spatial_convs[-1](Xs)

        # Temporal feature extraction
        Xt = torch.stack([conv(Xs) for conv in self.temporal_convs], dim=1)
        Xt = Xt.reshape(batch_size, -1, Xt.shape[-1])  # [batch_size, features, time]

        # Riemannian manifold embedding
        Xb = torch.bmm(Xt, Xt.transpose(1, 2)) / (input_window_samples - 1)

        # Cholesky decomposition
        L = CholeskyOperations.cholesky_decomposition(Xb)

        # Tangent space mapping
        Xb = CholeskyOperations.extract_tangent_space(L)

        # Classification
        Xm = self.fc(Xb)
        return Xm


class CSNet_MST(nn.Module):
    """
    Replace multi-branch temporal convolution with simple temporal convolution in CSNet.

    Args:
        n_chans (int): Channel number of the input EEG data.
        n_class (int): Class number of the classification task.
        spatial_expansion (int): Controls the spatial expansion of the spatial feature map.
        spatial_merge (int): Controls the spatial shrinkage of the spatial feature map.
        filters (list): Temporal convolution kernel sizes.
        temporal_expansion (int): Controls the temporal depth of the temporal convolution.
    """

    def __init__(self, n_chans: int, n_class: int = 4, spatial_expansion: int = 240, spatial_merge: int = 32,
                 filters: list = None, temporal_expansion: int = 2):
        super().__init__()
        # Set default temporal convolution kernel sizes if not specified
        filters = filters or [61]

        # Assign base size for spatial feature extraction based on input channels
        base_size = 2 if n_chans <= 64 else (3 if n_chans <= 128 else 4)

        # Feature dimensions based on spatial expansion
        feature_dim = [(spatial_expansion // (n_chans - base_size ** i + 1), base_size ** i) for i in range(2, 6) if base_size ** i < n_chans]
        feature_dim.append((spatial_expansion, n_chans))
        # Spatial convolution layers
        self.spatial_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, dim[0], kernel_size=(dim[1], 1))) for dim in feature_dim]
        )
        self.spatial_convs.append(nn.Sequential(
            nn.Conv2d(1, spatial_merge, kernel_size=(sum(dim[0] * (n_chans - dim[1] + 1) for dim in feature_dim), 1)),
            nn.BatchNorm2d(spatial_merge)
        ))

        # Temporal convolution layers
        self.temporal_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(spatial_merge, temporal_expansion * spatial_merge, kernel_size=(1, size), padding=(0, size // 2), groups=spatial_merge),
            nn.BatchNorm2d(temporal_expansion * spatial_merge)) for size in filters
        ])

        # Fully connected layer for classification
        self.fc = nn.Linear((len(filters) * temporal_expansion * spatial_merge + 1) * (len(filters) * temporal_expansion * spatial_merge) // 2,
                            n_class)

    def forward(self, input):
        batch_size, _, input_window_samples = input.shape
        input = input.unsqueeze(1)

        # Spatial feature extraction
        spatial_features = [conv(input).reshape(batch_size, -1, input_window_samples) for conv in self.spatial_convs[:-1]]
        Xs = torch.cat(spatial_features, 1).unsqueeze(1)
        Xs = self.spatial_convs[-1](Xs)

        # Temporal feature extraction
        Xt = torch.stack([conv(Xs) for conv in self.temporal_convs], dim=1)
        Xt = Xt.reshape(batch_size, -1, Xt.shape[-1])  # [batch_size, features, time]

        # Riemannian manifold embedding
        Xb = torch.bmm(Xt, Xt.transpose(1, 2)) / (input_window_samples - 1)

        # Cholesky decomposition
        L = CholeskyOperations.cholesky_decomposition(Xb)

        # Tangent space mapping
        Xb = CholeskyOperations.extract_tangent_space(L)

        # Classification
        Xm = self.fc(Xb)
        return Xm


class CSNet_SMT(nn.Module):
    """
    Replace multi-branch spatial convolution with simple spatial convolution in CSNet.

    Args:
        n_chans (int): Channel number of the input EEG data.
        n_class (int): Class number of the classification task.
        spatial_expansion (int): Controls the spatial expansion of the spatial feature map.
        spatial_merge (int): Controls the spatial shrinkage of the spatial feature map.
        filters (list): Temporal convolution kernel sizes.
        temporal_expansion (int): Controls the temporal depth of the temporal convolution.
    """

    def __init__(self, n_chans: int, n_class: int = 4, spatial_expansion: int = 240, spatial_merge: int = 32,
                 filters: list = None, temporal_expansion: int = 2):
        super().__init__()
        # Set default temporal convolution kernel sizes if not specified
        filters = filters or [41, 51, 61]

        # Assign base size for spatial feature extraction based on input channels
        base_size = 2 if n_chans <= 64 else (3 if n_chans <= 128 else 4)

        # Feature dimensions based on spatial expansion
        feature_dim = [(spatial_expansion // (n_chans - base_size ** i + 1), base_size ** i) for i in range(2, 2) if base_size ** i < n_chans]
        feature_dim.append((spatial_expansion, n_chans))
        # Spatial convolution layers
        self.spatial_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, dim[0], kernel_size=(dim[1], 1))) for dim in feature_dim]
        )
        self.spatial_convs.append(nn.Sequential(
            nn.Conv2d(1, spatial_merge, kernel_size=(sum(dim[0] * (n_chans - dim[1] + 1) for dim in feature_dim), 1)),
            nn.BatchNorm2d(spatial_merge)
        ))

        # Temporal convolution layers
        self.temporal_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(spatial_merge, temporal_expansion * spatial_merge, kernel_size=(1, size), padding=(0, size // 2), groups=spatial_merge),
            nn.BatchNorm2d(temporal_expansion * spatial_merge)) for size in filters
        ])

        # Fully connected layer for classification
        self.fc = nn.Linear((len(filters) * temporal_expansion * spatial_merge + 1) * (len(filters) * temporal_expansion * spatial_merge) // 2,
                            n_class)

    def forward(self, input):
        batch_size, _, input_window_samples = input.shape
        input = input.unsqueeze(1)

        # Spatial feature extraction
        spatial_features = [conv(input).reshape(batch_size, -1, input_window_samples) for conv in self.spatial_convs[:-1]]
        Xs = torch.cat(spatial_features, 1).unsqueeze(1)
        Xs = self.spatial_convs[-1](Xs)

        # Temporal feature extraction
        Xt = torch.stack([conv(Xs) for conv in self.temporal_convs], dim=1)
        Xt = Xt.reshape(batch_size, -1, Xt.shape[-1])  # [batch_size, features, time]

        # Riemannian manifold embedding
        Xb = torch.bmm(Xt, Xt.transpose(1, 2)) / (input_window_samples - 1)

        # Cholesky decomposition
        L = CholeskyOperations.cholesky_decomposition(Xb)

        # Tangent space mapping
        Xb = CholeskyOperations.extract_tangent_space(L)

        # Classification
        Xm = self.fc(Xb)
        return Xm


class CSNet_woCh(nn.Module):
    """
    Remove Cholesky Space from CSNet.

    Args:
        n_chans (int): Channel number of the input EEG data.
        n_class (int): Class number of the classification task.
        spatial_expansion (int): Controls the spatial expansion of the spatial feature map.
        spatial_merge (int): Controls the spatial shrinkage of the spatial feature map.
        filters (list): Temporal convolution kernel sizes.
        temporal_expansion (int): Controls the temporal depth of the temporal convolution.
    """

    def __init__(self, n_chans: int, n_class: int = 4, spatial_expansion: int = 240, spatial_merge: int = 32,
                 filters: list = None, temporal_expansion: int = 2):
        super().__init__()
        # Set default temporal convolution kernel sizes if not specified
        filters = filters or [41, 51, 61]

        # Assign base size for spatial feature extraction based on input channels
        base_size = 2 if n_chans <= 64 else (3 if n_chans <= 128 else 4)

        # Feature dimensions based on spatial expansion
        feature_dim = [(spatial_expansion // (n_chans - base_size ** i + 1), base_size ** i) for i in range(2, 6) if base_size ** i < n_chans]
        feature_dim.append((spatial_expansion, n_chans))
        # Spatial convolution layers
        self.spatial_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, dim[0], kernel_size=(dim[1], 1))) for dim in feature_dim]
        )
        self.spatial_convs.append(nn.Sequential(
            nn.Conv2d(1, spatial_merge, kernel_size=(sum(dim[0] * (n_chans - dim[1] + 1) for dim in feature_dim), 1)),
            nn.BatchNorm2d(spatial_merge)
        ))

        # Temporal convolution layers
        self.temporal_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(spatial_merge, temporal_expansion * spatial_merge, kernel_size=(1, size), padding=(0, size // 2), groups=spatial_merge),
            nn.BatchNorm2d(temporal_expansion * spatial_merge)) for size in filters
        ])

        # Fully connected layer for classification
        self.fc = nn.Linear((len(filters) * temporal_expansion * spatial_merge) * (len(filters) * temporal_expansion * spatial_merge),
                            n_class)

    def forward(self, input):
        batch_size, _, input_window_samples = input.shape
        input = input.unsqueeze(1)

        # Spatial feature extraction
        spatial_features = [conv(input).reshape(batch_size, -1, input_window_samples) for conv in self.spatial_convs[:-1]]
        Xs = torch.cat(spatial_features, 1).unsqueeze(1)
        Xs = self.spatial_convs[-1](Xs)

        # Temporal feature extraction
        Xt = torch.stack([conv(Xs) for conv in self.temporal_convs], dim=1)
        Xt = Xt.reshape(batch_size, -1, Xt.shape[-1])  # [batch_size, features, time]

        # Covariance computation
        Xb = torch.bmm(Xt, Xt.transpose(1, 2)) / (input_window_samples - 1)

        # Classification
        Xb = Xb.flatten(start_dim=1)
        Xm = self.fc(Xb)
        return Xm


if __name__ == '__main__':
    x = torch.randn(1, 62, 1000)  # batch_size, n_chans, input_window_samples
    model = CSNet_woCh(n_chans=62, n_class=4)  # CSNet_ST, CSNet_MST, CSNet_SMT, CSNet_woCh
    y = model(x)
    print(y.shape)
