import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# Numerical stability constraints for variance and standard deviation predictions
# These values prevent numerical instability (overflow/underflow) during training

# Log-variance bounds (used in Decoder)
# log_var represents log(variance), so variance = exp(log_var)
LOG_VAR_MIN = -7.0
LOG_VAR_MAX = 7.0

# Log-sigma bounds (used in LatentEncoder)
# log_sigma represents log(std), so std = exp(log_sigma)
LOG_SIGMA_MIN = -10.0
LOG_SIGMA_MAX = 2.0


class EmbeddingEncoder(nn.Module):
    """Encode GeoTessera embedding patches into feature vectors."""

    def __init__(
        self,
        patch_size: int = 3,
        in_channels: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128
    ):
        """
        Initialize embedding encoder.

        Args:
            patch_size: Size of embedding patch (e.g., 3x3)
            in_channels: Number of embedding channels (128 for GeoTessera)
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
        """
        super().__init__()

        # Deeper CNN to process spatial structure of embedding patch
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)

        # Residual connection for first block
        self.residual_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # Global pooling and projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode embedding patches.

        Args:
            embeddings: (batch, patch_size, patch_size, channels)

        Returns:
            Feature vectors (batch, output_dim)
        """
        # Reshape to (batch, channels, height, width)
        x = embeddings.permute(0, 3, 1, 2)

        # First block with residual
        identity = self.residual_proj(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)) + identity)

        # Third conv layer
        x = F.relu(self.bn3(self.conv3(x)) + x)

        # Global pooling
        x = self.pool(x).squeeze(-1).squeeze(-1)

        # Projection
        x = self.fc(x)

        return x


class ContextEncoder(nn.Module):
    """Encode context points (coord + embedding + agbd) into representations."""

    def __init__(
        self,
        coord_dim: int = 2,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128
    ):
        """
        Initialize context encoder.

        Args:
            coord_dim: Coordinate dimension (2 for lon/lat)
            embedding_dim: Embedding feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output representation dimension
        """
        super().__init__()

        input_dim = coord_dim + embedding_dim + 1  # coords + embedding + agbd

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        coords: torch.Tensor,
        embedding_features: torch.Tensor,
        agbd: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode context points.

        Args:
            coords: (batch, coord_dim)
            embedding_features: (batch, embedding_dim)
            agbd: (batch, 1)

        Returns:
            Representations (batch, output_dim)
        """
        x = torch.cat([coords, embedding_features, agbd], dim=-1)

        # First layer
        x = F.relu(self.ln1(self.fc1(x)))

        # Second layer with residual
        identity = x
        x = F.relu(self.ln2(self.fc2(x)) + identity)

        # Third layer with residual
        x = F.relu(self.ln3(self.fc3(x)) + x)

        # Output projection
        x = self.fc_out(x)

        return x


class Decoder(nn.Module):
    """
    Decode query point + context representation to AGBD prediction.

    IMPORTANT: This decoder outputs (mean, log_var) where log_var is the
    logarithm of the VARIANCE (not standard deviation).
    Convention: log_var = log(variance), so variance = exp(log_var), std = exp(0.5 * log_var)
    """

    def __init__(
        self,
        coord_dim: int = 2,
        embedding_dim: int = 128,
        context_dim: int = 128,
        hidden_dim: int = 256,
        output_uncertainty: bool = True
    ):
        """
        Initialize decoder.

        Args:
            coord_dim: Coordinate dimension
            embedding_dim: Embedding feature dimension
            context_dim: Context representation dimension
            hidden_dim: Hidden layer dimension
            output_uncertainty: Whether to output uncertainty estimate
        """
        super().__init__()

        self.output_uncertainty = output_uncertainty
        input_dim = coord_dim + embedding_dim + context_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # Output heads
        self.mean_head = nn.Linear(hidden_dim, 1)

        if output_uncertainty:
            self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        query_coords: torch.Tensor,
        query_embedding_features: torch.Tensor,
        context_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode query points.

        Args:
            query_coords: (batch, coord_dim)
            query_embedding_features: (batch, embedding_dim)
            context_repr: (batch, context_dim)

        Returns:
            (mean, log_variance) predictions, where log_variance is None if not output_uncertainty
        """
        x = torch.cat([query_coords, query_embedding_features, context_repr], dim=-1)

        # First layer
        x = F.relu(self.ln1(self.fc1(x)))

        # Second layer with residual
        identity = x
        x = F.relu(self.ln2(self.fc2(x)) + identity)

        # Third layer with residual
        x = F.relu(self.ln3(self.fc3(x)) + x)

        mean = self.mean_head(x)

        if self.output_uncertainty:
            log_var = self.log_var_head(x)
            # Clamp log_var to prevent numerical instability
            log_var = torch.clamp(log_var, min=LOG_VAR_MIN, max=LOG_VAR_MAX)
            return mean, log_var
        else:
            return mean, None


class AttentionAggregator(nn.Module):
    """Attention-based aggregation of context representations."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize attention aggregator.

        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_repr: torch.Tensor,
        context_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate context using attention.

        Args:
            query_repr: Query representations (n_query, dim)
            context_repr: Context representations (n_context, dim)

        Returns:
            Aggregated context for each query (n_query, dim)
        """
        # Expand dimensions for batch processing
        query = query_repr.unsqueeze(0)  # (1, n_query, dim)
        context = context_repr.unsqueeze(0)  # (1, n_context, dim)

        # Apply cross-attention
        attended, _ = self.attention(query, context, context)

        # Apply dropout and residual connection
        attended = self.dropout(attended)
        output = self.norm(attended.squeeze(0) + query_repr)

        return output


class LatentEncoder(nn.Module):
    """
    Encode context representations into latent distribution (stochastic path).

    IMPORTANT: This encoder outputs (mu, log_sigma) where log_sigma is the
    logarithm of the STANDARD DEVIATION (not variance).
    Convention: log_sigma = log(std), so sigma = exp(log_sigma)
    """

    def __init__(
        self,
        context_repr_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128
    ):
        """
        Initialize latent encoder.

        Args:
            context_repr_dim: Dimension of context representations
            hidden_dim: Hidden layer dimension
            latent_dim: Dimension of latent variable
        """
        super().__init__()

        self.fc1 = nn.Linear(context_repr_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Separate heads for mean and log-variance
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, context_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode context into latent distribution.

        Args:
            context_repr: Context representations (n_context, context_repr_dim)

        Returns:
            (mu, log_sigma) of latent distribution, where log_sigma = log(std)
            Shape: (1, latent_dim) each
        """
        # Mean pool context
        pooled = context_repr.mean(dim=0, keepdim=True)

        # Hidden layers
        x = F.relu(self.fc1(pooled))
        x = F.relu(self.fc2(x))

        # Predict distribution parameters
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)

        # Clamp log_sigma for numerical stability
        log_sigma = torch.clamp(log_sigma, min=LOG_SIGMA_MIN, max=LOG_SIGMA_MAX)

        return mu, log_sigma


class GEDINeuralProcess(nn.Module):
    """
    Neural Process for GEDI AGB interpolation with foundation model embeddings.

    Architecture modes:
    - 'deterministic': Only deterministic attention path (original implementation)
    - 'latent': Only latent stochastic path (global context)
    - 'anp': Full Attentive Neural Process (both paths)
    - 'cnp': Conditional Neural Process (mean pooling, no attention/latent)

    Components:
    1. Encode embedding patches to feature vectors
    2. Encode context points (coord + embedding feature + agbd)
    3. Deterministic path: Query-specific attention aggregation (optional)
    4. Latent path: Global stochastic latent variable (optional)
    5. Decode query points to AGBD predictions with uncertainty
    """

    def __init__(
        self,
        patch_size: int = 3,
        embedding_channels: int = 128,
        embedding_feature_dim: int = 128,
        context_repr_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        output_uncertainty: bool = True,
        architecture_mode: str = 'deterministic',
        num_attention_heads: int = 4
    ):
        """
        Initialize Neural Process.

        Args:
            patch_size: Size of embedding patches
            embedding_channels: Number of channels in embeddings
            embedding_feature_dim: Dimension of encoded embedding features
            context_repr_dim: Dimension of context representations
            hidden_dim: Hidden layer dimension
            latent_dim: Dimension of latent variable
            output_uncertainty: Whether to predict uncertainty
            architecture_mode: 'deterministic', 'latent', 'anp', or 'cnp'
            num_attention_heads: Number of attention heads
        """
        super().__init__()

        assert architecture_mode in ['deterministic', 'latent', 'anp', 'cnp'], \
            f"Invalid architecture_mode: {architecture_mode}"

        self.output_uncertainty = output_uncertainty
        self.architecture_mode = architecture_mode
        self.latent_dim = latent_dim

        # Determine which components to use
        self.use_attention = architecture_mode in ['deterministic', 'anp']
        self.use_latent = architecture_mode in ['latent', 'anp']

        # Embedding encoder (shared for context and query)
        self.embedding_encoder = EmbeddingEncoder(
            patch_size=patch_size,
            in_channels=embedding_channels,
            hidden_dim=hidden_dim,
            output_dim=embedding_feature_dim
        )

        # Context encoder
        self.context_encoder = ContextEncoder(
            coord_dim=2,
            embedding_dim=embedding_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=context_repr_dim
        )

        # Attention aggregator (deterministic path)
        if self.use_attention:
            self.attention_aggregator = AttentionAggregator(
                dim=context_repr_dim,
                num_heads=num_attention_heads
            )
            # Query projection for attention (coord + embedding -> context_repr_dim)
            self.query_proj = nn.Linear(2 + embedding_feature_dim, context_repr_dim)

        # Latent encoder (stochastic path)
        if self.use_latent:
            self.latent_encoder = LatentEncoder(
                context_repr_dim=context_repr_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim
            )

        # Decoder
        # Context dim depends on which paths are active
        decoder_context_dim = 0
        if self.use_attention or architecture_mode == 'cnp':
            decoder_context_dim += context_repr_dim
        if self.use_latent:
            decoder_context_dim += latent_dim

        self.decoder = Decoder(
            coord_dim=2,
            embedding_dim=embedding_feature_dim,
            context_dim=decoder_context_dim,
            hidden_dim=hidden_dim,
            output_uncertainty=output_uncertainty
        )

    def forward(
        self,
        context_coords: torch.Tensor,
        context_embeddings: torch.Tensor,
        context_agbd: torch.Tensor,
        query_coords: torch.Tensor,
        query_embeddings: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            context_coords: (n_context, 2)
            context_embeddings: (n_context, patch_size, patch_size, channels)
            context_agbd: (n_context, 1)
            query_coords: (n_query, 2)
            query_embeddings: (n_query, patch_size, patch_size, channels)
            training: Whether in training mode (affects latent sampling)

        Returns:
            (predicted_agbd, log_variance, z_mu, z_log_sigma) for query points
            - predicted_agbd: (n_query, 1)
            - log_variance: (n_query, 1) or None
            - z_mu: (1, latent_dim) or None (only if use_latent)
            - z_log_sigma: (1, latent_dim) or None (only if use_latent)
        """
        # Encode embeddings
        context_emb_features = self.embedding_encoder(context_embeddings)
        query_emb_features = self.embedding_encoder(query_embeddings)

        # Encode context points
        context_repr = self.context_encoder(
            context_coords,
            context_emb_features,
            context_agbd
        )

        # Initialize outputs
        z_mu, z_log_sigma = None, None
        context_components = []

        # Deterministic path (attention or mean pooling)
        if self.use_attention:
            # Query-specific attention aggregation
            query_repr = torch.cat([query_coords, query_emb_features], dim=-1)
            query_repr_projected = self.query_proj(query_repr)
            aggregated_context = self.attention_aggregator(
                query_repr_projected,
                context_repr
            )
            context_components.append(aggregated_context)
        elif self.architecture_mode == 'cnp':
            # Mean pooling for CNP baseline
            aggregated_context = context_repr.mean(dim=0, keepdim=True)
            aggregated_context = aggregated_context.expand(query_coords.shape[0], -1)
            context_components.append(aggregated_context)

        # Latent path (stochastic)
        if self.use_latent:
            # Encode context into latent distribution
            z_mu, z_log_sigma = self.latent_encoder(context_repr)

            # Sample latent variable using reparameterization trick
            if training:
                # Sample during training
                # NOTE: z_log_sigma is log(std), not log(variance)
                # Therefore: sigma = exp(log_sigma), not exp(0.5 * log_sigma)
                epsilon = torch.randn_like(z_mu, device=z_mu.device, dtype=z_mu.dtype)
                z = z_mu + epsilon * torch.exp(z_log_sigma)
            else:
                # Use mean during inference
                z = z_mu

            # Expand latent to match query batch size
            z_expanded = z.expand(query_coords.shape[0], -1)
            context_components.append(z_expanded)

        # Combine paths
        if len(context_components) > 0:
            combined_context = torch.cat(context_components, dim=-1)
        else:
            # This shouldn't happen with valid architecture_mode
            raise ValueError(f"No context components generated for mode: {self.architecture_mode}")

        # Decode query points
        pred_mean, pred_log_var = self.decoder(
            query_coords,
            query_emb_features,
            combined_context
        )

        return pred_mean, pred_log_var, z_mu, z_log_sigma

    def predict(
        self,
        context_coords: torch.Tensor,
        context_embeddings: torch.Tensor,
        context_agbd: torch.Tensor,
        query_coords: torch.Tensor,
        query_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty.

        Returns:
            (mean, std) predictions
        """
        pred_mean, pred_log_var, _, _ = self.forward(
            context_coords,
            context_embeddings,
            context_agbd,
            query_coords,
            query_embeddings,
            training=False
        )

        if pred_log_var is not None:
            pred_std = torch.exp(0.5 * pred_log_var)
        else:
            pred_std = torch.zeros_like(pred_mean)

        return pred_mean, pred_std


def kl_divergence_gaussian(
    mu: torch.Tensor,
    log_sigma: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between N(mu, sigma) and N(0, 1).

    KL(q||p) = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))

    Args:
        mu: Mean of approximate posterior (batch, latent_dim)
        log_sigma: Log standard deviation (batch, latent_dim), where log_sigma = log(std)

    Returns:
        KL divergence (scalar)

    Note:
        exp(2 * log_sigma) = exp(2 * log(std)) = std^2 = variance
        -2 * log_sigma = -log(std^2) = -log(variance)
    """
    # KL divergence formula for Gaussian
    kl = 0.5 * torch.sum(
        torch.exp(2 * log_sigma) + mu ** 2 - 1 - 2 * log_sigma,
        dim=-1
    )
    return kl.mean()


def neural_process_loss(
    pred_mean: torch.Tensor,
    pred_log_var: Optional[torch.Tensor],
    target: torch.Tensor,
    z_mu: Optional[torch.Tensor] = None,
    z_log_sigma: Optional[torch.Tensor] = None,
    kl_weight: float = 1.0
) -> Tuple[torch.Tensor, dict]:
    """
    Loss for Neural Process with optional KL divergence.

    Args:
        pred_mean: Predicted means (batch, 1)
        pred_log_var: Predicted log variances (batch, 1) or None
        target: Target values (batch, 1)
        z_mu: Latent mean (1, latent_dim) or None
        z_log_sigma: Latent log std (1, latent_dim) or None
        kl_weight: Weight for KL divergence term (beta-VAE style)

    Returns:
        (total_loss, loss_dict) where loss_dict contains individual components
    """
    # Reconstruction loss (negative log-likelihood)
    if pred_log_var is not None:
        # Gaussian negative log-likelihood
        nll = 0.5 * (
            pred_log_var +
            torch.exp(-pred_log_var) * (target - pred_mean) ** 2
        )
    else:
        # MSE loss
        nll = (target - pred_mean) ** 2

    nll = nll.mean()

    # KL divergence (if latent path is used)
    kl = torch.tensor(0.0, device=pred_mean.device)
    if z_mu is not None and z_log_sigma is not None:
        kl = kl_divergence_gaussian(z_mu, z_log_sigma)

    # Total loss
    total_loss = nll + kl_weight * kl

    # Return loss components for logging
    loss_dict = {
        'total': total_loss.item(),
        'nll': nll.item(),
        'kl': kl.item()
    }

    return total_loss, loss_dict
