"""
Contrastive Alignment Layer - CLAP-style Contrastive Learning
Aligns text and audio embeddings in shared semantic space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class ContrastiveAlignmentModule(nn.Module):
    """
    Contrastive learning module for aligning text and audio embeddings
    
    Implements:
    - InfoNCE contrastive loss (CLAP-style)
    - Auxiliary binary classifier for match prediction
    - Temperature-scaled cosine similarity
    """
    
    def __init__(
        self,
        embedding_dim=768,
        temperature=0.07,
        use_auxiliary_classifier=True,
        device='cpu'
    ):
        super(ContrastiveAlignmentModule, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.use_auxiliary_classifier = use_auxiliary_classifier
        self.device = device
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
        # Auxiliary binary classifier for match prediction
        if use_auxiliary_classifier:
            self.match_classifier = nn.Sequential(
                nn.Linear(embedding_dim * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        
        self.to(device)
    
    def contrastive_loss(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss
        
        Args:
            text_embeddings: (batch_size, embedding_dim)
            audio_embeddings: (batch_size, embedding_dim)
            labels: Optional ground truth alignment (batch_size,)
                   If None, assumes diagonal alignment (i-th text matches i-th audio)
        
        Returns:
            Contrastive loss scalar
        """
        batch_size = text_embeddings.shape[0]
        
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)
        
        # Compute cosine similarity matrix
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * text_embeddings @ audio_embeddings.T
        
        # Create labels for positive pairs
        if labels is None:
            # Assume diagonal alignment
            labels = torch.arange(batch_size, device=self.device)
        
        # Compute cross-entropy loss in both directions
        loss_t2a = F.cross_entropy(logits, labels)  # Text to audio
        loss_a2t = F.cross_entropy(logits.T, labels)  # Audio to text
        
        # Average bidirectional loss
        loss = (loss_t2a + loss_a2t) / 2.0
        
        return loss
    
    def auxiliary_match_loss(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
        match_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary binary classification loss for match prediction
        
        Args:
            text_embeddings: (batch_size, embedding_dim)
            audio_embeddings: (batch_size, embedding_dim)
            match_labels: Binary labels (batch_size,) - 1 for match, 0 for no match
        
        Returns:
            Binary cross-entropy loss
        """
        if not self.use_auxiliary_classifier:
            return torch.tensor(0.0, device=self.device)
        
        # Concatenate embeddings
        combined = torch.cat([text_embeddings, audio_embeddings], dim=1)
        
        # Predict match probability
        match_probs = self.match_classifier(combined).squeeze()
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(match_probs, match_labels.float())
        
        return loss
    
    def compute_similarity_matrix(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise similarity matrix between text and audio
        
        Returns:
            Similarity matrix of shape (batch_text, batch_audio)
        """
        # Normalize
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)
        
        # Cosine similarity
        similarity = text_embeddings @ audio_embeddings.T
        
        return similarity
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
        match_labels: Optional[torch.Tensor] = None,
        compute_auxiliary: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass computing both losses
        
        Args:
            text_embeddings: Text embeddings
            audio_embeddings: Audio embeddings
            match_labels: Optional binary match labels for auxiliary loss
            compute_auxiliary: Whether to compute auxiliary loss
        
        Returns:
            - Contrastive loss
            - Auxiliary loss (if compute_auxiliary=True and match_labels provided)
        """
        # Contrastive loss
        contrastive_loss = self.contrastive_loss(text_embeddings, audio_embeddings)
        
        # Auxiliary loss
        auxiliary_loss = None
        if compute_auxiliary and match_labels is not None and self.use_auxiliary_classifier:
            auxiliary_loss = self.auxiliary_match_loss(
                text_embeddings, 
                audio_embeddings, 
                match_labels
            )
        
        return contrastive_loss, auxiliary_loss


class HardNegativeMiner:
    """
    Mine hard negatives for contrastive learning
    Selects challenging negative pairs to improve learning
    """
    
    def __init__(self, margin=0.2):
        self.margin = margin
    
    def mine_hard_negatives(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
        k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find k hardest negative pairs for each anchor
        
        Args:
            text_embeddings: (batch_size, embedding_dim)
            audio_embeddings: (batch_size, embedding_dim)
            k: Number of hard negatives per anchor
        
        Returns:
            - Hard negative text indices
            - Hard negative audio indices
        """
        batch_size = text_embeddings.shape[0]
        
        # Compute similarity matrix
        sim_matrix = text_embeddings @ audio_embeddings.T
        
        # Mask diagonal (positive pairs)
        mask = torch.eye(batch_size, device=text_embeddings.device).bool()
        sim_matrix_masked = sim_matrix.masked_fill(mask, -1e9)
        
        # Find top-k most similar negatives for each anchor
        hard_neg_indices = torch.topk(sim_matrix_masked, k=k, dim=1).indices
        
        return hard_neg_indices, hard_neg_indices  # Symmetric for bidirectional


class BatchCreator:
    """
    Create training batches with positive and negative pairs
    Ensures balanced sampling for contrastive learning
    """
    
    def __init__(self, negative_ratio=2):
        self.negative_ratio = negative_ratio
    
    def create_batch(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
        positive_pairs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create batch with positive and negative pairs
        
        Args:
            text_embeddings: All text embeddings
            audio_embeddings: All audio embeddings
            positive_pairs: Indices of positive pairs (N, 2)
        
        Returns:
            - Text batch
            - Audio batch
            - Labels (1 for positive, 0 for negative)
        """
        num_positives = positive_pairs.shape[0]
        num_negatives = num_positives * self.negative_ratio
        
        # Positive pairs
        pos_text = text_embeddings[positive_pairs[:, 0]]
        pos_audio = audio_embeddings[positive_pairs[:, 1]]
        pos_labels = torch.ones(num_positives)
        
        # Sample random negative pairs
        total_samples = text_embeddings.shape[0]
        neg_text_idx = torch.randint(0, total_samples, (num_negatives,))
        neg_audio_idx = torch.randint(0, total_samples, (num_negatives,))
        
        # Ensure negatives are actually negative (not in positive pairs)
        neg_mask = neg_text_idx != neg_audio_idx
        neg_text_idx = neg_text_idx[neg_mask][:num_negatives]
        neg_audio_idx = neg_audio_idx[neg_mask][:num_negatives]
        
        neg_text = text_embeddings[neg_text_idx]
        neg_audio = audio_embeddings[neg_audio_idx]
        neg_labels = torch.zeros(len(neg_text_idx))
        
        # Combine
        text_batch = torch.cat([pos_text, neg_text], dim=0)
        audio_batch = torch.cat([pos_audio, neg_audio], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        # Shuffle
        shuffle_idx = torch.randperm(text_batch.shape[0])
        text_batch = text_batch[shuffle_idx]
        audio_batch = audio_batch[shuffle_idx]
        labels = labels[shuffle_idx]
        
        return text_batch, audio_batch, labels


# Example usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize contrastive module
    contrastive_module = ContrastiveAlignmentModule(
        embedding_dim=768,
        temperature=0.07,
        use_auxiliary_classifier=True,
        device=device
    )
    
    # Create dummy embeddings
    batch_size = 16
    text_emb = torch.randn(batch_size, 768).to(device)
    audio_emb = torch.randn(batch_size, 768).to(device)
    
    # Normalize (as they would be from encoders)
    text_emb = F.normalize(text_emb, p=2, dim=1)
    audio_emb = F.normalize(audio_emb, p=2, dim=1)
    
    # Compute losses
    print("=== Testing Contrastive Loss ===")
    contrastive_loss, _ = contrastive_module(text_emb, audio_emb, compute_auxiliary=False)
    print(f"Contrastive loss: {contrastive_loss.item():.4f}")
    
    # Test with auxiliary loss
    print("\n=== Testing Auxiliary Loss ===")
    match_labels = torch.randint(0, 2, (batch_size,)).to(device)
    contrastive_loss, aux_loss = contrastive_module(
        text_emb, 
        audio_emb, 
        match_labels=match_labels,
        compute_auxiliary=True
    )
    print(f"Contrastive loss: {contrastive_loss.item():.4f}")
    print(f"Auxiliary loss: {aux_loss.item():.4f}")
    
    # Test similarity matrix
    print("\n=== Testing Similarity Matrix ===")
    sim_matrix = contrastive_module.compute_similarity_matrix(text_emb, audio_emb)
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(f"Diagonal (positive pairs): {torch.diag(sim_matrix).mean().item():.4f}")
    print(f"Off-diagonal (negative pairs): {(sim_matrix.sum() - torch.diag(sim_matrix).sum()) / (batch_size * (batch_size - 1)):.4f}")
    
    # Test hard negative mining
    print("\n=== Testing Hard Negative Mining ===")
    miner = HardNegativeMiner(margin=0.2)
    hard_negs, _ = miner.mine_hard_negatives(text_emb, audio_emb, k=3)
    print(f"Hard negatives shape: {hard_negs.shape}")
    print(f"Sample hard negatives for anchor 0: {hard_negs[0]}")