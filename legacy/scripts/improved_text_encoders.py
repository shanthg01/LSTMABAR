"""
Improved Text Encoder Implementations
Provides multiple enhanced approaches for better audio description encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import numpy as np


# ============================================================================
# APPROACH 1: Better Base Model with Improved Projection
# ============================================================================

class ImprovedTextEncoder(nn.Module):
    """
    Enhanced text encoder with better base model and improved architecture
    
    Improvements:
    - Deeper projection network with bottleneck
    - GELU activation (better for transformers)
    - Configurable fine-tuning of backbone layers
    """
    
    def __init__(
        self,
        model_name='sentence-transformers/all-mpnet-base-v2',  # Better than MiniLM
        embedding_dim=768,
        projection_depth='deep',  # 'shallow', 'medium', 'deep'
        device='cpu'
    ):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        
        # Load better sentence transformer
        print(f"Loading improved text encoder: {model_name}")
        self.sentence_model = SentenceTransformer(model_name)
        self.base_dim = self.sentence_model.get_sentence_embedding_dimension()
        
        # Build projection based on depth
        self.projection = self._build_projection(projection_depth)
        self.to(device)
    
    def _build_projection(self, depth: str) -> nn.Module:
        """Build projection network with specified depth"""
        
        if depth == 'shallow':
            # Original simple projection
            return nn.Sequential(
                nn.Linear(self.base_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim)
            )
        
        elif depth == 'medium':
            # Improved 3-layer projection
            return nn.Sequential(
                nn.Linear(self.base_dim, self.embedding_dim * 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim)
            )
        
        elif depth == 'deep':
            # Deep projection with residual connection
            return DeepProjectionHead(self.base_dim, self.embedding_dim)
        
        else:
            raise ValueError(f"Unknown projection depth: {depth}")
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts into embeddings"""
        # Frozen sentence embeddings
        with torch.no_grad():
            base_embeddings = self.sentence_model.encode(
                texts,
                convert_to_tensor=True,
                device=self.device
            )
        
        # Make trainable
        base_embeddings = base_embeddings.clone().detach()
        
        # Trainable projection
        embeddings = self.projection(base_embeddings)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class DeepProjectionHead(nn.Module):
    """Deep projection head with residual connections"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        hidden_dim = output_dim * 2
        
        self.layers = nn.ModuleList([
            # Layer 1: Expand
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # Layer 2: Transform
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            
            # Layer 3: Contract
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # Layer 4: Refine
            nn.Linear(output_dim, output_dim),
        ])
        
        self.norm = nn.LayerNorm(output_dim)
        
        # Learnable residual gate
        self.gate = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        identity = x if x.shape[-1] == self.layers[-1].out_features else None
        
        # Forward through layers
        for layer in self.layers:
            x = layer(x)
        
        # Residual connection if dimensions match
        if identity is not None:
            x = self.gate * x + (1 - self.gate) * identity
        
        x = self.norm(x)
        return x


# ============================================================================
# APPROACH 2: HuggingFace Backbone Fine-Tuning
# ============================================================================

class HFBackboneTextEncoder(nn.Module):
    """
    Text encoder with direct HuggingFace model access for fine-tuning
    Allows unfreezing transformer layers
    """
    
    def __init__(
        self,
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        embedding_dim=768,
        num_unfrozen_layers=4,  # Improved: 4 instead of 2
        device='cpu'
    ):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_unfrozen_layers = num_unfrozen_layers
        
        # Load tokenizer and model
        print(f"Loading HF model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.base_dim = self.backbone.config.hidden_size
        
        # Improved projection
        self.projection = nn.Sequential(
            nn.Linear(self.base_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        # Configure fine-tuning
        self.configure_fine_tuning(num_unfrozen_layers)
        self.to(device)
    
    def configure_fine_tuning(self, num_unfrozen_layers: int):
        """
        Configure which layers to fine-tune
        
        Args:
            num_unfrozen_layers: Number of top transformer layers to unfreeze
                                For MiniLM-L6: max is 6
                                For MPNet: max is 12
        """
        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Determine total number of layers
        if hasattr(self.backbone.config, 'num_hidden_layers'):
            total_layers = self.backbone.config.num_hidden_layers
        else:
            total_layers = 6  # Default for MiniLM
        
        # Unfreeze last N layers
        for i in range(total_layers - num_unfrozen_layers, total_layers):
            layer_name = f"encoder.layer.{i}"
            for name, p in self.backbone.named_parameters():
                if name.startswith(layer_name):
                    p.requires_grad = True
        
        # Always keep projection trainable
        for p in self.projection.parameters():
            p.requires_grad = True
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Unfroze {num_unfrozen_layers} layers: {trainable:,} backbone params trainable")
    
    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling over sequence with attention mask"""
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts into embeddings"""
        # Tokenize
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        # Forward through backbone
        outputs = self.backbone(**tokens)
        
        # Mean pool
        pooled = self._mean_pool(outputs.last_hidden_state, tokens.attention_mask)
        
        # Project
        embeddings = self.projection(pooled)
        
        # Normalize
        return F.normalize(embeddings, p=2, dim=1)


# ============================================================================
# APPROACH 3: Domain-Specific Descriptor Embeddings
# ============================================================================

class AudioDescriptorAwareEncoder(nn.Module):
    """
    Text encoder with specialized audio descriptor awareness
    Combines general semantic embeddings with learned audio-specific features
    """
    
    def __init__(
        self,
        base_encoder: nn.Module,
        embedding_dim: int = 768,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.base_encoder = base_encoder
        
        # Audio descriptor vocabulary
        self.audio_descriptors = {
            'brightness': ['bright', 'dark', 'brilliant', 'dull', 'sharp', 'crisp', 'clear', 'dim'],
            'warmth': ['warm', 'cold', 'cool', 'hot', 'cozy', 'icy', 'mellow', 'rich'],
            'texture': ['smooth', 'rough', 'grainy', 'silky', 'crunchy', 'gritty', 'creamy'],
            'hardness': ['soft', 'hard', 'gentle', 'aggressive', 'harsh', 'mellow', 'abrasive'],
            'density': ['thick', 'thin', 'fat', 'lean', 'full', 'hollow', 'dense', 'sparse'],
            'attack': ['punchy', 'soft', 'sharp', 'gradual', 'instant', 'slow', 'fast'],
            'timbre': ['metallic', 'wooden', 'glassy', 'plastic', 'organic', 'synthetic'],
        }
        
        # Build descriptor vocabulary
        self.descriptor_vocab = self._build_vocab()
        self.vocab_size = len(self.descriptor_vocab)
        
        # Learnable descriptor embeddings
        self.descriptor_embed = nn.Embedding(self.vocab_size, embedding_dim)
        
        # Attention mechanism for blending
        self.blend_attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build descriptor to index mapping"""
        vocab = {}
        idx = 0
        for category, descriptors in self.audio_descriptors.items():
            for desc in descriptors:
                if desc not in vocab:
                    vocab[desc] = idx
                    idx += 1
        return vocab
    
    def _get_descriptor_embedding(self, text: str) -> torch.Tensor:
        """Extract and embed audio descriptors from text"""
        words = text.lower().split()
        
        # Find matching descriptors
        descriptor_indices = []
        for word in words:
            if word in self.descriptor_vocab:
                descriptor_indices.append(self.descriptor_vocab[word])
        
        if not descriptor_indices:
            # No descriptors found - return zero embedding
            return torch.zeros(self.embedding_dim, device=self.device)
        
        # Get embeddings and average
        indices = torch.tensor(descriptor_indices, device=self.device)
        embeddings = self.descriptor_embed(indices)
        return embeddings.mean(dim=0)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts with descriptor awareness"""
        # Get base embeddings
        base_emb = self.base_encoder(texts)
        
        # Extract descriptor embeddings
        descriptor_embs = []
        for text in texts:
            desc_emb = self._get_descriptor_embedding(text)
            descriptor_embs.append(desc_emb)
        descriptor_embs = torch.stack(descriptor_embs)
        
        # Learned blending
        combined = torch.cat([base_emb, descriptor_embs], dim=-1)
        alpha = self.blend_attention(combined)  # [B, 1]
        
        # Weighted combination
        output = alpha * descriptor_embs + (1 - alpha) * base_emb
        
        # Normalize
        return F.normalize(output, p=2, dim=1)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def get_optimizer_grouped_parameters(
    model: nn.Module,
    backbone_lr: float = 5e-5,
    projection_lr: float = 5e-4,
    weight_decay: float = 0.01
) -> List[Dict]:
    """
    Create parameter groups with discriminative learning rates
    
    This is crucial for fine-tuning: backbone needs smaller LR than projection
    """
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    
    optimizer_grouped_parameters = []
    
    # Check if we have backbone to fine-tune
    has_backbone = (hasattr(model, 'text_encoder') and 
                   hasattr(model.text_encoder, 'backbone'))
    
    if has_backbone:
        # Backbone with decay
        optimizer_grouped_parameters.append({
            'params': [p for n, p in model.text_encoder.backbone.named_parameters() 
                      if p.requires_grad and not any(nd in n for nd in no_decay)],
            'lr': backbone_lr,
            'weight_decay': weight_decay
        })
        
        # Backbone without decay
        optimizer_grouped_parameters.append({
            'params': [p for n, p in model.text_encoder.backbone.named_parameters() 
                      if p.requires_grad and any(nd in n for nd in no_decay)],
            'lr': backbone_lr,
            'weight_decay': 0.0
        })
    
    # Projection head with decay
    if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'projection'):
        optimizer_grouped_parameters.append({
            'params': [p for n, p in model.text_encoder.projection.named_parameters() 
                      if p.requires_grad and not any(nd in n for nd in no_decay)],
            'lr': projection_lr,
            'weight_decay': weight_decay
        })
        
        optimizer_grouped_parameters.append({
            'params': [p for n, p in model.text_encoder.projection.named_parameters() 
                      if p.requires_grad and any(nd in n for nd in no_decay)],
            'lr': projection_lr,
            'weight_decay': 0.0
        })
    
    # Contrastive module
    if hasattr(model, 'contrastive_module'):
        optimizer_grouped_parameters.append({
            'params': model.contrastive_module.parameters(),
            'lr': projection_lr,
            'weight_decay': weight_decay
        })
    
    return optimizer_grouped_parameters


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function
    between the initial lr set in the optimizer to 0, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.
    """
    from torch.optim.lr_scheduler import LambdaLR
    import math
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("TESTING IMPROVED TEXT ENCODERS")
    print("="*80)
    
    # Example 1: Improved projection-only encoder
    print("\n1. ImprovedTextEncoder with deep projection:")
    encoder1 = ImprovedTextEncoder(
        model_name='sentence-transformers/all-mpnet-base-v2',
        embedding_dim=768,
        projection_depth='deep',
        device=device
    )
    
    test_texts = [
        "bright and cutting guitar with metallic tone",
        "warm smooth bass with deep resonance",
        "harsh digital synth with buzzy character"
    ]
    
    embeddings1 = encoder1(test_texts)
    print(f"Output shape: {embeddings1.shape}")
    print(f"Embedding norm: {torch.norm(embeddings1[0]).item():.4f}")
    
    # Example 2: HF backbone fine-tuning
    print("\n2. HFBackboneTextEncoder with 4 unfrozen layers:")
    encoder2 = HFBackboneTextEncoder(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        embedding_dim=768,
        num_unfrozen_layers=4,
        device=device
    )
    
    embeddings2 = encoder2(test_texts)
    print(f"Output shape: {embeddings2.shape}")
    
    # Example 3: Descriptor-aware encoder
    print("\n3. AudioDescriptorAwareEncoder:")
    base_encoder = ImprovedTextEncoder(embedding_dim=768, device=device)
    encoder3 = AudioDescriptorAwareEncoder(
        base_encoder=base_encoder,
        embedding_dim=768,
        device=device
    )
    
    embeddings3 = encoder3(test_texts)
    print(f"Output shape: {embeddings3.shape}")
    print(f"Number of audio descriptors: {encoder3.vocab_size}")
    
    # Example 4: Optimizer with grouped parameters
    print("\n4. Creating discriminative optimizer:")
    from lstmabar_model import LSTMABAR
    
    model = LSTMABAR(embedding_dim=768, device=device)
    model.text_encoder = encoder2  # Use HF backbone version
    
    param_groups = get_optimizer_grouped_parameters(
        model,
        backbone_lr=5e-5,
        projection_lr=5e-4,
        weight_decay=0.01
    )
    
    optimizer = torch.optim.AdamW(param_groups)
    print(f"Created optimizer with {len(param_groups)} parameter groups")
    
    print("\n" + "="*80)
    print("All tests passed!")
