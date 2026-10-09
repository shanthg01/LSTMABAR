"""
LSTMABAR - Language-to-Sound Transformation Model using Archetype-Based Audio Representation
Complete two-tower architecture integrating all components
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import numpy as np

# Import all components (assuming they're in the same directory)
from text_tower import TextEncoder
from audio_tower import AudioEncoder
from contrastive_alignment import ContrastiveAlignmentModule
from archetype_predictor import ArchetypePredictionHead, ArchetypeLoss
from ddsp_transformation import DDSPTransformationEngine


class LSTMABAR(nn.Module):
    """
    Complete LSTMABAR model
    
    Architecture:
    1. Text Tower: Encodes descriptions → text embeddings
    2. Audio Tower: Encodes spectrograms → audio embeddings  
    3. Contrastive Alignment: Aligns text and audio in shared space
    4. Archetype Predictor: Predicts mixture weights from joint embeddings
    5. DDSP Engine: Transforms audio based on archetype weights
    """
    
    def __init__(
        self,
        embedding_dim=768,
        text_model='sentence-transformers/all-MiniLM-L6-v2',
        audio_architecture='resnet',  # 'resnet' or 'ast'
        num_archetypes=5,
        sample_rate=44100,
        use_quantum_attention=False,
        n_qubits=8,
        circuit_depth=2,
        dropout_rate=0.3,
        noise_strength=0.1,
        temperature=0.07,
        device='cpu'
    ):
        super(LSTMABAR, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_archetypes = num_archetypes
        self.sample_rate = sample_rate
        self.device = device
        
        # Text Tower
        self.text_encoder = TextEncoder(
            model_name=text_model,
            embedding_dim=embedding_dim,
            use_quantum_attention=use_quantum_attention,
            n_qubits=n_qubits,
            circuit_depth=circuit_depth,
            dropout_rate=dropout_rate,
            noise_strength=noise_strength,
            device=device
        )
        
        # Audio Tower
        self.audio_encoder = AudioEncoder(
            embedding_dim=embedding_dim,
            architecture=audio_architecture,
            sample_rate=sample_rate,
            use_archetype_supervision=True,
            device=device
        )
        
        # Contrastive Alignment
        self.contrastive_module = ContrastiveAlignmentModule(
            embedding_dim=embedding_dim,
            temperature=temperature,
            use_auxiliary_classifier=True,
            device=device
        )
        
        # Archetype Prediction Head
        self.archetype_predictor = ArchetypePredictionHead(
            embedding_dim=embedding_dim,
            num_archetypes=num_archetypes,
            device=device
        )
        
        # DDSP Transformation Engine
        self.ddsp_engine = DDSPTransformationEngine(
            sample_rate=sample_rate,
            learnable_filters=True,
            device=device
        )
        
        # Loss functions
        self.archetype_loss_fn = ArchetypeLoss(loss_type='mse')
        
        self.to(device)
    
    def encode_text(self, descriptions: List[str]) -> torch.Tensor:
        """Encode text descriptions"""
        return self.text_encoder(descriptions)
    
    def encode_audio(
        self,
        audio: torch.Tensor,
        return_archetype_pred: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode audio waveforms"""
        return self.audio_encoder(audio, return_archetype_pred=return_archetype_pred)
    
    def predict_archetypes(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Predict archetype mixture from embeddings"""
        return self.archetype_predictor(text_embeddings, audio_embeddings)
    
    def transform_audio(
        self,
        audio: torch.Tensor,
        archetype_weights: torch.Tensor
    ) -> torch.Tensor:
        """Transform audio using DDSP engine"""
        return self.ddsp_engine(audio, archetype_weights)
    
    def forward(
        self,
        descriptions: List[str],
        audio: torch.Tensor,
        target_archetype_weights: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass
        
        Args:
            descriptions: List of text descriptions
            audio: Input audio waveforms (batch, samples)
            target_archetype_weights: Ground truth archetype weights for training
            return_intermediates: Whether to return intermediate representations
        
        Returns:
            Dict containing:
            - transformed_audio: Output audio
            - losses: Dict of individual losses
            - (optional) intermediate representations
        """
        # Encode text and audio
        text_embeddings = self.encode_text(descriptions)
        audio_embeddings, audio_archetype_pred = self.encode_audio(
            audio, 
            return_archetype_pred=True
        )
        
        # Contrastive alignment
        contrastive_loss, auxiliary_loss = self.contrastive_module(
            text_embeddings,
            audio_embeddings,
            match_labels=None,  # Assume all pairs match in training
            compute_auxiliary=False
        )
        
        # Predict archetype mixture
        predicted_weights = self.predict_archetypes(text_embeddings, audio_embeddings)
        
        # Archetype prediction loss (if targets provided)
        archetype_loss = None
        if target_archetype_weights is not None:
            archetype_loss = self.archetype_loss_fn(
                predicted_weights,
                target_archetype_weights
            )
        
        # Auxiliary archetype supervision loss (from audio encoder)
        audio_archetype_loss = None
        if audio_archetype_pred is not None and target_archetype_weights is not None:
            audio_archetype_loss = self.archetype_loss_fn(
                audio_archetype_pred,
                target_archetype_weights
            )
        
        # Transform audio
        transformed_audio = self.transform_audio(audio, predicted_weights)
        
        # Prepare output
        output = {
            'transformed_audio': transformed_audio,
            'predicted_weights': predicted_weights,
            'losses': {
                'contrastive': contrastive_loss,
                'archetype_prediction': archetype_loss,
                'audio_archetype_supervision': audio_archetype_loss
            }
        }
        
        if return_intermediates:
            output['intermediates'] = {
                'text_embeddings': text_embeddings,
                'audio_embeddings': audio_embeddings,
                'audio_archetype_pred': audio_archetype_pred
            }
        
        return output
    
    def inference(
        self,
        descriptions: List[str],
        audio: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Inference mode: transform audio with description
        
        Returns:
            - Transformed audio
            - Metadata dict with predictions
        """
        self.eval()
        
        with torch.no_grad():
            # Encode
            text_embeddings = self.encode_text(descriptions)
            audio_embeddings, _ = self.encode_audio(audio)
            
            # Predict archetypes
            predicted_weights = self.predict_archetypes(text_embeddings, audio_embeddings)
            
            # Transform
            transformed_audio = self.transform_audio(audio, predicted_weights)
        
        metadata = {
            'predicted_weights': predicted_weights.cpu().numpy(),
            'archetype_names': ['sine', 'square', 'sawtooth', 'triangle', 'noise']
        }
        
        return transformed_audio, metadata
    
    def compute_total_loss(
        self,
        loss_dict: Dict[str, Optional[torch.Tensor]],
        loss_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute weighted total loss
        
        Args:
            loss_dict: Dict of individual losses
            loss_weights: Optional weights for each loss component
        
        Returns:
            Total weighted loss
        """
        if loss_weights is None:
            # Default weights (as suggested in architecture doc)
            loss_weights = {
                'contrastive': 0.7,
                'archetype_prediction': 0.2,
                'audio_archetype_supervision': 0.1
            }
        
        total_loss = 0.0
        for loss_name, loss_value in loss_dict.items():
            if loss_value is not None and loss_name in loss_weights:
                total_loss += loss_weights[loss_name] * loss_value
        
        return total_loss
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': {
                'embedding_dim': self.embedding_dim,
                'num_archetypes': self.num_archetypes,
                'sample_rate': self.sample_rate
            }
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = False):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {path} (epoch {checkpoint['epoch']})")
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            return checkpoint['optimizer_state_dict']
        
        return None


class LSTMABARTrainer:
    """
    Training manager for LSTMABAR model
    """
    
    def __init__(
        self,
        model: LSTMABAR,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        self.model = model
        self.loss_weights = loss_weights
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'contrastive_loss': [],
            'archetype_loss': []
        }
    
    def train_step(
        self,
        descriptions: List[str],
        audio: torch.Tensor,
        target_weights: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step
        
        Returns:
            Dict of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(
            descriptions,
            audio,
            target_archetype_weights=target_weights
        )
        
        # Compute total loss
        total_loss = self.model.compute_total_loss(
            output['losses'],
            self.loss_weights
        )
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Return loss values
        losses = {
            'total': total_loss.item(),
            'contrastive': output['losses']['contrastive'].item() if output['losses']['contrastive'] is not None else 0.0,
            'archetype': output['losses']['archetype_prediction'].item() if output['losses']['archetype_prediction'] is not None else 0.0
        }
        
        return losses
    
    def validate(
        self,
        descriptions: List[str],
        audio: torch.Tensor,
        target_weights: torch.Tensor
    ) -> Dict[str, float]:
        """
        Validation step
        
        Returns:
            Dict of validation loss values
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            output = self.model(
                descriptions,
                audio,
                target_archetype_weights=target_weights
            )
            
            # Compute total loss
            total_loss = self.model.compute_total_loss(
                output['losses'],
                self.loss_weights
            )
            
            # Return loss values
            losses = {
                'total': total_loss.item(),
                'contrastive': output['losses']['contrastive'].item() if output['losses']['contrastive'] is not None else 0.0,
                'archetype': output['losses']['archetype_prediction'].item() if output['losses']['archetype_prediction'] is not None else 0.0
            }
        
        return losses
    
    def train_epoch(self, dataloader, epoch: int):
        """Train for one epoch"""
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            descriptions = batch['descriptions']
            audio = batch['audio'].to(self.model.device)
            target_weights = batch['archetype_weights'].to(self.model.device)
            
            # Training step
            losses = self.train_step(descriptions, audio, target_weights)
            epoch_losses.append(losses)
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}: "
                      f"Loss={losses['total']:.4f}, "
                      f"Contrastive={losses['contrastive']:.4f}, "
                      f"Archetype={losses['archetype']:.4f}")
        
        # Average losses
        avg_losses = {
            key: np.mean([l[key] for l in epoch_losses])
            for key in epoch_losses[0].keys()
        }
        
        # Update history
        self.history['train_loss'].append(avg_losses['total'])
        self.history['contrastive_loss'].append(avg_losses['contrastive'])
        self.history['archetype_loss'].append(avg_losses['archetype'])
        
        # Step scheduler
        self.scheduler.step()
        
        return avg_losses
    
    def validate_epoch(self, dataloader):
        """Validate for one epoch"""
        epoch_losses = []
        
        for batch in dataloader:
            descriptions = batch['descriptions']
            audio = batch['audio'].to(self.model.device)
            target_weights = batch['archetype_weights'].to(self.model.device)
            
            # Validation step
            losses = self.validate(descriptions, audio, target_weights)
            epoch_losses.append(losses)
        
        # Average losses
        avg_losses = {
            key: np.mean([l[key] for l in epoch_losses])
            for key in epoch_losses[0].keys()
        }
        
        # Update history
        self.history['val_loss'].append(avg_losses['total'])
        
        return avg_losses


# Example usage and testing
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize LSTMABAR model
    print("\n=== Initializing LSTMABAR Model ===")
    model = LSTMABAR(
        embedding_dim=768,
        audio_architecture='resnet',
        num_archetypes=5,
        sample_rate=44100,
        use_quantum_attention=False,
        temperature=0.07,
        device=device
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    
    # Dummy data
    batch_size = 4
    descriptions = [
        "bright and crunchy guitar",
        "warm smooth piano melody",
        "harsh digital synth",
        "soft mellow vocals"
    ]
    
    audio_length = 44100 * 2  # 2 seconds
    dummy_audio = torch.randn(batch_size, audio_length).to(device)
    
    # Target archetype weights (from MusicCaps training data)
    target_weights = torch.tensor([
        [0.1, 0.15, 0.5, 0.15, 0.1],   # Bright guitar
        [0.6, 0.05, 0.1, 0.2, 0.05],   # Warm piano
        [0.1, 0.55, 0.15, 0.1, 0.1],   # Digital synth
        [0.5, 0.1, 0.1, 0.2, 0.1]      # Soft vocals
    ]).to(device)
    
    # Forward pass
    output = model(
        descriptions,
        dummy_audio,
        target_archetype_weights=target_weights,
        return_intermediates=True
    )
    
    print(f"\nOutput keys: {output.keys()}")
    print(f"Transformed audio shape: {output['transformed_audio'].shape}")
    print(f"Predicted weights shape: {output['predicted_weights'].shape}")
    
    print("\nPredicted archetype weights:")
    for i, desc in enumerate(descriptions):
        print(f"{i+1}. '{desc}'")
        print(f"   Predicted: {output['predicted_weights'][i].detach().cpu().numpy()}")
        print(f"   Target:    {target_weights[i].cpu().numpy()}")
    
    print("\nLosses:")
    for loss_name, loss_value in output['losses'].items():
        if loss_value is not None:
            print(f"  {loss_name}: {loss_value.item():.4f}")
    
    # Test inference mode
    print("\n=== Testing Inference Mode ===")
    test_descriptions = ["bright metallic sound with some distortion"]
    test_audio = torch.randn(1, 44100 * 2).to(device)
    
    transformed, metadata = model.inference(test_descriptions, test_audio)
    
    print(f"Inference input shape: {test_audio.shape}")
    print(f"Inference output shape: {transformed.shape}")
    print(f"Predicted archetype mixture:")
    for name, weight in zip(metadata['archetype_names'], metadata['predicted_weights'][0]):
        print(f"  {name}: {weight:.4f}")
    
    # Test trainer
    print("\n=== Testing Trainer ===")
    trainer = LSTMABARTrainer(
        model,
        learning_rate=1e-4,
        loss_weights={'contrastive': 0.7, 'archetype_prediction': 0.3}
    )
    
    # Simulate training step
    losses = trainer.train_step(descriptions[:2], dummy_audio[:2], target_weights[:2])
    print(f"Training step losses: {losses}")
    
    # Test checkpoint saving
    print("\n=== Testing Checkpoint Save/Load ===")
    model.save_checkpoint('test_checkpoint.pth', epoch=0, optimizer_state=trainer.optimizer.state_dict())
    
    # Create new model and load
    model2 = LSTMABAR(
        embedding_dim=768,
        audio_architecture='resnet',
        device=device
    )
    optimizer_state = model2.load_checkpoint('test_checkpoint.pth', load_optimizer=True)
    
    print("Checkpoint loaded successfully!")
    
    print("\n=== LSTMABAR Model Test Complete ===")