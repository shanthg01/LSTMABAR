"""
Text Tower - NLP Encoding of Sonic Semantics
Encodes natural language descriptions into semantic embeddings
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional

class TextEncoder(nn.Module):
    """
    Text tower for encoding sonic descriptions into semantic embeddings
    
    Architecture:
    - BERT/RoBERTa backbone
    - Projection layer to match audio embedding dimension
    - Optional quantum attention augmentation
    """
    
    def __init__(
        self,
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        embedding_dim=768,
        use_quantum_attention=False,
        device='cpu'
    ):
        super(TextEncoder, self).__init__()
        
        self.device = device
        self.embedding_dim = embedding_dim
        self.use_quantum_attention = use_quantum_attention
        
        # Load pretrained sentence transformer
        print(f"Loading text encoder: {model_name}")
        self.sentence_model = SentenceTransformer(model_name)
        self.base_dim = self.sentence_model.get_sentence_embedding_dimension()
        
        # Projection layer to match audio embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(self.base_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Optional quantum attention block (placeholder for VQC integration)
        if use_quantum_attention:
            self.quantum_attention = QuantumAttentionBlock(embedding_dim)
        
        self.to(device)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text descriptions into embeddings
        
        Args:
            texts: List of text descriptions
        
        Returns:
            Text embeddings of shape (batch_size, embedding_dim)
        """
        # Get base embeddings from sentence transformer
        with torch.no_grad():
            base_embeddings = self.sentence_model.encode(
                texts, 
                convert_to_tensor=True,
                device=self.device
            )

        # Clone to make it a normal tensor for autograd
        base_embeddings = base_embeddings.clone().detach().requires_grad_(True)
        
        # Project to target dimension
        embeddings = self.projection(base_embeddings)
        
        # Optional quantum attention
        if self.use_quantum_attention:
            embeddings = self.quantum_attention(embeddings)
        
        # L2 normalize for contrastive learning
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def encode_aspects(self, aspect_lists: List[List[str]]) -> torch.Tensor:
        """
        Encode aspect lists (comma-separated descriptors from MusicCaps)
        
        Args:
            aspect_lists: List of aspect lists, e.g., [["bright", "metallic"], ["warm", "smooth"]]
        
        Returns:
            Aspect embeddings
        """
        # Join aspects into single strings
        texts = [", ".join(aspects) for aspects in aspect_lists]
        return self.forward(texts)


class QuantumAttentionBlock(nn.Module):
    """
    Quantum-inspired attention mechanism for enhanced semantic entanglement
    
    Placeholder for hybrid VQC encoder - can be extended with PennyLane/Qiskit
    """
    
    def __init__(self, embedding_dim: int):
        super(QuantumAttentionBlock, self).__init__()
        
        # Classical approximation of quantum attention
        # Replace with actual VQC when implementing quantum extension
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quantum-inspired attention
        
        In full quantum implementation, this would:
        1. Encode embeddings into quantum states
        2. Apply variational quantum circuit
        3. Measure and decode back to classical embeddings
        """
        # Self-attention (simulating quantum entanglement)
        x_unsqueezed = x.unsqueeze(1)  # Add sequence dimension
        attn_output, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        attn_output = attn_output.squeeze(1)
        
        # Residual connection and normalization
        output = self.layer_norm(x + attn_output)
        
        return output


class SemanticSynonymLearner:
    """
    Learns mappings for unknown descriptors via embedding similarity
    Handles adaptive vocabulary expansion
    """
    
    def __init__(self, text_encoder: TextEncoder):
        self.text_encoder = text_encoder
        
        # Core audio descriptor vocabulary
        self.known_descriptors = {
            'brightness': ['bright', 'dark', 'sharp', 'dull', 'brilliant'],
            'warmth': ['warm', 'cold', 'cool', 'hot', 'cozy'],
            'texture': ['smooth', 'rough', 'grainy', 'silky', 'crunchy'],
            'hardness': ['soft', 'hard', 'gentle', 'aggressive', 'harsh'],
            'density': ['thick', 'thin', 'fat', 'lean', 'full']
        }
        
        # Build embedding cache for known descriptors
        self.descriptor_embeddings = self._build_descriptor_cache()
    
    def _build_descriptor_cache(self) -> Dict[str, torch.Tensor]:
        """Build embeddings for all known descriptors"""
        cache = {}
        
        all_descriptors = []
        for category, descriptors in self.known_descriptors.items():
            all_descriptors.extend(descriptors)
        
        embeddings = self.text_encoder([desc for desc in all_descriptors])
        
        idx = 0
        for category, descriptors in self.known_descriptors.items():
            for desc in descriptors:
                cache[desc] = embeddings[idx]
                idx += 1
        
        return cache
    
    def find_similar_descriptors(
        self, 
        unknown_word: str, 
        top_k: int = 3,
        threshold: float = 0.6
    ) -> List[tuple]:
        """
        Find similar known descriptors for an unknown word
        
        Args:
            unknown_word: New descriptor not in vocabulary
            top_k: Number of similar words to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of (descriptor, similarity_score) tuples
        """
        # Encode unknown word
        unknown_embedding = self.text_encoder([unknown_word])[0]
        
        # Compute similarities to all known descriptors
        similarities = []
        for descriptor, embedding in self.descriptor_embeddings.items():
            sim = torch.cosine_similarity(
                unknown_embedding.unsqueeze(0),
                embedding.unsqueeze(0)
            ).item()
            
            if sim >= threshold:
                similarities.append((descriptor, sim))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def expand_vocabulary(self, new_descriptor: str, category: str):
        """
        Add new descriptor to vocabulary
        
        Args:
            new_descriptor: Descriptor to add
            category: Which category it belongs to
        """
        if category in self.known_descriptors:
            self.known_descriptors[category].append(new_descriptor)
            
            # Update cache
            embedding = self.text_encoder([new_descriptor])[0]
            self.descriptor_embeddings[new_descriptor] = embedding
            
            print(f"Added '{new_descriptor}' to {category} category")


class TextAugmentation:
    """
    Data augmentation for text descriptions
    Generates variations while preserving semantic meaning
    """
    
    def __init__(self):
        self.synonym_map = {
            'bright': ['brilliant', 'sharp', 'crisp', 'clear'],
            'warm': ['cozy', 'soft', 'mellow', 'rich'],
            'harsh': ['aggressive', 'abrasive', 'hard', 'rough'],
            'smooth': ['silky', 'gentle', 'soft', 'flowing'],
            'crunchy': ['gritty', 'textured', 'grainy', 'distorted']
        }
    
    def augment_description(self, text: str, num_variations: int = 3) -> List[str]:
        """
        Generate semantic variations of a description
        
        Args:
            text: Original description
            num_variations: Number of variations to generate
        
        Returns:
            List of augmented descriptions
        """
        words = text.lower().split()
        variations = [text]  # Include original
        
        for _ in range(num_variations):
            new_words = []
            for word in words:
                if word in self.synonym_map and np.random.random() > 0.5:
                    # Replace with synonym
                    new_words.append(np.random.choice(self.synonym_map[word]))
                else:
                    new_words.append(word)
            
            variations.append(' '.join(new_words))
        
        return variations


# Example usage
if __name__ == "__main__":
    # Initialize text encoder
    text_encoder = TextEncoder(
        embedding_dim=768,
        use_quantum_attention=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Encode some descriptions
    descriptions = [
        "bright and crunchy guitar",
        "warm smooth piano melody",
        "harsh digital synth"
    ]
    
    embeddings = text_encoder(descriptions)
    print(f"Text embeddings shape: {embeddings.shape}")
    print(f"Sample embedding norm: {torch.norm(embeddings[0]).item():.4f}")
    
    # Test synonym learner
    synonym_learner = SemanticSynonymLearner(text_encoder)
    similar = synonym_learner.find_similar_descriptors("sparkly", top_k=3)
    print(f"\nSimilar to 'sparkly': {similar}")
    
    # Test augmentation
    augmenter = TextAugmentation()
    variations = augmenter.augment_description("bright warm sound", num_variations=3)
    print(f"\nAugmented descriptions: {variations}")