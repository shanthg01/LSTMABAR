"""
Text Tower - NLP Encoding of Sonic Semantics
Encodes natural language descriptions into semantic embeddings
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import pennylane as qml
import numpy as np
from typing import List, Dict, Optional

class TextEncoder(nn.Module):
    """
    Text tower for encoding sonic descriptions into semantic embeddings

    Architecture:
    - BERT/RoBERTa backbone (via SentenceTransformer)
    - Projection layer to match audio embedding dimension
    - Optional quantum attention augmentation
    """

    def __init__(
        self,
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        embedding_dim=768,
        use_quantum_attention=False,
        n_qubits=8,
        circuit_depth=2,
        dropout_rate=0.3, 
        noise_strength=0.1,
        device='cpu'
    ):
        super(TextEncoder, self).__init__()

        self.device = device
        self.embedding_dim = embedding_dim
        self.use_quantum_attention = use_quantum_attention

        # Load pretrained sentence transformer
        print(f"Loading text encoder: {model_name}")
        from sentence_transformers import SentenceTransformer
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

        # Optional quantum attention block - NOW WITH ALL PARAMETERS
        if use_quantum_attention:
            self.quantum_attention = QuantumAttentionBlock(
                embedding_dim=embedding_dim,
                n_qubits=n_qubits,
                dropout_rate=dropout_rate,
                noise_strength=noise_strength,
                circuit_depth=circuit_depth
            )
        else:
            self.quantum_attention = None

        self.to(device)

    def forward(self, texts):
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

    def encode_aspects(self, aspect_lists):
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


# class QuantumAttentionBlock(nn.Module):
#     """
#     Quantum-inspired attention mechanism for enhanced semantic entanglement
    
#     Hybrid VQC encoder using PennyLane for quantum circuit simulation
#     """
    
#     def __init__(self, embedding_dim: int, n_qubits: int = 8):
#         super(QuantumAttentionBlock, self).__init__()
        
#         self.embedding_dim = embedding_dim
#         self.n_qubits = n_qubits
        
#         # Create quantum device
#         self.dev = qml.device("default.qubit", wires=n_qubits)
        
#         # Projection layers to/from quantum circuit
#         self.to_quantum = nn.Linear(embedding_dim, n_qubits)
#         self.from_quantum = nn.Linear(n_qubits, embedding_dim)
        
#         # Layer normalization for stability
#         self.norm = nn.LayerNorm(embedding_dim)
        
#         # Create quantum node
#         @qml.qnode(self.dev, interface="torch")
#         def quantum_circuit(inputs):
#             """
#             Quantum self-attention circuit with:
#             - Basis embedding
#             - Parameterized rotation layers
#             - Entanglement via controlled gates
#             - QFT for frequency analysis
#             - Grover operator for amplitude amplification
#             """
#             # Prepare inputs for basis embedding (binary encoding)
#             bi_inputs = torch.sigmoid(inputs)  # Normalize to [0, 1]
#             bi_inputs = (bi_inputs > 0.5).float()  # Binarize
            
#             # Basis embedding
#             qml.BasisEmbedding(features=bi_inputs.detach().numpy(), wires=range(self.n_qubits))
            
#             # Parameterized rotation layers with entanglement
#             for layer in range(3):
#                 # Single qubit rotations
#                 for i in range(self.n_qubits):
#                     qml.RX(inputs[i % len(inputs)] * (layer + 1), wires=i)
#                     qml.RY(inputs[(i + 1) % len(inputs)] * (layer + 1), wires=i)
#                     qml.RZ(inputs[(i + 2) % len(inputs)] * (layer + 1), wires=i)
                
#                 # Entangling layers
#                 for i in range(self.n_qubits - 1):
#                     qml.CRZ(np.pi / (layer + 2), wires=[i, (i + 1) % self.n_qubits])
#                     qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            
#             # Quantum Fourier Transform and its inverse
#             qml.QFT(wires=range(self.n_qubits))
#             qml.adjoint(qml.QFT)(wires=range(self.n_qubits))
            
#             # Grover operator for amplitude amplification
#             qml.GroverOperator(wires=range(self.n_qubits))
            
#             # Final rotation and basis re-embedding
#             for i in range(self.n_qubits):
#                 qml.Hadamard(wires=i)
#                 qml.T(wires=i)
#                 qml.RZ(inputs[i % len(inputs)], wires=i)
            
#             qml.BasisEmbedding(features=bi_inputs.detach().numpy(), wires=range(self.n_qubits))
            
#             # Measure expectation values
#             return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
        
#         self.quantum_circuit = quantum_circuit
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Apply quantum attention to embeddings
        
#         Args:
#             x: Input embeddings of shape (batch_size, embedding_dim)
            
#         Returns:
#             Quantum-enhanced embeddings of shape (batch_size, embedding_dim)
#         """
#         batch_size = x.shape[0]
        
#         # Store residual for skip connection
#         residual = x
        
#         # Project to quantum circuit dimension
#         quantum_inputs = self.to_quantum(x)  # (batch_size, n_qubits)
        
#         # Process each sample through quantum circuit
#         quantum_outputs = []
#         for i in range(batch_size):
#             circuit_input = quantum_inputs[i]
#             # Execute quantum circuit
#             output = self.quantum_circuit(circuit_input)
#             quantum_outputs.append(torch.stack(output))
        
#         quantum_outputs = torch.stack(quantum_outputs).float()  # (batch_size, n_qubits) - convert to float32
        
#         # Project back to embedding dimension
#         enhanced = self.from_quantum(quantum_outputs)
        
#         # Residual connection and normalization
#         output = self.norm(enhanced + residual)
        
#         return output

class QuantumAttentionBlock(nn.Module):
    """
    Enhanced quantum attention with regularization to prevent overfitting
    """
    
    def __init__(
        self, 
        embedding_dim: int, 
        n_qubits: int = 8,
        dropout_rate: float = 0.3,
        noise_strength: float = 0.1,
        circuit_depth: int = 2  # Reduced from 3 to prevent overfitting
    ):
        super(QuantumAttentionBlock, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        self.noise_strength = noise_strength
        self.circuit_depth = circuit_depth
        
        # Create quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Deeper projection with MORE regularization
        self.to_quantum = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.Dropout(dropout_rate),
            nn.GELU(),  # Smoother activation
            nn.Linear(embedding_dim // 2, n_qubits),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        self.from_quantum = nn.Sequential(
            nn.Linear(n_qubits, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(embedding_dim)
        
        # Learnable gate parameter for adaptive residual mixing
        self.gate = nn.Parameter(torch.tensor(0.3))  # Start with 30% quantum influence
        
        # Create quantum circuit
        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs):
            """Simplified quantum circuit to reduce overfitting"""
            
            # Add input noise during training for regularization
            if self.training:
                noise = torch.randn_like(inputs) * self.noise_strength
                inputs = inputs + noise
            
            # Prepare binary inputs for basis embedding
            bi_inputs = torch.sigmoid(inputs)
            bi_inputs = (bi_inputs > 0.5).float()
            
            # Basis embedding
            qml.BasisEmbedding(features=bi_inputs.detach().numpy(), wires=range(self.n_qubits))
            
            # Reduced parameterized layers (prevents memorization)
            for layer in range(self.circuit_depth):
                # Single qubit rotations
                for i in range(self.n_qubits):
                    qml.RX(inputs[i % len(inputs)] * (layer + 1), wires=i)
                    qml.RY(inputs[(i + 1) % len(inputs)] * (layer + 1), wires=i)
                    qml.RZ(inputs[(i + 2) % len(inputs)] * (layer + 1), wires=i)
                
                # Entangling layers
                for i in range(self.n_qubits - 1):
                    qml.CRZ(np.pi / (layer + 2), wires=[i, (i + 1) % self.n_qubits])
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            
            # Simplified final layer (removed QFT and Grover to reduce capacity)
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(inputs[i % len(inputs)], wires=i)
            
            # Measure expectation values
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
        
        self.quantum_circuit = quantum_circuit
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quantum attention to embeddings
        
        Args:
            x: Input embeddings of shape (batch_size, embedding_dim)
            
        Returns:
            Quantum-enhanced embeddings of shape (batch_size, embedding_dim)
        """
        batch_size = x.shape[0]
        residual = x
        
        # Project to quantum dimension
        quantum_inputs = self.to_quantum(x)
        
        # Process through quantum circuit
        quantum_outputs = []
        for i in range(batch_size):
            output = self.quantum_circuit(quantum_inputs[i])
            quantum_outputs.append(torch.stack(output))
        
        quantum_outputs = torch.stack(quantum_outputs).float()
        
        # Project back to embedding dimension
        enhanced = self.from_quantum(quantum_outputs)
        
        # Gated residual connection (learnable adaptive mixing)
        gate_sigmoid = torch.sigmoid(self.gate)
        output = self.norm(gate_sigmoid * enhanced + (1 - gate_sigmoid) * residual)
        
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