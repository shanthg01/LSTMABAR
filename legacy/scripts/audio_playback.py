"""
Audio Playback Engine - helper class
**what does this do**
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List

def collect_feedback_with_audio(
        self,
        description: str,
        original_audio: np.ndarray,
        transformed_audio: np.ndarray,
        predicted_weights: np.ndarray,
        text_embedding: torch.Tensor,
        audio_embedding: torch.Tensor,
        sample_rate: int = 44100,
        auto_play: bool = True
    ) -> float:
        """
        Interactive feedback collection with audio playback

        Args:
            description: Text description used
            original_audio: Original input audio
            transformed_audio: Model's transformed audio
            predicted_weights: Predicted archetype weights
            text_embedding: Text embedding used
            audio_embedding: Audio embedding used
            sample_rate: Audio sample rate
            auto_play: Whether to auto-play audio (for Jupyter)

        Returns:
            User rating (1-5)
        """
        import IPython.display as ipd
        from IPython.display import display, HTML

        print("\n" + "="*60)
        print("HUMAN FEEDBACK COLLECTION")
        print("="*60)
        print(f"\nDescription: '{description}'")
        print(f"\nPredicted Archetype Weights:")
        archetype_names = ['sine', 'square', 'sawtooth', 'triangle', 'noise']
        for name, weight in zip(archetype_names, predicted_weights):
            bar = "█" * int(weight * 20)
            print(f"  {name:10s}: {bar} {weight:.3f}")

        print("\n" + "-"*60)
        print("AUDIO PLAYBACK")
        print("-"*60)

        # Display original audio
        print("\n▶️  ORIGINAL AUDIO:")
        display(ipd.Audio(original_audio, rate=sample_rate, autoplay=False))

        # Display transformed audio
        print("\n▶️  TRANSFORMED AUDIO:")
        display(ipd.Audio(transformed_audio, rate=sample_rate, autoplay=False))

        # Get user rating
        print("\n" + "-"*60)
        print("RATING INSTRUCTIONS")
        print("-"*60)
        print("Rate how well the transformation matches the description:")
        print("  5 = Perfect match")
        print("  4 = Good match")
        print("  3 = Acceptable match")
        print("  2 = Poor match")
        print("  1 = Very poor match")

        while True:
            try:
                rating_input = input("\nYour rating (1-5): ")
                rating = float(rating_input)
                if 1 <= rating <= 5:
                    break
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 5.")

        # Add feedback to buffer
        predicted_weights_tensor = torch.from_numpy(predicted_weights).float()
        self.add_feedback(
            text_embedding,
            audio_embedding,
            predicted_weights_tensor,
            rating
        )

        print(f"\n✓ Feedback recorded: {rating}/5")
        print("="*60 + "\n")

        return rating

    def batch_collect_feedback(
        self,
        samples: List[Dict],
        sample_rate: int = 44100,
        max_samples: Optional[int] = None
    ) -> List[float]:
        """
        Collect feedback for multiple samples

        Args:
            samples: List of dicts with 'description', 'original_audio',
                    'transformed_audio', 'predicted_weights', 'text_emb', 'audio_emb'
            sample_rate: Audio sample rate
            max_samples: Maximum number of samples to collect (None = all)

        Returns:
            List of ratings
        """
        ratings = []
        n_samples = min(len(samples), max_samples) if max_samples else len(samples)

        print(f"\n{'='*60}")
        print(f"BATCH FEEDBACK COLLECTION: {n_samples} samples")
        print(f"{'='*60}\n")

        for i, sample in enumerate(samples[:n_samples]):
            print(f"\n>>> Sample {i+1}/{n_samples}")

            rating = self.collect_feedback_with_audio(
                description=sample['description'],
                original_audio=sample['original_audio'],
                transformed_audio=sample['transformed_audio'],
                predicted_weights=sample['predicted_weights'],
                text_embedding=sample['text_embedding'],
                audio_embedding=sample['audio_embedding'],
                sample_rate=sample_rate
            )

            ratings.append(rating)

            # Option to stop early
            if i < n_samples - 1:
                continue_input = input("Continue to next sample? (y/n): ").lower()
                if continue_input != 'y':
                    print("Feedback collection stopped.")
                    break

        return ratings

    def save_audio_comparison(
        self,
        original_audio: np.ndarray,
        transformed_audio: np.ndarray,
        description: str,
        predicted_weights: np.ndarray,
        output_path: str,
        sample_rate: int = 44100
    ):
        """
        Save audio files for offline feedback collection

        Args:
            original_audio: Original audio
            transformed_audio: Transformed audio
            description: Text description
            predicted_weights: Predicted weights
            output_path: Base path for saving (without extension)
            sample_rate: Sample rate
        """
        import soundfile as sf
        import json

        # Save audio files
        sf.write(f"{output_path}_original.wav", original_audio, sample_rate)
        sf.write(f"{output_path}_transformed.wav", transformed_audio, sample_rate)

        # Save metadata
        metadata = {
            'description': description,
            'predicted_weights': {
                'sine': float(predicted_weights[0]),
                'square': float(predicted_weights[1]),
                'sawtooth': float(predicted_weights[2]),
                'triangle': float(predicted_weights[3]),
                'noise': float(predicted_weights[4])
            }
        }

        with open(f"{output_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Audio comparison saved to {output_path}_*.wav")