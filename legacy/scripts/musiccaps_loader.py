import pandas as pd
import numpy as np
import os
import yt_dlp
from pathlib import Path
import json
import librosa
import soundfile as sf

class MusicCapsLoader:
    """Load and process MusicCaps dataset for audio NLP project"""
    
    def __init__(self, csv_path, audio_dir='./musiccaps_audio'):
        """
        Args:
            csv_path: Path to musiccaps-public.csv from Kaggle
            audio_dir: Directory to save downloaded audio clips
        """
        self.csv_path = csv_path
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(exist_ok=True)
        
        # Load dataset
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} MusicCaps examples")
        
    def download_audio_clips(self, max_clips=100, use_balanced_subset=True):
        """
        Download audio clips from YouTube
        
        Args:
            max_clips: Maximum number of clips to download
            use_balanced_subset: If True, only download from balanced 1k subset
        """
        # Filter to balanced subset if requested
        if use_balanced_subset and 'is_balanced_subset' in self.df.columns:
            subset = self.df[self.df['is_balanced_subset'] == True]
            print(f"Using balanced subset: {len(subset)} examples")
        else:
            subset = self.df
        
        # Limit number of clips
        subset = subset.head(max_clips)
        
        downloaded = []
        failed = []
        
        for idx, row in subset.iterrows():
            ytid = row['ytid']
            start_s = row['start_s']
            end_s = row['end_s']
            
            output_path = self.audio_dir / f"{ytid}_{start_s}_{end_s}.wav"
            
            # Skip if already downloaded
            if output_path.exists():
                print(f"✓ Already exists: {ytid}")
                downloaded.append(str(output_path))
                continue
            
            # Download clip
            success = self._download_clip(ytid, start_s, end_s, output_path)
            
            if success:
                downloaded.append(str(output_path))
                print(f"✓ Downloaded: {ytid} ({len(downloaded)}/{max_clips})")
            else:
                failed.append(ytid)
                print(f"✗ Failed: {ytid}")
        
        print(f"\n=== Download Summary ===")
        print(f"Successfully downloaded: {len(downloaded)}")
        print(f"Failed: {len(failed)}")
        
        return downloaded, failed
    
    def _download_clip(self, ytid, start_s, end_s, output_path):
        """Download a single YouTube clip segment"""
        url = f"https://www.youtube.com/watch?v={ytid}"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': str(output_path.with_suffix('')),
            'quiet': True,
            'no_warnings': True,
            'external_downloader': 'ffmpeg',
            'external_downloader_args': [
                '-ss', str(start_s),
                '-to', str(end_s),
            ],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return True
        except Exception as e:
            print(f"Error downloading {ytid}: {e}")
            return False
    
    def extract_audio_descriptors(self, save_path='musiccaps_descriptors.json'):
        """
        Extract and organize audio descriptors from MusicCaps
        
        Creates mapping between:
        - Aspect lists (comma-separated descriptors)
        - Full captions (multi-sentence descriptions)
        - Audio file paths
        """
        descriptors = []
        
        for idx, row in self.df.iterrows():
            ytid = row['ytid']
            audio_path = self.audio_dir / f"{ytid}_{row['start_s']}_{row['end_s']}.wav"
            
            # Only include if audio file exists
            if not audio_path.exists():
                continue
            
            # Extract aspect list (comma-separated)
            aspects = row['aspect_list'].strip('[]').replace('"', '').split(', ')
            
            descriptor_entry = {
                'audio_path': str(audio_path),
                'ytid': ytid,
                'aspect_list': aspects,
                'caption': row['caption'],
                'audioset_labels': row['audioset_names'].strip('[]').replace('"', '').split(', '),
            }
            
            descriptors.append(descriptor_entry)
        
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(descriptors, f, indent=2)
        
        print(f"Extracted {len(descriptors)} audio-descriptor pairs")
        print(f"Saved to {save_path}")
        
        return descriptors
    
    def create_archetype_training_data(self, use_tfidf_weighting=True):
        """
        Map MusicCaps descriptors to archetype weight vectors
        
        Returns training data with normalized archetype mixtures as vectors
        Each example has a weight for ALL archetypes (not just one-hot)
        
        Args:
            use_tfidf_weighting: If True, use term frequency weighting for keywords
        
        Returns:
            List of dicts with 'description', 'archetype_vector', 'archetype_dict'
        """
        # Define mapping from MusicCaps descriptors to archetypes
        # More comprehensive keyword lists for better matching
        archetype_keywords = {
            'sine': {
                'primary': ['smooth', 'pure', 'mellow', 'soft', 'gentle', 'warm', 'round', 'clean', 
                           'flowing', 'sustained', 'continuous'],
                'secondary': ['calm', 'peaceful', 'subtle', 'delicate', 'simple', 'fundamental']
            },
            'square': {
                'primary': ['digital', 'electronic', 'synthetic', 'retro', 'robotic', 'buzzy', 
                           'harsh', 'angular', 'arcade', '8-bit', 'chip'],
                'secondary': ['mechanical', 'artificial', 'pixelated', 'video game', 'computer']
            },
            'sawtooth': {
                'primary': ['bright', 'sharp', 'cutting', 'metallic', 'aggressive', 'brassy', 
                           'piercing', 'shrill', 'edgy', 'intense'],
                'secondary': ['energetic', 'powerful', 'punchy', 'bold', 'strong', 'dominant']
            },
            'triangle': {
                'primary': ['hollow', 'wooden', 'flute', 'clarinet', 'muted', 'filtered', 
                           'subdued', 'thin', 'nasal', 'reedy'],
                'secondary': ['breathy', 'airy', 'light', 'sparse', 'minimal']
            },
            'noise': {
                'primary': ['distorted', 'noisy', 'rough', 'fuzzy', 'grainy', 'textured', 
                           'chaotic', 'static', 'gritty', 'crunchy', 'crackling'],
                'secondary': ['harsh', 'abrasive', 'raw', 'dirty', 'lo-fi', 'saturated']
            }
        }
        
        training_data = []
        
        for idx, row in self.df.iterrows():
            audio_path = self.audio_dir / f"{row['ytid']}_{row['start_s']}_{row['end_s']}.wav"
            
            if not audio_path.exists():
                continue
            
            # Extract aspects and caption
            aspects = row['aspect_list'].strip('[]').replace('"', '').lower().split(', ')
            caption = row['caption'].lower()
            
            # Combine all text for analysis
            all_text = ' '.join(aspects) + ' ' + caption
            
            # Calculate archetype scores with primary/secondary weighting
            archetype_scores = {arch: 0.0 for arch in archetype_keywords.keys()}
            
            for archetype, keyword_groups in archetype_keywords.items():
                # Primary keywords get full weight
                for keyword in keyword_groups['primary']:
                    count = all_text.count(keyword)
                    if use_tfidf_weighting:
                        # Weight by keyword specificity (longer = more specific)
                        weight = len(keyword) / 10.0  # Normalize by average word length
                        archetype_scores[archetype] += count * weight * 2.0  # Primary multiplier
                    else:
                        archetype_scores[archetype] += count * 2.0
                
                # Secondary keywords get half weight
                for keyword in keyword_groups['secondary']:
                    count = all_text.count(keyword)
                    if use_tfidf_weighting:
                        weight = len(keyword) / 10.0
                        archetype_scores[archetype] += count * weight * 1.0  # Secondary multiplier
                    else:
                        archetype_scores[archetype] += count * 1.0
            
            # Normalize to create probability distribution (weights sum to 1)
            total = sum(archetype_scores.values())
            
            if total > 0:
                archetype_weights = {k: v/total for k, v in archetype_scores.items()}
            else:
                # Uniform distribution if no matches (neutral sound)
                archetype_weights = {k: 0.2 for k in archetype_keywords.keys()}
            
            # Create archetype vector in consistent order
            archetype_order = ['sine', 'square', 'sawtooth', 'triangle', 'noise']
            archetype_vector = np.array([archetype_weights[arch] for arch in archetype_order])
            
            # Verify vector sums to 1.0 (within floating point precision)
            assert abs(np.sum(archetype_vector) - 1.0) < 1e-6, "Archetype weights must sum to 1"
            
            training_data.append({
                'audio_path': str(audio_path),
                'description': row['caption'],
                'aspects': aspects,
                'archetype_dict': archetype_weights,  # Dict format for readability
                'archetype_vector': archetype_vector.tolist(),  # Vector format for training
                'archetype_order': archetype_order  # Order of vector elements
            })
        
        # Print statistics about archetype distributions
        self._print_archetype_statistics(training_data)
        
        return training_data
    
    def _print_archetype_statistics(self, training_data):
        """Print statistics about archetype weight distributions"""
        if not training_data:
            return
        
        print("\n=== Archetype Distribution Statistics ===")
        
        archetype_order = training_data[0]['archetype_order']
        vectors = np.array([item['archetype_vector'] for item in training_data])
        
        for i, archetype in enumerate(archetype_order):
            weights = vectors[:, i]
            print(f"\n{archetype.upper()}:")
            print(f"  Mean weight: {np.mean(weights):.3f}")
            print(f"  Std dev: {np.std(weights):.3f}")
            print(f"  Max weight: {np.max(weights):.3f}")
            print(f"  % samples with weight > 0.3: {(weights > 0.3).sum() / len(weights) * 100:.1f}%")
        
        # Find examples with dominant archetypes
        print("\n=== Examples with Dominant Archetypes ===")
        for archetype_idx, archetype in enumerate(archetype_order):
            dominant_idx = np.argmax(vectors[:, archetype_idx])
            dominant_weight = vectors[dominant_idx, archetype_idx]
            if dominant_weight > 0.4:  # Only show if significantly dominant
                example = training_data[dominant_idx]
                print(f"\n{archetype.upper()} dominant ({dominant_weight:.2f}):")
                print(f"  Description: {example['description'][:100]}...")
                print(f"  Vector: {[f'{w:.2f}' for w in example['archetype_vector']]}")
    
    def save_training_data(self, training_data, output_path='musiccaps_training_data.npz'):
        """
        Save training data in efficient numpy format
        
        Saves both the vectors (for neural network training) and metadata (for analysis)
        """
        # Extract components
        vectors = np.array([item['archetype_vector'] for item in training_data])
        descriptions = [item['description'] for item in training_data]
        audio_paths = [item['audio_path'] for item in training_data]
        
        # Save as compressed numpy archive
        np.savez_compressed(
            output_path,
            archetype_vectors=vectors,
            descriptions=descriptions,
            audio_paths=audio_paths,
            archetype_order=training_data[0]['archetype_order']
        )
        
        print(f"\nSaved training data to {output_path}")
        print(f"  Shape of archetype vectors: {vectors.shape}")
        print(f"  Number of examples: {len(descriptions)}")
        
        # Also save readable JSON version
        json_path = output_path.replace('.npz', '.json')
        with open(json_path, 'w') as f:
            json.dump(training_data[:100], f, indent=2)  # Save first 100 for inspection
        print(f"  Saved first 100 examples to {json_path} for inspection")
        
        return output_path
    
    def load_training_data(self, input_path='musiccaps_training_data.npz'):
        """Load saved training data"""
        data = np.load(input_path, allow_pickle=True)
        
        print(f"Loaded training data from {input_path}")
        print(f"  Archetype vectors shape: {data['archetype_vectors'].shape}")
        print(f"  Number of descriptions: {len(data['descriptions'])}")
        print(f"  Archetype order: {data['archetype_order']}")
        
        return {
            'vectors': data['archetype_vectors'],
            'descriptions': data['descriptions'].tolist(),
            'audio_paths': data['audio_paths'].tolist(),
            'archetype_order': data['archetype_order'].tolist()
        }
    
    def get_statistics(self):
        """Print dataset statistics"""
        print("\n=== MusicCaps Dataset Statistics ===")
        print(f"Total examples: {len(self.df)}")
        
        if 'is_balanced_subset' in self.df.columns:
            balanced = self.df[self.df['is_balanced_subset'] == True]
            print(f"Balanced subset: {len(balanced)}")
        
        if 'is_audioset_eval' in self.df.columns:
            eval_split = self.df[self.df['is_audioset_eval'] == True]
            train_split = self.df[self.df['is_audioset_eval'] == False]
            print(f"Eval split: {len(eval_split)}")
            print(f"Train split: {len(train_split)}")
        
        # Count downloaded audio files
        audio_files = list(self.audio_dir.glob('*.wav'))
        print(f"Downloaded audio files: {len(audio_files)}")
        
        # Sample aspects
        print("\n=== Sample Aspect Lists ===")
        for i in range(min(3, len(self.df))):
            aspects = self.df.iloc[i]['aspect_list']
            print(f"{i+1}. {aspects}")
        
        print("\n=== Sample Captions ===")
        for i in range(min(3, len(self.df))):
            caption = self.df.iloc[i]['caption']
            print(f"{i+1}. {caption[:150]}...")


# Example usage and workflow
if __name__ == "__main__":
    # Step 1: Load MusicCaps dataset
    loader = MusicCapsLoader('musiccaps-public.csv')
    
    # Step 2: Show statistics
    loader.get_statistics()
    
    # Step 3: Download audio clips (start with balanced subset)
    print("\n=== Downloading Audio Clips ===")
    downloaded, failed = loader.download_audio_clips(
        max_clips=50,  # Start small
        use_balanced_subset=True
    )
    
    # Step 4: Extract descriptors
    print("\n=== Extracting Descriptors ===")
    descriptors = loader.extract_audio_descriptors()
    
    # Step 5: Create archetype training data with weight vectors
    print("\n=== Creating Archetype Training Data ===")
    training_data = loader.create_archetype_training_data(use_tfidf_weighting=True)
    
    print(f"\nReady for training with {len(training_data)} examples!")
    
    # Step 6: Save training data for later use
    if training_data:
        loader.save_training_data(training_data, 'musiccaps_training_data.npz')
    
    # Example: Show first training samples
    if training_data:
        print("\n=== Sample Training Examples ===")
        for i in range(min(3, len(training_data))):
            sample = training_data[i]
            print(f"\n--- Example {i+1} ---")
            print(f"Description: {sample['description'][:120]}...")
            print(f"Archetype weights (dict): {sample['archetype_dict']}")
            print(f"Archetype vector: {[f'{w:.3f}' for w in sample['archetype_vector']]}")
            print(f"Vector sum: {sum(sample['archetype_vector']):.6f} (should be 1.0)")
    
    # Example: Load training data
    print("\n=== Testing Load Function ===")
    loaded = loader.load_training_data('musiccaps_training_data.npz')
    print(f"Loaded {loaded['vectors'].shape[0]} training examples")