> **Archived v1 (W266 submission, tag `v1-w266`).** Kept for reference only. See [../docs/design.md](../docs/design.md) §1 for the audit of this version and the v2 redesign.

# Two-Tower Model Architecture: LSTMABAR (Language-to-Sound Transformation Model using Archetype-Based Audio Representation)

*Contributions: project ideation, project outline, literature review, architecture design, text_tower, audio_tower, contrastive_alignment, ddsp_transformation, archetype_predictor, musiccaps_loader, model training and experimentation*

## 1. High-Level Overview

The model transforms an input audio clip into a modified version consistent with a natural language description. It aligns **text semantics** with **audio timbral archetypes**, bridging subjective descriptors (“warm,” “crunchy”) and objective acoustic features (spectral centroid, harmonic ratio, MFCCs).

A **two-tower contrastive learning framework**—inspired by CLAP and MuLan—forms the core. The **text tower** encodes language semantics, while the **audio tower** embeds audio features. Joint embeddings feed into an **archetype prediction head**, which reconstructs sound via differentiable DSP.

---

## 2. Model Architecture Diagram (Conceptual Flow)

**Text Input** (“warm, crunchy guitar”)  
→ Text Encoder (BERT / quantum-augmented)  
→ Text Embedding (`dₜ`)

**Audio Input** (WAV file)  
→ Audio Encoder (ResNet / Audio Spectrogram Transformer)  
→ Audio Embedding (`dₐ`)

→ Contrastive Alignment (CLAP-style loss)  
→ Joint Embedding Fusion Layer  
→ Archetype Mixture Predictor (MLP regression head)  
→ Transformation Engine (DDSP-based synthesis)  
→ Output Audio

---

## 3. Detailed Component Breakdown

### Text Tower — NLP Encoding of Sonic Semantics

- **Model:** Sentence Transformer (BERT or RoBERTa backbone) fine-tuned on augmented MusicCaps-like captions emphasizing timbral descriptors.
- **Processing:**
  - Token embeddings projected to match audio embedding dimensions.
  - **Optional:** Quantum Attention Block (hybrid VQC encoder) for entanglement between co-occurring descriptors.
- **Output:** Fixed-size semantic vector `dt ∈ R^768`

### Audio Tower — Spectrogram-Aware Acoustic Encoding

- **Input:** Log-mel spectrograms or CQT-transformed audio segments.
- **Backbones:**
  - ResNet-18 / EfficientNet trained on DDSP or synthetic archetype data.
  - Audio Spectrogram Transformer (AST) using time–frequency patch embeddings.
- **Auxiliary Prediction:** Predict activation for five archetypes (sine, square, sawtooth, triangle, noise).
- **Output:** Audio embedding `da ∈ R^768`, dimensionally matched with text tower.

### Alignment Layer — Contrastive Learning and Matching

- **Loss Function:**

  \[
  L_{contrastive} = -\log \frac{\exp(sim(d_t, d_a^+)/\tau)}{\sum_{j=1}^{N}\exp(sim(d_t, d_{a_j})/\tau)}
  \]

  where `sim` = cosine similarity, `τ` = temperature.

- **Negative Sampling:** Large batches (N > 128), MuLan structure.
- **Auxiliary Match Classifier:**
  
  \[
  L_{aux} = -[y \log p + (1-y)\log(1-p)]
  \]

### Archetype Prediction Head — Mixture Weight Estimator

- **Inputs:** Concatenated joint embedding `[dt; da]`
- **Architecture:** Two-layer MLP.
- **Outputs:** Mixture vector `w ∈ R^5`, softmax-normalized for archetype contributions.
- **RLHF Integration:** Feedback-based policy gradient fine-tuning for perceptual consistency.

---

## 4. Differentiable Audio Transformation Pipeline

- **Archetype mixture vector** conditions differentiable DSP block:

  - **DDSP Module (Engel et al., 2020):**
    - Harmonic oscillator bank, weighted by predicted archetype contributions.
    - Learnable filters for “brightness” and “warmth.”
    - Harmonic Distortion Layer for nonlinear timbres like “crunchy.”

  - **Processing Steps:**
    - Decompose input audio into harmonic, noise, residual tracks.
    - Apply gain/filtering per archetype weights.
    - Reconstruct waveform via inverse STFT or DDSP synthesizer.
    - Output waveform time-aligned for A/B comparison.

---

## 5. Adaptive Learning and Feedback Loop

- **Synonym Learner:** Maps new descriptive terms to known attributes via embedding space clustering.
- **Feedback Collector:** Gathers user corrections, incrementally updates mappings.
- **Online Fine-Tuning:** Updates model with gradient steps from user feedback.

---

## 6. Training and Evaluation Framework

- **Pretraining:** Joint on synthetic archetype dataset, MusicCaps, DDSP samples.  
  - 70% contrastive learning, 30% archetype supervision.
- **Fine-tuning:** On paired real-world descriptors (FreeSound, DAW forums) with RLHF signals.
- **Evaluation Metrics:**
  - Semantic Textual Similarity (STS)
  - Spectral Centroid Error
  - MFCC Cosine Similarity
  - Archetype Classification Accuracy
  - Human Evaluations (Likert scale)

---

## 7. Optional Quantum Attention Extension

- **VQC Integration:** Encodes sentence tokens into Hilbert space.
- **Benefits:** Nonlinear entanglement of co-occurring adjectives, quantum kernel similarity for nuanced mappings.

---

## 8. Summary of Data Flow

1. **Input:** (Audio, Text)
2. Encode audio → `da`
3. Encode text → `dt`
4. Align via contrastive loss.
5. Merge embeddings → predict archetype weights `w`
6. Transform input audio with DDSP engine under `w`
7. Output transformed waveform.
8. Collect feedback → update classifier (RLHF stage)
