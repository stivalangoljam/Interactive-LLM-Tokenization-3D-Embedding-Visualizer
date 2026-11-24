# Interactive LLM Tokenization & 3D Embedding Visualizer

An interactive educational web platform demonstrating how Large Language Models (LLMs) process natural human language: from raw text strings to Byte-Pair Encoding (BPE) tokenization, vocabulary integer IDs, 384-dimensional vector embeddings, and 3D spatial projections.

Designed for college-level AI/ML courses, demonstrations, and research exploration.

---

## 🚀 Quick Start

### Option 1: Open Directly in Your Browser (Zero Installation)
Simply double-click `index.html` or open it with Google Chrome:
```bash
open index.html
```

### Option 2: Run via Built-in Local HTTP Server
The application includes a lightweight, zero-dependency Node.js server:
```bash
npm start
# or
node server.js
```
Then navigate to: **`http://localhost:3000`** in Google Chrome.

---

## 🌟 Key Features & Architecture

### 1. Raw Text Input & Preset Demonstrations
- Large text input supporting arbitrary words, sentences, or paragraphs.
- Includes preset buttons for key linguistic and ML concepts:
  - `Hello world` (Classic baseline)
  - `Artificial Intelligence is amazing` (Multi-token BPE & semantic concepts)
  - `The king and queen are ruling the kingdom` (Semantic clustering & gender vector offsets)
  - `Unbelievable things happen unexpectedly` (Subword decomposition: `Un-bel-ievable`)
  - `The car is faster than the bicycle` (Domain similarity: transportation & velocity)
  - `Cat bat hat dog car` (Spelling vs Semantic similarity minimal pairs)
  - `King queen prince princess apple banana car vehicle` (Multi-cluster semantic space)

### 2. Genuine GPT-2 Byte-Pair Encoding (BPE) Tokenization
- Real pretrained GPT-2 tokenization boundaries (not basic whitespace splitting).
- Correctly preserves whitespace prefixes (rendered as `·` or `␣`) and punctuation.
- Displays metrics in real-time:
  - **Total Tokens**
  - **Character Count**
  - **Word Count**
  - **Compression Ratio** (Characters per Token)
- Assigns a consistent, deterministic neon color to each unique token across all 4 sections.

### 3. Vocabulary Mapping Table
- Displays:
  - **Pos #**: Sequential token order in the sentence.
  - **Token**: The decoded subword unit.
  - **Token ID**: Authentic GPT-2 vocabulary integer ID (from the 50,257 dictionary).
  - **Color**: Assigned visual indicator matching the 3D nodes and token chips.
  - **Embedding Dimension**: 384-dimensional representation.
- Includes real-time search filtering and cross-section hover/click highlighting.
- Educational callout explaining the fundamental distinction between arbitrary token IDs and continuous vector representations.

### 4. 3D Vector Embedding Space (Three.js)
- **3D Coordinate System**:
  - `+X Axis` (Neon Rose / PC 1)
  - `+Y Axis` (Neon Green / PC 2)
  - `+Z Axis` (Neon Cyan / PC 3)
  - Glowing 3D floor grid and origin marker.
- **Glowing Point Markers & Billboarded Labels**:
  - Word nodes with glowing additive halo shaders.
  - Billboarded 3D canvas sprites that always face the camera.
  - Interactive mouse raycasting: hover tooltips, click selection ring, auto-focus.
- **Two Distinct Similarity Modes**:
  - **Semantic Similarity Mode**: Uses 384-dimensional Sentence Transformer (`all-MiniLM-L6-v2`) vectors projected via **Principal Component Analysis (PCA)**. Words with shared conceptual meaning (e.g. *King & Queen*, *Car & Bicycle*) cluster in close proximity.
  - **Spelling Similarity Mode**: Uses **Normalized Levenshtein Edit Distance** projected via **Classical Multidimensional Scaling (MDS)**. Words with similar orthography (e.g. *Cat, Bat, Hat*) cluster tightly regardless of their meaning.
  - Smooth 3D animation (ease-in-out cubic lerp) when switching between modes.
- **Word Comparison Tool**:
  - Select two words or click points in 3D.
  - Calculates high-dimensional **Cosine Similarity** ($\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$) on original 384D vectors.
  - Calculates **Normalized Levenshtein Spelling Similarity** ($1.0 - \frac{d(A, B)}{\max(|A|, |B|)}$).
  - Renders an animated glowing **3D laser line** connecting the two points in space with a floating similarity badge.
  - Highlights whether words have high semantic similarity despite differing spellings, or vice versa.
- **Word Vector Profile Panel**:
  - Vector dimensions (384 Dimensions, Float32).
  - L2 Unit Norm ($1.000$) and Value Range ($[\min, \max]$).
  - Numerical vector preview (first 16 floating-point components).
  - Vector magnitude heatmap strip.
  - Top-K nearest neighbors in the active text ranked by cosine similarity.

### 5. Interactive End-to-End LLM Processing Pipeline
- Flowchart illustrating the 6 critical stages of an LLM:
  1. **Raw Text Input** (UTF-8 Strings)
  2. **Tokenization** (BPE Subword Decomposition)
  3. **Vocabulary IDs** (Discrete Integers $\in [0, V-1]$)
  4. **Vector Embeddings** (Continuous Dense Space $\mathbb{R}^{384}$)
  5. **Transformer Layers** (Multi-Head Self-Attention & Feed-Forward)
  6. **Output Prediction** (Softmax Probability Distribution over Vocabulary)
- Interactive buttons to jump smoothly to the corresponding section.

---

## 🔬 Mathematical Implementation

### 1. Dual Principal Component Analysis (Gram Matrix PCA)
Given $N$ embedding vectors $\mathbf{x}_i \in \mathbb{R}^{384}$:
1. Center vectors: $\tilde{\mathbf{X}} = \mathbf{X} - \boldsymbol{\mu}$.
2. Form the Gram matrix $\mathbf{K} = \tilde{\mathbf{X}} \tilde{\mathbf{X}}^T \in \mathbb{R}^{N \times N}$.
3. Solve for top 3 eigenvectors $\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3$ using Power Iteration with Hotelling deflation:
   $$\mathbf{K} \mathbf{q} = \lambda \mathbf{q}, \quad \mathbf{K}_{\text{deflated}} = \mathbf{K} - \lambda \mathbf{q} \mathbf{q}^T$$
4. Projected 3D coordinates:
   $$\mathbf{y}_i = \left[ u_1[i]\sqrt{\lambda_1}, \; u_2[i]\sqrt{\lambda_2}, \; u_3[i]\sqrt{\lambda_3} \right]$$
5. Variance explained:
   $$\text{Variance Ratio} = \frac{\lambda_1 + \lambda_2 + \lambda_3}{\text{Trace}(\mathbf{K})}$$

### 2. High-Dimensional Cosine Similarity
$$\text{Sim}_{\text{cosine}}(\mathbf{u}, \mathbf{v}) = \frac{\sum_{i=1}^{384} u_i v_i}{\sqrt{\sum_{i=1}^{384} u_i^2} \sqrt{\sum_{i=1}^{384} v_i^2}}$$

### 3. Normalized Levenshtein Spelling Similarity
$$\text{Sim}_{\text{spelling}}(s_1, s_2) = 1.0 - \frac{\text{lev}(s_1, s_2)}{\max(|s_1|, |s_2|)}$$

---

## 📁 Project Structure

```
project/
├── index.html          # Main HTML5 application interface
├── server.js           # Lightweight zero-dependency static HTTP server
├── package.json        # Project metadata and start scripts
├── test.mjs            # Automated verification & test suite
├── css/
│   └── styles.css      # Futuristic cyber dashboard styling & animations
└── js/
    ├── bundle.js       # Universal standalone bundle (runs via file:// or http://)
    ├── pca.js          # Mathematical PCA & Classical MDS implementation
    ├── tokenizer.js    # Genuine GPT-2 BPE tokenizer & vocabulary dictionary
    ├── embeddings.js   # 384D Sentence Transformer engine & similarity metrics
    ├── visualizer3d.js # Three.js 3D coordinate space visualization
    └── app.js          # Main application coordinator & cross-section sync
```
