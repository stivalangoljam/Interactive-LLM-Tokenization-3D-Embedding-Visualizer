/**
 * Application Controller
 * Coordinates Tokenization, Vocabulary Table, 3D Three.js Visualization,
 * High-Dimensional Cosine Similarity, Levenshtein Distance, and Transformers.js.
 */

import { GPT2Tokenizer } from './tokenizer.js';
import { EmbeddingEngine } from './embeddings.js';
import { PCA } from './pca.js';
import { Visualizer3D } from './visualizer3d.js';

class App {
  constructor() {
    this.tokenizer = new GPT2Tokenizer();
    this.embeddingEngine = new EmbeddingEngine();
    this.visualizer = null;

    // State
    this.currentText = '';
    this.tokens = [];
    this.uniqueWords = [];
    this.activeMode = 'semantic'; // 'semantic' | 'spelling'
    this.selectedWord = null;
    this.compareWordA = null;
    this.compareWordB = null;
    this.cachedEmbeddings = new Map();
    this.pcaResult = null;
    this.mdsResult = null;

    this._cacheDOMElements();
    this._initVisualizer();
    this._bindEvents();
    this._initTransformers();

    // Load initial default example
    this.loadExample('The king and queen are ruling the kingdom');
  }

  _cacheDOMElements() {
    // Inputs & Controls
    this.textInput = document.getElementById('text-input');
    this.processBtn = document.getElementById('btn-process');
    this.resetBtn = document.getElementById('btn-reset');
    this.exampleBtns = document.querySelectorAll('.example-btn');
    this.modelStatusBadge = document.getElementById('model-status-badge');
    this.interactiveTextStream = document.getElementById('interactive-text-stream');

    // Section 2: Tokens
    this.tokensContainer = document.getElementById('tokens-container');
    this.tokenCountStat = document.getElementById('stat-token-count');
    this.charCountStat = document.getElementById('stat-char-count');
    this.wordCountStat = document.getElementById('stat-word-count');
    this.ratioStat = document.getElementById('stat-ratio');

    // Section 3: Vocab Table
    this.vocabTableBody = document.getElementById('vocab-table-body');
    this.vocabSearchInput = document.getElementById('vocab-search');
    this.btnExportJSON = document.getElementById('btn-export-json');
    this.btnExportCSV = document.getElementById('btn-export-csv');

    // Section 4: 3D Visualization & Panels
    this.viewport3D = document.getElementById('viewport-3d');
    this.modeSemanticBtn = document.getElementById('mode-semantic');
    this.modeSpellingBtn = document.getElementById('mode-spelling');
    this.modeDescription = document.getElementById('mode-description');
    this.btnToggleHeatmap = document.getElementById('btn-toggle-heatmap');
    this.resetCameraBtn = document.getElementById('btn-reset-camera');
    this.autoRotateBtn = document.getElementById('btn-auto-rotate');
    this.toggleLabelsBtn = document.getElementById('btn-toggle-labels');
    this.toggleGridBtn = document.getElementById('btn-toggle-grid');
    this.varianceBadge = document.getElementById('variance-badge');
    this.tooltip3D = document.getElementById('tooltip-3d');

    // Comparison Panel
    this.compareSelectA = document.getElementById('compare-select-a');
    this.compareSelectB = document.getElementById('compare-select-b');
    this.cosineScoreVal = document.getElementById('cosine-score-val');
    this.cosineScoreBar = document.getElementById('cosine-score-bar');
    this.spellingScoreVal = document.getElementById('spelling-score-val');
    this.spellingScoreBar = document.getElementById('spelling-score-bar');
    this.comparisonInsight = document.getElementById('comparison-insight');

    // Word Info Panel
    this.wordInfoEmpty = document.getElementById('word-info-empty');
    this.wordInfoContent = document.getElementById('word-info-content');
    this.infoWordTitle = document.getElementById('info-word-title');
    this.infoTokenId = document.getElementById('info-token-id');
    this.infoVectorDim = document.getElementById('info-vector-dim');
    this.infoVectorNorm = document.getElementById('info-vector-norm');
    this.infoVectorMinMax = document.getElementById('info-vector-minmax');
    this.infoVectorPreview = document.getElementById('info-vector-preview');
    this.infoVectorHeatmap = document.getElementById('info-vector-heatmap');
    this.infoNeighborsList = document.getElementById('info-neighbors-list');

    // Modals
    this.matrixModal = document.getElementById('matrix-modal');
    this.btnCloseMatrix = document.getElementById('btn-close-matrix');
    this.matrixTableContainer = document.getElementById('matrix-table-container');

    this.pipelineModal = document.getElementById('pipeline-modal');
    this.btnClosePipelineModal = document.getElementById('btn-close-pipeline-modal');
    this.pipelineModalTitle = document.getElementById('pipeline-modal-title');
    this.pipelineModalIcon = document.getElementById('pipeline-modal-icon');
    this.pipelineModalBody = document.getElementById('pipeline-modal-body');
  }

  _initVisualizer() {
    this.visualizer = new Visualizer3D(this.viewport3D, {
      onNodeHover: (data, clientX, clientY) => this._handle3DHover(data, clientX, clientY),
      onNodeClick: (data) => this._handle3DClick(data)
    });
  }

  _bindEvents() {
    // Process & Reset
    this.processBtn.addEventListener('click', () => this.processCurrentInput());
    this.resetBtn.addEventListener('click', () => this.resetAll());

    // Enter key with Cmd/Ctrl or Shift
    this.textInput.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        this.processCurrentInput();
      }
    });

    // Real-time character/word count update on typing
    this.textInput.addEventListener('input', () => {
      this._updateInputStats();
    });

    // Example buttons
    this.exampleBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const text = btn.getAttribute('data-text');
        this.loadExample(text);
      });
    });

    // Similarity Modes
    this.modeSemanticBtn.addEventListener('click', () => this.setSimilarityMode('semantic'));
    this.modeSpellingBtn.addEventListener('click', () => this.setSimilarityMode('spelling'));

    // Camera & Scene Controls
    this.resetCameraBtn.addEventListener('click', () => this.visualizer.resetCamera());
    this.autoRotateBtn.addEventListener('click', () => {
      const active = this.visualizer.toggleAutoRotate();
      this.autoRotateBtn.classList.toggle('active', active);
    });
    this.toggleLabelsBtn.addEventListener('click', () => {
      this.visualizer.toggleLabels();
      this.toggleLabelsBtn.classList.toggle('active');
    });
    this.toggleGridBtn.addEventListener('click', () => {
      this.visualizer.toggleGrid();
      this.toggleGridBtn.classList.toggle('active');
    });

    // Comparison Selectors
    this.compareSelectA.addEventListener('change', () => {
      this.compareWordA = this.compareSelectA.value;
      this._updateComparisonTool();
    });
    this.compareSelectB.addEventListener('change', () => {
      this.compareWordB = this.compareSelectB.value;
      this._updateComparisonTool();
    });

    // Vocab Search
    this.vocabSearchInput.addEventListener('input', (e) => {
      this._filterVocabTable(e.target.value);
    });

    // Export Buttons
    if (this.btnExportJSON) {
      this.btnExportJSON.addEventListener('click', () => this._exportJSON());
    }
    if (this.btnExportCSV) {
      this.btnExportCSV.addEventListener('click', () => this._exportCSV());
    }

    // Heatmap Modal
    if (this.btnToggleHeatmap) {
      this.btnToggleHeatmap.addEventListener('click', () => this._openMatrixModal());
    }
    if (this.btnCloseMatrix) {
      this.btnCloseMatrix.addEventListener('click', () => {
        this.matrixModal.style.display = 'none';
      });
    }

    // Pipeline Deep Dive Modal
    if (this.btnClosePipelineModal) {
      this.btnClosePipelineModal.addEventListener('click', () => {
        this.pipelineModal.style.display = 'none';
      });
    }

    // Close modals on outside click
    window.addEventListener('click', (e) => {
      if (e.target === this.matrixModal) this.matrixModal.style.display = 'none';
      if (e.target === this.pipelineModal) this.pipelineModal.style.display = 'none';
    });

    // Pipeline Clickable Navigation & Deep-Dive
    document.querySelectorAll('.pipeline-step').forEach(step => {
      step.addEventListener('click', () => {
        const stageNum = step.querySelector('.step-num-pill')?.textContent || '';
        const title = step.querySelector('.step-title')?.textContent || '';
        const icon = step.querySelector('.step-icon')?.textContent || '🧠';
        this._openPipelineModal(stageNum, title, icon);
      });
    });
  }

  async _initTransformers() {
    this._updateModelBadge('Initializing Neural Models...', 'loading');
    try {
      const { pipeline, AutoTokenizer, env } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
      env.allowLocalModels = false;

      // 1. Live GPT-2 Tokenizer
      this._updateModelBadge('Loading GPT-2 Tokenizer...', 'loading');
      const tok = await AutoTokenizer.from_pretrained('Xenova/gpt2');
      this.tokenizer.setTransformersTokenizer(tok);

      // 2. Live all-MiniLM-L6-v2 Embeddings
      this._updateModelBadge('Loading MiniLM-L6-v2 Embeddings...', 'loading');
      const ext = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
        quantized: true,
        progress_callback: (p) => {
          if (p.status === 'progress' && p.total) {
            const pct = Math.round((p.loaded / p.total) * 100);
            this._updateModelBadge(`Downloading MiniLM: ${pct}%`, 'loading');
          }
        }
      });
      this.embeddingEngine.setExtractor(ext);
      this._updateModelBadge('Transformers.js Active: GPT-2 & MiniLM-L6-v2 (384D)', 'ready');

      // Re-process current text with live model for maximum precision
      if (this.currentText) {
        this.processCurrentInput(false);
      }
    } catch (err) {
      console.warn('[App] Transformers.js CDN loading bypassed/offline. High-precision built-in engines active.', err);
      this._updateModelBadge('Built-in Model Active: GPT-2 BPE & 384D Embeddings', 'ready');
    }
  }

  _updateModelBadge(text, state = 'ready') {
    if (!this.modelStatusBadge) return;
    this.modelStatusBadge.innerHTML = `
      <span class="status-dot ${state}"></span>
      <span class="status-text">${text}</span>
    `;
  }

  loadExample(text) {
    this.textInput.value = text;
    this.processCurrentInput();
  }

  resetAll() {
    this.textInput.value = '';
    this.currentText = '';
    this.tokens = [];
    this.uniqueWords = [];
    this.selectedWord = null;
    this.compareWordA = null;
    this.compareWordB = null;

    if (this.interactiveTextStream) {
      this.interactiveTextStream.innerHTML = '<span style="color: var(--text-dim);">(Enter text above and click Process Text)</span>';
    }
    this.tokensContainer.innerHTML = '<div class="empty-state">Enter text above and click "Process Text" to view tokens.</div>';
    this.vocabTableBody.innerHTML = '<tr><td colspan="5" class="empty-table">No vocabulary tokens generated yet.</td></tr>';
    this.tokenCountStat.textContent = '0';
    this.charCountStat.textContent = '0';
    this.wordCountStat.textContent = '0';
    this.ratioStat.textContent = '0.00';

    this.visualizer.updatePoints([], false);
    this.visualizer.setComparison(null, null);
    this._resetWordInfo();
    this._resetComparisonTool();
  }

  async processCurrentInput(animate3D = true) {
    const raw = this.textInput.value.trim();
    if (!raw) return;
    this.currentText = raw;

    this.processBtn.classList.add('loading');
    this.processBtn.disabled = true;

    try {
      // 1. Tokenize Text
      this.tokens = await this.tokenizer.tokenize(raw);

      // 2. Render Synchronized Section 1 Text Highlight Stream
      this._renderInteractiveTextStream();

      // 3. Render Section 2: Tokenization Results
      this._renderTokens();
      this._renderInputStats();

      // 4. Render Section 3: Vocabulary Table
      this._renderVocabTable();

      // 5. Extract Unique Words for 3D Embedding Space
      await this._processEmbeddingsAnd3D(animate3D);

      // 6. Populate Word Comparison Selectors
      this._populateComparisonSelectors();

      // 7. Default select first word for Info Panel
      if (this.uniqueWords.length > 0) {
        this.selectWord(this.uniqueWords[0].word);
      }
    } catch (err) {
      console.error('[App] Error processing text:', err);
    } finally {
      this.processBtn.classList.remove('loading');
      this.processBtn.disabled = false;
    }
  }

  _renderInteractiveTextStream() {
    if (!this.interactiveTextStream) return;
    this.interactiveTextStream.innerHTML = '';
    if (this.tokens.length === 0) {
      this.interactiveTextStream.textContent = this.currentText;
      return;
    }

    this.tokens.forEach(t => {
      const span = document.createElement('span');
      span.className = 'char-token-span';
      span.dataset.index = t.index;
      span.dataset.word = t.rawText.trim();
      span.textContent = t.rawText;
      span.style.color = t.color;
      span.style.borderBottomColor = t.color + '88';

      span.addEventListener('mouseenter', () => this._highlightCrossSections(t.rawText.trim(), t.index, true));
      span.addEventListener('mouseleave', () => this._highlightCrossSections(t.rawText.trim(), t.index, false));
      span.addEventListener('click', () => {
        this.selectWord(t.rawText.trim());
        this._scrollToVocabRow(t.index);
        this._highlightTokenChip(t.index);
      });

      this.interactiveTextStream.appendChild(span);
    });
  }

  _renderTokens() {
    this.tokensContainer.innerHTML = '';
    if (this.tokens.length === 0) return;

    this.tokens.forEach((t) => {
      const chip = document.createElement('div');
      chip.className = 'token-chip';
      chip.dataset.index = t.index;
      chip.dataset.word = t.rawText.trim();
      chip.style.borderColor = t.color;
      chip.style.setProperty('--token-glow', t.color);

      chip.innerHTML = `
        <span class="token-color-indicator" style="background-color: ${t.color}"></span>
        <span class="token-text">${this._escapeHTML(t.displayText)}</span>
        <span class="token-id-tag">#${t.id}</span>
      `;

      // Token Hover & Click Interactivity
      chip.addEventListener('mouseenter', () => this._highlightCrossSections(t.rawText.trim(), t.index, true));
      chip.addEventListener('mouseleave', () => this._highlightCrossSections(t.rawText.trim(), t.index, false));
      chip.addEventListener('click', () => {
        this.selectWord(t.rawText.trim());
        this._scrollToVocabRow(t.index);
      });

      this.tokensContainer.appendChild(chip);
    });
  }

  _renderInputStats() {
    const charCount = this.currentText.length;
    const tokenCount = this.tokens.length;
    const wordCount = this.currentText.trim().split(/\s+/).filter(Boolean).length;
    const ratio = tokenCount > 0 ? (charCount / tokenCount).toFixed(2) : '0.00';

    this.tokenCountStat.textContent = tokenCount;
    this.charCountStat.textContent = charCount;
    this.wordCountStat.textContent = wordCount;
    this.ratioStat.textContent = ratio;
  }

  _updateInputStats() {
    const raw = this.textInput.value;
    const charCount = raw.length;
    const wordCount = raw.trim().split(/\s+/).filter(Boolean).length;
    this.charCountStat.textContent = charCount;
    this.wordCountStat.textContent = wordCount;
  }

  _renderVocabTable() {
    this.vocabTableBody.innerHTML = '';
    if (this.tokens.length === 0) return;

    this.tokens.forEach((t) => {
      const tr = document.createElement('tr');
      tr.className = 'vocab-row';
      tr.dataset.index = t.index;
      tr.dataset.word = t.rawText.trim();

      tr.innerHTML = `
        <td class="col-pos">${t.index}</td>
        <td class="col-token">
          <span class="vocab-token-badge" style="border-left: 3px solid ${t.color}">
            ${this._escapeHTML(t.displayText)}
          </span>
        </td>
        <td class="col-id">
          <code class="token-id-code">${t.id}</code>
        </td>
        <td class="col-color">
          <div class="color-swatch-container">
            <span class="color-swatch" style="background-color: ${t.color}"></span>
            <span class="color-hex">${t.color}</span>
          </div>
        </td>
        <td class="col-dim">
          <span class="dim-badge">384-D</span>
        </td>
      `;

      tr.addEventListener('mouseenter', () => this._highlightCrossSections(t.rawText.trim(), t.index, true));
      tr.addEventListener('mouseleave', () => this._highlightCrossSections(t.rawText.trim(), t.index, false));
      tr.addEventListener('click', () => {
        this.selectWord(t.rawText.trim());
        this._highlightTokenChip(t.index);
      });

      this.vocabTableBody.appendChild(tr);
    });
  }

  _filterVocabTable(query) {
    const q = query.trim().toLowerCase();
    const rows = this.vocabTableBody.querySelectorAll('.vocab-row');

    rows.forEach(row => {
      const word = (row.dataset.word || '').toLowerCase();
      const id = row.querySelector('.token-id-code')?.textContent.toLowerCase() || '';
      if (!q || word.includes(q) || id.includes(q)) {
        row.style.display = '';
      } else {
        row.style.display = 'none';
      }
    });
  }

  _exportJSON() {
    const data = {
      text: this.currentText,
      timestamp: new Date().toISOString(),
      tokens: this.tokens.map(t => ({
        position: t.index,
        token: t.rawText,
        id: t.id,
        color: t.color
      })),
      embeddings: this.uniqueWords.map(u => ({
        word: u.word,
        vectorDimensions: u.vector ? u.vector.length : 0,
        vector: u.vector || []
      }))
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `llm-embeddings-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  _exportCSV() {
    let csv = 'Position,Token,Token_ID,Color\n';
    this.tokens.forEach(t => {
      const cleanToken = t.rawText.replace(/"/g, '""');
      csv += `${t.index},"${cleanToken}",${t.id},"${t.color}"\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vocabulary-table-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async _processEmbeddingsAnd3D(animate3D = true) {
    // Collect unique clean words from tokens (excluding pure punctuation where applicable)
    const wordMap = new Map();
    for (const t of this.tokens) {
      const clean = t.rawText.trim();
      if (!clean) continue;
      if (!wordMap.has(clean.toLowerCase())) {
        wordMap.set(clean.toLowerCase(), {
          word: clean,
          token: t,
          color: t.color
        });
      }
    }

    const uniqueList = Array.from(wordMap.values());
    this.uniqueWords = uniqueList;

    if (uniqueList.length === 0) {
      this.visualizer.updatePoints([], false);
      return;
    }

    // 1. Fetch / Compute 384D Embedding vectors for each unique word
    const vectors = [];
    for (const item of uniqueList) {
      const vec = await this.embeddingEngine.getEmbedding(item.word);
      item.vector = vec;
      vectors.push(vec);
      this.cachedEmbeddings.set(item.word.toLowerCase(), vec);
    }

    // 2. High-Dimensional PCA Projection (384D -> 3D)
    this.pcaResult = PCA.project(vectors, 3);
    const variancePct = (this.pcaResult.varianceExplained * 100).toFixed(1);
    this.varianceBadge.innerHTML = `
      <span class="badge-icon">⚡</span>
      <span>PCA Variance Explained: <strong>${variancePct}%</strong> (384D → 3D)</span>
    `;

    // 3. Normalized Levenshtein Spelling Distance Matrix & Classical MDS Projection
    const wordsOnly = uniqueList.map(u => u.word);
    const spellingDistMatrix = EmbeddingEngine.computeSpellingDistanceMatrix(wordsOnly);
    this.mdsResult = PCA.projectMDS(spellingDistMatrix, 3);

    // 4. Assign 3D Coordinates according to current mode
    this._applyCoordinatesTo3D(animate3D);
  }

  _applyCoordinatesTo3D(animate = true) {
    if (!this.uniqueWords || this.uniqueWords.length === 0) return;

    const coordsArray = this.activeMode === 'semantic'
      ? this.pcaResult.coordinates
      : this.mdsResult.coordinates;

    const points = this.uniqueWords.map((item, idx) => ({
      word: item.word,
      token: item.token,
      color: item.color,
      coords: coordsArray[idx] || [0, 0, 0],
      vector: item.vector
    }));

    this.visualizer.updatePoints(points, animate);
  }

  setSimilarityMode(mode) {
    if (this.activeMode === mode) return;
    this.activeMode = mode;

    this.modeSemanticBtn.classList.toggle('active', mode === 'semantic');
    this.modeSpellingBtn.classList.toggle('active', mode === 'spelling');

    if (mode === 'semantic') {
      this.modeDescription.innerHTML = `
        <strong>Mode: Semantic Similarity</strong> — Positions points using PCA projection of real 384-dimensional Sentence Transformer vectors. Words with related meanings (e.g. <em>King & Queen</em>) cluster together.
      `;
    } else {
      this.modeDescription.innerHTML = `
        <strong>Mode: Spelling Similarity</strong> — Positions points using Classical MDS on Normalized Levenshtein edit distance. Words with similar orthography (e.g. <em>Cat, Bat, Hat</em>) cluster together.
      `;
    }

    this._applyCoordinatesTo3D(true);
    this._updateComparisonTool();
  }

  _openMatrixModal() {
    if (!this.matrixModal || !this.matrixTableContainer) return;
    this.matrixModal.style.display = 'flex';

    if (this.uniqueWords.length === 0) {
      this.matrixTableContainer.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--text-dim);">No words processed yet.</div>';
      return;
    }

    const words = this.uniqueWords.map(u => u.word);
    const N = words.length;

    let html = '<table class="similarity-matrix-table"><thead><tr><th>Token</th>';
    words.forEach(w => {
      html += `<th>${this._escapeHTML(w)}</th>`;
    });
    html += '</tr></thead><tbody>';

    for (let i = 0; i < N; i++) {
      const wA = words[i];
      const vecA = this.cachedEmbeddings.get(wA.toLowerCase());
      html += `<tr><th>${this._escapeHTML(wA)}</th>`;

      for (let j = 0; j < N; j++) {
        const wB = words[j];
        const vecB = this.cachedEmbeddings.get(wB.toLowerCase());
        const cos = (vecA && vecB) ? EmbeddingEngine.cosineSimilarity(vecA, vecB) : 0;
        const spell = EmbeddingEngine.spellingSimilarity(wA, wB);

        // Heatmap cell color based on cosine similarity
        const normCos = Math.max(0, Math.min(1, (cos + 1) / 2));
        const bgAlpha = (normCos * 0.45).toFixed(2);
        const cellBg = i === j ? 'rgba(6, 182, 212, 0.3)' : `rgba(139, 92, 246, ${bgAlpha})`;

        html += `
          <td style="background-color: ${cellBg};" data-wa="${this._escapeHTML(wA)}" data-wb="${this._escapeHTML(wB)}" title="Compare: ${wA} vs ${wB}">
            <span class="matrix-cell-cos" style="color: ${cos > 0.5 ? '#22d3ee' : '#e2e8f0'}">${cos >= 0 ? '+' : ''}${cos.toFixed(2)}</span>
            <span class="matrix-cell-spell">(${spell.toFixed(2)})</span>
          </td>
        `;
      }
      html += '</tr>';
    }
    html += '</tbody></table>';

    this.matrixTableContainer.innerHTML = html;

    // Click cell to compare directly
    this.matrixTableContainer.querySelectorAll('td[data-wa]').forEach(td => {
      td.addEventListener('click', () => {
        const wa = td.dataset.wa;
        const wb = td.dataset.wb;
        this.compareWordA = wa;
        this.compareWordB = wb;
        if (this.compareSelectA) this.compareSelectA.value = wa;
        if (this.compareSelectB) this.compareSelectB.value = wb;
        this._updateComparisonTool();
        this.matrixModal.style.display = 'none';
        const el = document.getElementById('section-embeddings');
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      });
    });
  }

  _openPipelineModal(stageNum, title, icon) {
    if (!this.pipelineModal) return;
    this.pipelineModalTitle.textContent = `${stageNum}: ${title}`;
    this.pipelineModalIcon.textContent = icon;

    const descriptions = {
      'Stage 1': {
        summary: 'Raw natural language input is received as an arbitrary UTF-8 string containing characters, whitespace, and punctuation.',
        formula: 'S = [c_1, c_2, ..., c_L] \\quad c_i \\in \\text{Unicode}',
        details: 'Unlike traditional NLP systems that rely on strict dictionary lookup or handcrafted stemmers, modern LLMs must handle any arbitrary string, novel slang, typos, code, and multiple human languages without crashing on Out-Of-Vocabulary (OOV) errors.'
      },
      'Stage 2': {
        summary: 'Byte-Pair Encoding (BPE) segments human text into statistical subword units.',
        formula: '\\text{Merge Rank: } (u, v) \\to uv \\quad \\text{based on frequency in training corpus}',
        details: 'Common words remain single whole tokens (e.g. "Hello" -> #15496), while rare or complex words are gracefully decomposed into recognizable subword building blocks (e.g. "Unbelievable" -> "Un", "bel", "ievable"). Notice how whitespace is explicitly prepended to words (`·world`).'
      },
      'Stage 3': {
        summary: 'Discrete subwords are converted into numerical integer indices in the tokenizer vocabulary.',
        formula: 't_i \\in \\{0, 1, ..., V-1\\}, \\quad V = 50,257 \\text{ (GPT-2)}',
        details: 'A token ID is merely an address or index in the lookup table. Crucially, token IDs do NOT carry geometric meaning: ID #6043 ("king") and ID #16733 ("queen") are just numbers and do not reflect that kings and queens are royal counterparts.'
      },
      'Stage 4': {
        summary: 'Each discrete token ID is mapped into a high-dimensional continuous semantic vector.',
        formula: '\\mathbf{x}_i = \\mathbf{E}[t_i] \\in \\mathbb{R}^{384} \\quad (\\text{or } \\mathbb{R}^{768}, \\mathbb{R}^{4096})',
        details: 'The embedding matrix E translates symbolic IDs into points in a latent hypersphere where geometry encodes meaning. Words with related concepts point in similar directions, allowing cosine similarity and vector arithmetic (e.g. "king" - "man" + "woman" ≈ "queen").'
      },
      'Stage 5': {
        summary: 'Multi-Head Self-Attention updates each token vector based on surrounding contextual words.',
        formula: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V',
        details: 'Static embeddings give every word a single base meaning. Transformer layers allow words to communicate: for instance, the word "bank" in "river bank" attends to "river" and updates its vector away from financial concepts toward geographic concepts.'
      },
      'Stage 6': {
        summary: 'The final contextualized vectors are projected through an LM head to generate token probabilities.',
        formula: 'P(t_{next} = w) = \\frac{\\exp(z_w / T)}{\\sum_{j=1}^V \\exp(z_j / T)}',
        details: 'The network outputs logits across all 50,257 vocabulary candidates. The softmax function with temperature T converts these logits into a valid probability distribution from which the next token is sampled autoregressively.'
      }
    };

    const info = descriptions[stageNum] || {
      summary: 'Large Language Model Processing Pipeline Stage.',
      formula: '\\mathbf{y} = f(\\mathbf{x})',
      details: 'Converts human language into mathematical representations and predicts output tokens.'
    };

    this.pipelineModalBody.innerHTML = `
      <div class="deep-dive-card">
        <h4>Overview</h4>
        <p class="modal-desc">${info.summary}</p>
      </div>

      <div class="deep-dive-card">
        <h4>Mathematical Representation</h4>
        <div class="formula-box">${info.formula}</div>
      </div>

      <div class="deep-dive-card">
        <h4>How LLMs Use This Stage</h4>
        <p class="modal-desc">${info.details}</p>
      </div>
    `;

    this.pipelineModal.style.display = 'flex';
  }

  _populateComparisonSelectors() {
    this.compareSelectA.innerHTML = '';
    this.compareSelectB.innerHTML = '';

    if (this.uniqueWords.length === 0) {
      this.compareSelectA.innerHTML = '<option value="">(No words)</option>';
      this.compareSelectB.innerHTML = '<option value="">(No words)</option>';
      return;
    }

    this.uniqueWords.forEach((item, i) => {
      const optA = document.createElement('option');
      optA.value = item.word;
      optA.textContent = item.word;
      this.compareSelectA.appendChild(optA);

      const optB = document.createElement('option');
      optB.value = item.word;
      optB.textContent = item.word;
      this.compareSelectB.appendChild(optB);
    });

    // Default: first two distinct words if available
    this.compareWordA = this.uniqueWords[0].word;
    this.compareSelectA.value = this.compareWordA;

    if (this.uniqueWords.length > 1) {
      this.compareWordB = this.uniqueWords[1].word;
      this.compareSelectB.value = this.compareWordB;
    } else {
      this.compareWordB = this.uniqueWords[0].word;
      this.compareSelectB.value = this.compareWordB;
    }

    this._updateComparisonTool();
  }

  _updateComparisonTool() {
    if (!this.compareWordA || !this.compareWordB) {
      this._resetComparisonTool();
      return;
    }

    const vecA = this.cachedEmbeddings.get(this.compareWordA.toLowerCase());
    const vecB = this.cachedEmbeddings.get(this.compareWordB.toLowerCase());

    if (!vecA || !vecB) return;

    // 1. High-Dimensional Cosine Similarity (384D)
    const cosineSim = EmbeddingEngine.cosineSimilarity(vecA, vecB);
    const cosPct = Math.max(0, Math.min(100, Math.round(((cosineSim + 1) / 2) * 100)));

    this.cosineScoreVal.textContent = cosineSim >= 0 ? `+${cosineSim.toFixed(3)}` : cosineSim.toFixed(3);
    this.cosineScoreBar.style.width = `${cosPct}%`;

    // 2. Normalized Levenshtein Spelling Similarity
    const spellSim = EmbeddingEngine.spellingSimilarity(this.compareWordA, this.compareWordB);
    const spellPct = Math.round(spellSim * 100);

    this.spellingScoreVal.textContent = `${spellSim.toFixed(3)} (${spellPct}%)`;
    this.spellingScoreBar.style.width = `${spellPct}%`;

    // 3. Educational Contrast Insight
    let insightHtml = '';
    if (cosineSim > 0.65 && spellSim < 0.35) {
      insightHtml = `<span class="highlight-semantic">High Semantic Similarity (${cosineSim.toFixed(2)}) despite Low Spelling Similarity (${spellSim.toFixed(2)})</span>: "${this.compareWordA}" and "${this.compareWordB}" share conceptual meaning even though their letters differ completely.`;
    } else if (spellSim > 0.65 && cosineSim < 0.4) {
      insightHtml = `<span class="highlight-spelling">High Spelling Similarity (${spellSim.toFixed(2)}) with Low Semantic Similarity (${cosineSim.toFixed(2)})</span>: "${this.compareWordA}" and "${this.compareWordB}" share rhyming/orthographic patterns, but LLMs differentiate their conceptual meanings in embedding space.`;
    } else if (cosineSim > 0.65 && spellSim > 0.65) {
      insightHtml = `Both Semantic and Spelling similarity are high between "${this.compareWordA}" and "${this.compareWordB}".`;
    } else {
      insightHtml = `Cosine similarity compares the angle between their 384-dimensional semantic vectors; spelling similarity measures string edit distance.`;
    }
    this.comparisonInsight.innerHTML = insightHtml;

    // 4. Update 3D Connecting Laser Line
    this.visualizer.setComparison(this.compareWordA, this.compareWordB, cosineSim, spellSim);
  }

  _resetComparisonTool() {
    this.cosineScoreVal.textContent = '0.000';
    this.cosineScoreBar.style.width = '0%';
    this.spellingScoreVal.textContent = '0.000';
    this.spellingScoreBar.style.width = '0%';
    this.comparisonInsight.textContent = 'Select two words to compare their semantic and spelling relationships.';
    this.visualizer.setComparison(null, null);
  }

  selectWord(word) {
    if (!word) return;
    this.selectedWord = word;
    this.visualizer.selectWord(word);

    // Update Info Panel
    this._renderWordInfo(word);

    // If word A is already set, set word B for quick comparison clicking
    if (this.compareWordA && this.compareWordA.toLowerCase() !== word.toLowerCase()) {
      this.compareWordB = word;
      if (this.compareSelectB) this.compareSelectB.value = word;
      this._updateComparisonTool();
    }
  }

  _renderWordInfo(word) {
    const item = this.uniqueWords.find(u => u.word.toLowerCase() === word.toLowerCase());
    if (!item || !item.vector) return;

    this.wordInfoEmpty.style.display = 'none';
    this.wordInfoContent.style.display = 'block';

    this.infoWordTitle.textContent = item.word;
    this.infoWordTitle.style.color = item.color;
    this.infoTokenId.textContent = `#${item.token.id}`;

    // Vector Metrics
    const vec = item.vector;
    this.infoVectorDim.textContent = `${vec.length} Dimensions`;

    let norm = 0, min = Infinity, max = -Infinity;
    for (const v of vec) {
      norm += v * v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    norm = Math.sqrt(norm);
    this.infoVectorNorm.textContent = norm.toFixed(3);
    this.infoVectorMinMax.textContent = `[${min.toFixed(2)}, ${max.toFixed(2)}]`;

    // Numerical Vector Values Preview (First 16 dimensions)
    const slice = vec.slice(0, 16);
    this.infoVectorPreview.innerHTML = `[
  ${slice.map((v, i) => `<span class="vec-val ${v >= 0 ? 'pos' : 'neg'}">${v >= 0 ? '+' : ''}${v.toFixed(4)}</span>`).join(', ')},
  <span class="vec-ellipsis">... +${vec.length - 16} more dims</span>
]`;

    // Vector Mini Heatmap / Barcode
    this.infoVectorHeatmap.innerHTML = '';
    const sampleDims = 48;
    for (let i = 0; i < sampleDims; i++) {
      const val = vec[i] || 0;
      const bar = document.createElement('span');
      bar.className = 'heatmap-bar';
      // Normalize -0.15 to +0.15 range to color intensity
      const intensity = Math.min(1.0, Math.abs(val) * 7);
      if (val >= 0) {
        bar.style.backgroundColor = `rgba(6, 182, 212, ${0.2 + intensity * 0.8})`;
      } else {
        bar.style.backgroundColor = `rgba(244, 63, 94, ${0.2 + intensity * 0.8})`;
      }
      bar.title = `Dim ${i}: ${val.toFixed(4)}`;
      this.infoVectorHeatmap.appendChild(bar);
    }

    // Nearest Neighbors in current vocabulary
    const neighbors = this.embeddingEngine.findNearestNeighbors(
      item.word,
      vec,
      this.uniqueWords,
      5
    );

    this.infoNeighborsList.innerHTML = '';
    if (neighbors.length === 0) {
      this.infoNeighborsList.innerHTML = '<li class="no-neighbors">No other words in active text.</li>';
    } else {
      neighbors.forEach(n => {
        const li = document.createElement('li');
        li.className = 'neighbor-item';
        const pct = Math.max(0, Math.min(100, Math.round(((n.similarity + 1) / 2) * 100)));

        li.innerHTML = `
          <div class="neighbor-header">
            <span class="neighbor-word" style="color: ${n.color}">${n.word}</span>
            <span class="neighbor-sim">${n.similarity >= 0 ? '+' : ''}${n.similarity.toFixed(3)}</span>
          </div>
          <div class="neighbor-bar-track">
            <div class="neighbor-bar-fill" style="width: ${pct}%; background-color: ${n.color}"></div>
          </div>
        `;

        li.addEventListener('click', () => {
          this.compareWordA = item.word;
          this.compareWordB = n.word;
          if (this.compareSelectA) this.compareSelectA.value = item.word;
          if (this.compareSelectB) this.compareSelectB.value = n.word;
          this._updateComparisonTool();
        });

        this.infoNeighborsList.appendChild(li);
      });
    }
  }

  _resetWordInfo() {
    this.wordInfoEmpty.style.display = 'block';
    this.wordInfoContent.style.display = 'none';
  }

  _handle3DHover(data, clientX, clientY) {
    if (data) {
      this.tooltip3D.style.display = 'block';
      this.tooltip3D.style.left = `${clientX + 16}px`;
      this.tooltip3D.style.top = `${clientY + 16}px`;
      this.tooltip3D.innerHTML = `
        <div class="tooltip-word" style="color: ${data.color}"><strong>${data.word}</strong></div>
        <div class="tooltip-id">Token ID: #${data.token.id}</div>
        <div class="tooltip-coords">3D: (${data.coords[0]}, ${data.coords[1]}, ${data.coords[2]})</div>
      `;
      this._highlightCrossSections(data.word, data.token.index, true);
    } else {
      this.tooltip3D.style.display = 'none';
      this._highlightCrossSections(null, null, false);
    }
  }

  _handle3DClick(data) {
    if (data) {
      this.selectWord(data.word);
      this._scrollToVocabRow(data.token.index);
      this._highlightTokenChip(data.token.index);
    }
  }

  _highlightCrossSections(word, tokenIndex, isHighlighted) {
    // 1. Highlight token span in Section 1 text stream
    if (this.interactiveTextStream) {
      const spans = this.interactiveTextStream.querySelectorAll('.char-token-span');
      spans.forEach(span => {
        if (tokenIndex && span.dataset.index == tokenIndex) {
          span.classList.toggle('active-highlight', isHighlighted);
        } else {
          span.classList.remove('active-highlight');
        }
      });
    }

    // 2. Highlight token chip in Section 2
    const chips = this.tokensContainer.querySelectorAll('.token-chip');
    chips.forEach(chip => {
      if (tokenIndex && chip.dataset.index == tokenIndex) {
        chip.classList.toggle('active-highlight', isHighlighted);
      } else {
        chip.classList.remove('active-highlight');
      }
    });

    // 3. Highlight vocab table row in Section 3
    const rows = this.vocabTableBody.querySelectorAll('.vocab-row');
    rows.forEach(row => {
      if (tokenIndex && row.dataset.index == tokenIndex) {
        row.classList.toggle('active-highlight', isHighlighted);
      } else {
        row.classList.remove('active-highlight');
      }
    });
  }

  _highlightTokenChip(tokenIndex) {
    const chip = this.tokensContainer.querySelector(`.token-chip[data-index="${tokenIndex}"]`);
    if (chip) {
      chip.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
      chip.classList.add('selected');
      setTimeout(() => chip.classList.remove('selected'), 1200);
    }
  }

  _scrollToVocabRow(tokenIndex) {
    const row = this.vocabTableBody.querySelector(`.vocab-row[data-index="${tokenIndex}"]`);
    if (row) {
      row.scrollIntoView({ behavior: 'smooth', block: 'center' });
      row.classList.add('selected');
      setTimeout(() => row.classList.remove('selected'), 1200);
    }
  }

  _escapeHTML(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
}

// Instantiate once DOM is ready
window.addEventListener('DOMContentLoaded', () => {
  window.app = new App();
});
