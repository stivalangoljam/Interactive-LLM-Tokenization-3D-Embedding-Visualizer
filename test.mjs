import { GPT2Tokenizer } from './js/tokenizer.js';
import { EmbeddingEngine } from './js/embeddings.js';
import { PCA } from './js/pca.js';

const tok = new GPT2Tokenizer();
const emb = new EmbeddingEngine();

async function testPipeline(text) {
  console.log('====================================================');
  console.log('Testing Input:', text);
  const tokens = await tok.tokenize(text);
  console.log('Token count:', tokens.length);
  console.log('Tokens with IDs:');
  tokens.forEach(t => console.log(`  - [${t.displayText}] -> ID: #${t.id} (Color: ${t.color})`));
  
  const words = Array.from(new Set(tokens.map(t => t.rawText.trim()).filter(Boolean)));
  console.log('Extracted unique words:', words);

  const vectors = [];
  for (const w of words) {
    const v = await emb.getEmbedding(w);
    vectors.push(v);
  }
  console.log('Vector dimension size:', vectors[0].length);

  const pca = PCA.project(vectors, 3);
  console.log('PCA 3D Coordinates:');
  pca.coordinates.forEach((c, i) => console.log(`  - ${words[i]}: [${c.join(', ')}]`));
  console.log('PCA Variance explained:', (pca.varianceExplained * 100).toFixed(1) + '%');
  
  // Test Cosine Similarity vs Spelling Similarity
  if (words.length >= 2) {
    const cos = EmbeddingEngine.cosineSimilarity(vectors[0], vectors[1]);
    const spell = EmbeddingEngine.spellingSimilarity(words[0], words[1]);
    console.log(`Pair (${words[0]}, ${words[1]}): Cosine = ${cos.toFixed(3)}, Spelling = ${spell.toFixed(3)}`);
  }
}

async function run() {
  await testPipeline('Hello world');
  await testPipeline('Artificial Intelligence is amazing');
  await testPipeline('The king and queen are ruling the kingdom');
  await testPipeline('Unbelievable things happen unexpectedly');
  await testPipeline('The car is faster than the bicycle');
  await testPipeline('cat bat hat dog car');
}

run();
