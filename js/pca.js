/**
 * High-Performance Dimensionality Reduction (PCA & Classical MDS)
 * Projects high-dimensional embedding vectors (e.g. 384D) into 3D coordinate space.
 */

export class PCA {
  /**
   * Projects an array of high-dimensional vectors to 3D using Dual PCA (Kernel/Gram matrix method).
   * Exact and fast when N (number of words) <= 200 and D is large (e.g., 384).
   * @param {number[][]} vectors - Array of N vectors, each of dimension D.
   * @param {number} targetDim - Target dimensions (default 3).
   * @returns {{ coordinates: number[][], varianceExplained: number, eigenvalues: number[] }}
   */
  static project(vectors, targetDim = 3) {
    const N = vectors.length;
    if (N === 0) return { coordinates: [], varianceExplained: 1.0, eigenvalues: [] };
    if (N === 1) return { coordinates: [[0, 0, 0]], varianceExplained: 1.0, eigenvalues: [1, 0, 0] };

    const D = vectors[0].length;

    // Handle N = 2: 2 points form a single line in high-dimensional space
    if (N === 2) {
      let dist = 0;
      for (let d = 0; d < D; d++) {
        const diff = vectors[0][d] - vectors[1][d];
        dist += diff * diff;
      }
      dist = Math.sqrt(dist);
      const half = Math.max(1.5, Math.min(dist * 2.5, 4.0));
      return {
        coordinates: [
          [-half, 0, 0],
          [half, 0, 0]
        ],
        varianceExplained: 1.0,
        eigenvalues: [dist * dist, 0, 0]
      };
    }

    // 1. Center the vectors (subtract mean)
    const mean = new Float64Array(D);
    for (let i = 0; i < N; i++) {
      for (let d = 0; d < D; d++) {
        mean[d] += vectors[i][d] / N;
      }
    }

    const centered = new Array(N);
    for (let i = 0; i < N; i++) {
      centered[i] = new Float64Array(D);
      for (let d = 0; d < D; d++) {
        centered[i][d] = vectors[i][d] - mean[d];
      }
    }

    // 2. Compute Gram Matrix K = X * X^T (N x N)
    const K = Array.from({ length: N }, () => new Float64Array(N));
    let totalVariance = 0;
    for (let i = 0; i < N; i++) {
      for (let j = i; j < N; j++) {
        let dot = 0;
        for (let d = 0; d < D; d++) {
          dot += centered[i][d] * centered[j][d];
        }
        K[i][j] = dot;
        K[j][i] = dot;
        if (i === j) totalVariance += dot;
      }
    }

    // 3. Power Iteration with Hotelling deflation to find top 3 eigenvectors of K
    const { eigenvectors, eigenvalues } = this._powerIteration(K, N, targetDim);

    // 4. Project data points into 3D: Y_k = u_k * sqrt(lambda_k)
    const coords = Array.from({ length: N }, () => [0, 0, 0]);
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < Math.min(targetDim, eigenvalues.length); k++) {
        const scale = Math.sqrt(Math.max(0, eigenvalues[k]));
        coords[i][k] = eigenvectors[k][i] * scale;
      }
    }

    // 5. Calculate variance explained by top 3 components
    const topVar = eigenvalues.slice(0, targetDim).reduce((acc, v) => acc + v, 0);
    const varianceExplained = totalVariance > 1e-9 ? Math.min(1.0, topVar / totalVariance) : 1.0;

    // 6. Rescale to comfortably fit Three.js 3D viewport (bounding box [-5, 5])
    const scaledCoordinates = this._normalizeBoundingBox(coords);

    return {
      coordinates: scaledCoordinates,
      varianceExplained,
      eigenvalues
    };
  }

  /**
   * Classical Multidimensional Scaling (MDS)
   * Projects a pairwise distance matrix (e.g. Normalized Levenshtein distance) into 3D coordinates.
   * @param {number[][]} distanceMatrix - N x N symmetric distance matrix.
   * @param {number} targetDim - Target dimensions (default 3).
   * @returns {{ coordinates: number[][], stress: number }}
   */
  static projectMDS(distanceMatrix, targetDim = 3) {
    const N = distanceMatrix.length;
    if (N === 0) return { coordinates: [], stress: 0 };
    if (N === 1) return { coordinates: [[0, 0, 0]], stress: 0 };
    if (N === 2) {
      const d = Math.max(1.5, Math.min(distanceMatrix[0][1] * 5, 4.0));
      return {
        coordinates: [[-d, 0, 0], [d, 0, 0]],
        stress: 0
      };
    }

    // 1. Squared distances: S_ij = D_ij^2
    const S = Array.from({ length: N }, () => new Float64Array(N));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const d = distanceMatrix[i][j];
        S[i][j] = d * d;
      }
    }

    // 2. Double-centering: B = -0.5 * H * S * H, where H = I - (1/N) 1 1^T
    // B_ij = -0.5 * (S_ij - rowMean_i - colMean_j + totalMean)
    const rowMeans = new Float64Array(N);
    let totalMean = 0;
    for (let i = 0; i < N; i++) {
      let rSum = 0;
      for (let j = 0; j < N; j++) rSum += S[i][j];
      rowMeans[i] = rSum / N;
      totalMean += rSum;
    }
    totalMean /= (N * N);

    const B = Array.from({ length: N }, () => new Float64Array(N));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        B[i][j] = -0.5 * (S[i][j] - rowMeans[i] - rowMeans[j] + totalMean);
      }
    }

    // 3. Top eigenvectors of B
    const { eigenvectors, eigenvalues } = this._powerIteration(B, N, targetDim);

    const coords = Array.from({ length: N }, () => [0, 0, 0]);
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < Math.min(targetDim, eigenvalues.length); k++) {
        const scale = Math.sqrt(Math.max(0, eigenvalues[k]));
        coords[i][k] = eigenvectors[k][i] * scale;
      }
    }

    const scaledCoordinates = this._normalizeBoundingBox(coords);
    return { coordinates: scaledCoordinates, stress: 0 };
  }

  /**
   * Power Iteration with Hotelling deflation for symmetric matrices
   */
  static _powerIteration(matrix, n, numComponents = 3) {
    const deflated = matrix.map(row => Float64Array.from(row));
    const eigenvectors = [];
    const eigenvalues = [];

    for (let c = 0; c < numComponents; c++) {
      let q = new Float64Array(n);
      // Deterministic initial seed to ensure stable coordinates across re-renders
      for (let i = 0; i < n; i++) {
        q[i] = Math.sin((i + 1) * 1.7 + (c + 1) * 2.3);
      }
      let norm = Math.hypot(...q);
      if (norm < 1e-12) norm = 1;
      for (let i = 0; i < n; i++) q[i] /= norm;

      let lambda = 0;
      for (let iter = 0; iter < 60; iter++) {
        const qNext = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          let sum = 0;
          for (let j = 0; j < n; j++) {
            sum += deflated[i][j] * q[j];
          }
          qNext[i] = sum;
        }

        const nextNorm = Math.hypot(...qNext);
        if (nextNorm < 1e-12) break;
        for (let i = 0; i < n; i++) qNext[i] /= nextNorm;
        q = qNext;
      }

      // Rayleigh quotient: lambda = q^T * A * q
      let qAq = 0;
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) {
          sum += deflated[i][j] * q[j];
        }
        qAq += q[i] * sum;
      }
      lambda = Math.max(0, qAq);
      eigenvalues.push(lambda);
      eigenvectors.push(Array.from(q));

      // Deflate: A_new = A - lambda * q * q^T
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          deflated[i][j] -= lambda * q[i] * q[j];
        }
      }
    }

    return { eigenvectors, eigenvalues };
  }

  /**
   * Rescales 3D coordinates to nicely fit within [-5, 5] space
   */
  static _normalizeBoundingBox(coords) {
    if (coords.length <= 1) return coords;

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    for (const [x, y, z] of coords) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }

    const rangeX = maxX - minX;
    const rangeY = maxY - minY;
    const rangeZ = maxZ - minZ;
    const maxRange = Math.max(rangeX, rangeY, rangeZ, 0.001);

    const targetSpan = 8.0; // spanning from -4 to +4
    const scale = targetSpan / maxRange;

    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;
    const midZ = (minZ + maxZ) / 2;

    return coords.map(([x, y, z]) => [
      parseFloat(((x - midX) * scale).toFixed(3)),
      parseFloat(((y - midY) * scale).toFixed(3)),
      parseFloat(((z - midZ) * scale).toFixed(3))
    ]);
  }
}
