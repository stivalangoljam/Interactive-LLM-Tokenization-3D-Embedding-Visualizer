/**
 * Three.js 3D Coordinate-Space Embedding Visualizer
 * Supports OrbitControls, dynamic axes, glowing sphere nodes, billboarded labels,
 * raycasting hover/click, animated 3D comparison laser line, and smooth mode transitions.
 */

export class Visualizer3D {
  constructor(containerElement, options = {}) {
    this.container = containerElement;
    this.options = options;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;

    // Node storage
    this.nodes = new Map(); // word -> { mesh, halo, spriteLabel, currentPos, targetPos, data }
    this.activeWords = [];

    // Interaction state
    this.hoveredNode = null;
    this.selectedNodeA = null;
    this.selectedNodeB = null;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2(-999, -999);

    // Visual Elements
    this.axisGroup = new THREE.Group();
    this.gridGroup = new THREE.Group();
    this.nodesGroup = new THREE.Group();
    this.comparisonGroup = new THREE.Group();
    this.selectionRing = null;

    // Animation state
    this.isTransitioning = false;
    this.transitionStart = 0;
    this.transitionDuration = 800; // ms
    this.autoRotate = false;

    // Callbacks
    this.onNodeHover = options.onNodeHover || null;
    this.onNodeClick = options.onNodeClick || null;

    this._initThree();
    this._createEnvironment();
    this._setupEventListeners();
    this._animate();
  }

  _initThree() {
    const width = this.container.clientWidth || 800;
    const height = this.container.clientHeight || 550;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x060913, 0.015);

    // Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    this.camera.position.set(11, 8, 14);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x060913, 1.0);
    this.container.appendChild(this.renderer.domElement);

    // OrbitControls
    if (window.THREE && window.THREE.OrbitControls) {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.maxDistance = 50;
      this.controls.minDistance = 2;
      this.controls.target.set(0, 0, 0);
    }

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 1.2);
    dirLight1.position.set(10, 20, 15);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xa855f7, 0.8);
    dirLight2.position.set(-15, -10, -10);
    this.scene.add(dirLight2);

    // Groups
    this.scene.add(this.gridGroup);
    this.scene.add(this.axisGroup);
    this.scene.add(this.nodesGroup);
    this.scene.add(this.comparisonGroup);

    // Glow Texture for Halo Sprites
    this.glowTexture = this._createGlowTexture();
  }

  _createEnvironment() {
    // 1. Subtle Cyber Floor Grid
    const gridSize = 14;
    const gridDivisions = 14;
    const gridColorCenter = new THREE.Color(0x06b6d4);
    const gridColorLines = new THREE.Color(0x1e293b);

    const gridHelper = new THREE.GridHelper(gridSize, gridDivisions, gridColorCenter, gridColorLines);
    gridHelper.position.y = -5.0;
    gridHelper.material.opacity = 0.4;
    gridHelper.material.transparent = true;
    this.gridGroup.add(gridHelper);

    // 2. Glowing Coordinate Axes
    this._createGlowingAxes();

    // 3. Subtle background particle field
    const particleCount = 200;
    const particleGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 40;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 30;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 40;

      const c = new THREE.Color(Math.random() > 0.5 ? 0x06b6d4 : 0x8b5cf6);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const particleMaterial = new THREE.PointsMaterial({
      size: 0.12,
      vertexColors: true,
      transparent: true,
      opacity: 0.35,
      blending: THREE.AdditiveBlending
    });

    const particles = new THREE.Points(particleGeometry, particleMaterial);
    this.scene.add(particles);

    // 4. Animated Selection Ring
    const ringGeo = new THREE.TorusGeometry(0.55, 0.03, 16, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x22d3ee,
      transparent: true,
      opacity: 0.9,
      wireframe: false
    });
    this.selectionRing = new THREE.Mesh(ringGeo, ringMat);
    this.selectionRing.visible = false;
    this.scene.add(this.selectionRing);
  }

  _createGlowingAxes() {
    const axisLength = 6.5;

    // Helper to build single axis line and arrow
    const buildAxis = (dir, colorHex, labelText) => {
      const group = new THREE.Group();

      // Main line
      const lineGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        dir.clone().multiplyScalar(axisLength)
      ]);
      const lineMat = new THREE.LineBasicMaterial({
        color: colorHex,
        linewidth: 2,
        transparent: true,
        opacity: 0.85
      });
      const line = new THREE.Line(lineGeo, lineMat);
      group.add(line);

      // Arrow cone
      const coneGeo = new THREE.ConeGeometry(0.15, 0.4, 16);
      const coneMat = new THREE.MeshBasicMaterial({ color: colorHex });
      const cone = new THREE.Mesh(coneGeo, coneMat);
      cone.position.copy(dir.clone().multiplyScalar(axisLength));
      cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      group.add(cone);

      // Axis label sprite
      const sprite = this._createTextSprite(labelText, {
        color: colorHex,
        fontSize: 22,
        isAxisLabel: true
      });
      sprite.position.copy(dir.clone().multiplyScalar(axisLength + 0.6));
      group.add(sprite);

      return group;
    };

    // X Axis: Neon Rose
    this.axisGroup.add(buildAxis(new THREE.Vector3(1, 0, 0), 0xf43f5e, '+X (Dim 1 / PC 1)'));
    // Y Axis: Neon Green
    this.axisGroup.add(buildAxis(new THREE.Vector3(0, 1, 0), 0x10b981, '+Y (Dim 2 / PC 2)'));
    // Z Axis: Neon Cyan
    this.axisGroup.add(buildAxis(new THREE.Vector3(0, 0, 1), 0x06b6d4, '+Z (Dim 3 / PC 3)'));
  }

  /**
   * Updates or builds the 3D points
   * @param {Array} wordItems - [{ word, token, color, coords: [x,y,z], vector }]
   * @param {boolean} animateTransition - smooth glide to new positions
   */
  updatePoints(wordItems, animateTransition = true) {
    this.activeWords = wordItems;
    const currentWordSet = new Set(wordItems.map(item => item.word.toLowerCase()));

    // Remove old nodes no longer in active input
    for (const [key, node] of this.nodes.entries()) {
      if (!currentWordSet.has(key)) {
        this.nodesGroup.remove(node.mesh);
        this.nodesGroup.remove(node.halo);
        this.nodesGroup.remove(node.spriteLabel);
        this.nodes.delete(key);
      }
    }

    // Add or update nodes
    const now = performance.now();
    for (const item of wordItems) {
      const key = item.word.toLowerCase();
      const targetPos = new THREE.Vector3(...item.coords);

      if (this.nodes.has(key)) {
        const node = this.nodes.get(key);
        node.data = item;
        node.targetPos.copy(targetPos);
        if (!animateTransition) {
          node.currentPos.copy(targetPos);
          this._applyNodePosition(node, targetPos);
        }
      } else {
        // Create new node
        const node = this._createNodeMesh(item, targetPos);
        this.nodes.set(key, node);
      }
    }

    if (animateTransition) {
      this.isTransitioning = true;
      this.transitionStart = now;
    }

    // Update Comparison Laser if both selected words still exist
    this.updateComparisonLine();
  }

  _createNodeMesh(item, position) {
    const color = new THREE.Color(item.color || 0x22d3ee);

    // 1. Center Sphere
    const sphereGeo = new THREE.SphereGeometry(0.24, 32, 32);
    const sphereMat = new THREE.MeshStandardMaterial({
      color: color,
      emissive: color,
      emissiveIntensity: 0.5,
      roughness: 0.25,
      metalness: 0.7
    });
    const mesh = new THREE.Mesh(sphereGeo, sphereMat);
    mesh.position.copy(position);
    mesh.userData = { word: item.word, key: item.word.toLowerCase() };
    this.nodesGroup.add(mesh);

    // 2. Glowing Halo Sprite
    const haloMat = new THREE.SpriteMaterial({
      map: this.glowTexture,
      color: color,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const halo = new THREE.Sprite(haloMat);
    halo.scale.set(0.9, 0.9, 1);
    halo.position.copy(position);
    this.nodesGroup.add(halo);

    // 3. Crisp Billboarded Text Label
    const spriteLabel = this._createTextSprite(item.word, {
      color: item.color,
      fontSize: 20
    });
    spriteLabel.position.set(position.x, position.y + 0.48, position.z);
    this.nodesGroup.add(spriteLabel);

    const node = {
      mesh,
      halo,
      spriteLabel,
      currentPos: position.clone(),
      targetPos: position.clone(),
      data: item
    };

    return node;
  }

  _applyNodePosition(node, pos) {
    node.mesh.position.copy(pos);
    node.halo.position.copy(pos);
    node.spriteLabel.position.set(pos.x, pos.y + 0.48, pos.z);
  }

  /**
   * Draws a glowing 3D laser line between Word A and Word B
   */
  setComparison(wordA, wordB, cosineScore, spellingScore) {
    this.selectedNodeA = wordA ? this.nodes.get(wordA.toLowerCase()) : null;
    this.selectedNodeB = wordB ? this.nodes.get(wordB.toLowerCase()) : null;
    this.updateComparisonLine(cosineScore, spellingScore);
  }

  updateComparisonLine(cosineScore = null, spellingScore = null) {
    // Clear existing comparison lines and midpoint labels
    while (this.comparisonGroup.children.length > 0) {
      const child = this.comparisonGroup.children[0];
      this.comparisonGroup.remove(child);
    }

    if (!this.selectedNodeA || !this.selectedNodeB) return;

    const posA = this.selectedNodeA.mesh.position;
    const posB = this.selectedNodeB.mesh.position;

    // Outer Glow Laser Line
    const lineGeo = new THREE.BufferGeometry().setFromPoints([posA, posB]);
    const glowMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      linewidth: 3,
      transparent: true,
      opacity: 0.95
    });
    const line = new THREE.Line(lineGeo, glowMat);
    this.comparisonGroup.add(line);

    // Core Bright Line
    const coreMat = new THREE.LineBasicMaterial({
      color: 0xffffff,
      linewidth: 1,
      transparent: true,
      opacity: 0.9
    });
    const coreLine = new THREE.Line(lineGeo, coreMat);
    this.comparisonGroup.add(coreLine);

    // Midpoint floating similarity badge
    if (cosineScore !== null) {
      const mid = posA.clone().lerp(posB, 0.5);
      const badgeText = `Cos: ${cosineScore.toFixed(3)}`;
      const spriteBadge = this._createTextSprite(badgeText, {
        color: '#38bdf8',
        fontSize: 18,
        bgColor: 'rgba(15, 23, 42, 0.85)',
        borderColor: '#38bdf8'
      });
      spriteBadge.position.set(mid.x, mid.y + 0.35, mid.z);
      this.comparisonGroup.add(spriteBadge);
    }
  }

  /**
   * Focuses on a single selected word
   */
  selectWord(word) {
    const key = word ? word.toLowerCase() : null;
    const node = key ? this.nodes.get(key) : null;

    if (node) {
      this.selectionRing.visible = true;
      this.selectionRing.position.copy(node.mesh.position);
      this.selectionRing.lookAt(this.camera.position);
    } else {
      this.selectionRing.visible = false;
    }
  }

  resetCamera() {
    if (this.controls) {
      this.controls.reset();
      this.camera.position.set(11, 8, 14);
      this.controls.target.set(0, 0, 0);
      this.controls.update();
    }
  }

  toggleAutoRotate(enabled) {
    this.autoRotate = enabled !== undefined ? enabled : !this.autoRotate;
    if (this.controls) {
      this.controls.autoRotate = this.autoRotate;
      this.controls.autoRotateSpeed = 1.0;
    }
    return this.autoRotate;
  }

  toggleGrid(visible) {
    this.gridGroup.visible = visible !== undefined ? visible : !this.gridGroup.visible;
  }

  toggleLabels(visible) {
    const isVis = visible !== undefined ? visible : !this.axisGroup.visible;
    for (const node of this.nodes.values()) {
      node.spriteLabel.visible = isVis;
    }
  }

  _setupEventListeners() {
    window.addEventListener('resize', () => this._onResize());

    this.renderer.domElement.addEventListener('mousemove', (e) => this._onMouseMove(e));
    this.renderer.domElement.addEventListener('click', (e) => this._onClick(e));
  }

  _onResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) return;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  _onMouseMove(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const meshes = Array.from(this.nodes.values()).map(n => n.mesh);
    const intersects = this.raycaster.intersectObjects(meshes);

    if (intersects.length > 0) {
      const hitMesh = intersects[0].object;
      const key = hitMesh.userData.key;
      const node = this.nodes.get(key);

      if (this.hoveredNode !== node) {
        if (this.hoveredNode) this._setHoverState(this.hoveredNode, false);
        this.hoveredNode = node;
        this._setHoverState(this.hoveredNode, true);
        this.renderer.domElement.style.cursor = 'pointer';

        if (this.onNodeHover) {
          this.onNodeHover(node.data, event.clientX, event.clientY);
        }
      }
    } else {
      if (this.hoveredNode) {
        this._setHoverState(this.hoveredNode, false);
        this.hoveredNode = null;
        this.renderer.domElement.style.cursor = 'default';
        if (this.onNodeHover) {
          this.onNodeHover(null);
        }
      }
    }
  }

  _onClick(event) {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const meshes = Array.from(this.nodes.values()).map(n => n.mesh);
    const intersects = this.raycaster.intersectObjects(meshes);

    if (intersects.length > 0) {
      const hitMesh = intersects[0].object;
      const key = hitMesh.userData.key;
      const node = this.nodes.get(key);
      this.selectWord(node.data.word);

      if (this.onNodeClick) {
        this.onNodeClick(node.data);
      }
    }
  }

  _setHoverState(node, isHovered) {
    if (isHovered) {
      node.mesh.scale.set(1.4, 1.4, 1.4);
      node.mesh.material.emissiveIntensity = 1.0;
      node.halo.scale.set(1.4, 1.4, 1);
      node.halo.material.opacity = 1.0;
    } else {
      node.mesh.scale.set(1.0, 1.0, 1.0);
      node.mesh.material.emissiveIntensity = 0.5;
      node.halo.scale.set(0.9, 0.9, 1);
      node.halo.material.opacity = 0.75;
    }
  }

  _animate() {
    requestAnimationFrame(() => this._animate());

    // Handle position interpolation between modes (Spelling vs Semantic)
    if (this.isTransitioning) {
      const elapsed = performance.now() - this.transitionStart;
      const progress = Math.min(1.0, elapsed / this.transitionDuration);
      // Ease in-out cubic
      const ease = progress < 0.5
        ? 4 * progress * progress * progress
        : 1 - Math.pow(-2 * progress + 2, 3) / 2;

      for (const node of this.nodes.values()) {
        node.currentPos.lerpVectors(node.mesh.position, node.targetPos, ease * 0.25);
        this._applyNodePosition(node, node.currentPos);
      }

      this.updateComparisonLine();
      if (this.selectionRing.visible && this.hoveredNode) {
        this.selectionRing.position.copy(this.hoveredNode.mesh.position);
      }

      if (progress >= 1.0) {
        this.isTransitioning = false;
        for (const node of this.nodes.values()) {
          node.currentPos.copy(node.targetPos);
          this._applyNodePosition(node, node.targetPos);
        }
        this.updateComparisonLine();
      }
    }

    // Selection ring idle pulse / spin
    if (this.selectionRing && this.selectionRing.visible) {
      this.selectionRing.rotation.z += 0.02;
      this.selectionRing.lookAt(this.camera.position);
    }

    if (this.controls) {
      this.controls.update();
    }

    this.renderer.render(this.scene, this.camera);
  }

  _createGlowTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');

    const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
    grad.addColorStop(0.2, 'rgba(255, 255, 255, 0.8)');
    grad.addColorStop(0.5, 'rgba(255, 255, 255, 0.25)');
    grad.addColorStop(1, 'rgba(255, 255, 255, 0)');

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 128, 128);

    const texture = new THREE.CanvasTexture(canvas);
    return texture;
  }

  _createTextSprite(text, opts = {}) {
    const fontSize = opts.fontSize || 20;
    const fontFace = opts.fontFace || 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    const textColor = opts.color || '#ffffff';
    const bgColor = opts.bgColor || 'rgba(10, 15, 29, 0.82)';
    const borderColor = opts.borderColor || textColor;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.font = `600 ${fontSize}px ${fontFace}`;

    const textWidth = ctx.measureText(text).width;
    const paddingX = 14;
    const paddingY = 8;
    const canvasWidth = textWidth + paddingX * 2;
    const canvasHeight = fontSize + paddingY * 2;

    canvas.width = Math.ceil(canvasWidth);
    canvas.height = Math.ceil(canvasHeight);

    // Redo font setup after canvas resize
    ctx.font = `600 ${fontSize}px ${fontFace}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Rounded background pill
    const radius = 6;
    ctx.beginPath();
    ctx.moveTo(radius, 0);
    ctx.lineTo(canvas.width - radius, 0);
    ctx.quadraticCurveTo(canvas.width, 0, canvas.width, radius);
    ctx.lineTo(canvas.width, canvas.height - radius);
    ctx.quadraticCurveTo(canvas.width, canvas.height, canvas.width - radius, canvas.height);
    ctx.lineTo(radius, canvas.height);
    ctx.quadraticCurveTo(0, canvas.height, 0, canvas.height - radius);
    ctx.lineTo(0, radius);
    ctx.quadraticCurveTo(0, 0, radius, 0);
    ctx.closePath();

    ctx.fillStyle = bgColor;
    ctx.fill();

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = borderColor;
    ctx.stroke();

    // Text with soft glow
    ctx.fillStyle = textColor;
    ctx.fillText(text, canvas.width / 2, canvas.height / 2 + 1);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false
    });
    const sprite = new THREE.Sprite(spriteMat);
    const aspect = canvas.width / canvas.height;
    const baseHeight = opts.isAxisLabel ? 0.35 : 0.32;
    sprite.scale.set(baseHeight * aspect, baseHeight, 1);
    return sprite;
  }
}
