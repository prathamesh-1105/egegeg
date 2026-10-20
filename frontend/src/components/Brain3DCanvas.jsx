import React, { useEffect, useRef } from 'react';

export default function Brain3DCanvas({ predictedClass = 'REST', height = 340, isProcessing = false }) {
  const mountRef = useRef(null);

  useEffect(() => {
    if (!window.THREE || !mountRef.current) return;

    const THREE = window.THREE;
    const container = mountRef.current;
    const width = container.clientWidth;
    const h = height;

    // Scene & Camera
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / h, 0.1, 1000);
    camera.position.set(0, 0.4, 4.2);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const cyanLight = new THREE.PointLight(0x00F2FE, 2.5, 10);
    cyanLight.position.set(2, 2, 2);
    scene.add(cyanLight);

    const blueLight = new THREE.PointLight(0x3B82F6, 2.5, 10);
    blueLight.position.set(-2, -1, 2);
    scene.add(blueLight);

    // Brain group
    const brainGroup = new THREE.Group();
    scene.add(brainGroup);

    // Generate Neural Brain Silhouette Point Cloud
    const nodeCount = 380;
    const nodes = [];
    const nodePositions = [];
    const colors = [];

    const colorDefault = new THREE.Color(0x38BDF8);
    const colorLeftActive = new THREE.Color(0x00F2FE); // Left hemisphere / C3
    const colorRightActive = new THREE.Color(0xF59E0B); // Right hemisphere / C4

    for (let i = 0; i < nodeCount; i++) {
      const u = Math.random() * Math.PI * 2;
      const v = (Math.random() - 0.5) * Math.PI;

      let x = 1.35 * Math.cos(v) * Math.cos(u);
      let y = 1.0 * Math.sin(v);
      let z = 1.55 * Math.cos(v) * Math.sin(u);

      // Longitudinal fissure gap
      if (Math.abs(x) < 0.15) {
        x += (x >= 0 ? 0.12 : -0.12);
      }

      // Cortical fold noise
      const fold = Math.sin(x * 6) * Math.cos(y * 6) * Math.sin(z * 6) * 0.12;
      x += fold;
      y += fold;
      z += fold;

      nodes.push({ x, y, z, originalX: x, originalY: y, originalZ: z });
      nodePositions.push(x, y, z);
      colors.push(colorDefault.r, colorDefault.g, colorDefault.b);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(nodePositions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const pointMaterial = new THREE.PointsMaterial({
      size: 0.08,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending
    });

    const pointCloud = new THREE.Points(geometry, pointMaterial);
    brainGroup.add(pointCloud);

    // Neural Synapse Network Lines
    const lineIndices = [];
    for (let i = 0; i < nodeCount; i++) {
      for (let j = i + 1; j < nodeCount; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dz = nodes[i].z - nodes[j].z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < 0.38) {
          lineIndices.push(i, j);
        }
      }
    }

    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute('position', geometry.getAttribute('position'));
    lineGeometry.setIndex(lineIndices);

    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0x0284C7,
      transparent: true,
      opacity: 0.28,
      blending: THREE.AdditiveBlending
    });

    const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    brainGroup.add(lines);

    // Outer Glow Shell
    const shellGeo = new THREE.SphereGeometry(1.65, 24, 24);
    const shellMat = new THREE.MeshBasicMaterial({
      color: 0x00F2FE,
      wireframe: true,
      transparent: true,
      opacity: 0.04
    });
    const shell = new THREE.Mesh(shellGeo, shellMat);
    brainGroup.add(shell);

    // Mouse drag interaction
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;

    const onMouseDown = (e) => {
      isDragging = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    };

    const onMouseMove = (e) => {
      if (!isDragging) return;
      const deltaX = e.clientX - prevMouseX;
      const deltaY = e.clientY - prevMouseY;
      brainGroup.rotation.y += deltaX * 0.008;
      brainGroup.rotation.x += deltaY * 0.008;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    };

    const onMouseUp = () => { isDragging = false; };

    container.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);

    // Animation Loop
    let animId;
    let clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const time = clock.getElapsedTime();

      if (!isDragging) {
        brainGroup.rotation.y += 0.006;
      }

      // Pulse nodes
      const posAttr = geometry.getAttribute('position');
      const colAttr = geometry.getAttribute('color');
      const pNorm = (predictedClass || '').toUpperCase();

      for (let i = 0; i < nodeCount; i++) {
        const n = nodes[i];
        const pulse = Math.sin(time * 3 + i * 0.2) * 0.03;
        posAttr.setXYZ(i, n.originalX * (1 + pulse), n.originalY * (1 + pulse), n.originalZ * (1 + pulse));

        // Regional cortical highlighting
        if (pNorm.includes('RIGHT') && n.originalX < -0.2 && n.originalY > 0.2) {
          // Left Motor Cortex C3 lights up for Right fist
          colAttr.setXYZ(i, colorLeftActive.r, colorLeftActive.g, colorLeftActive.b);
        } else if (pNorm.includes('LEFT') && n.originalX > 0.2 && n.originalY > 0.2) {
          // Right Motor Cortex C4 lights up for Left fist
          colAttr.setXYZ(i, colorRightActive.r, colorRightActive.g, colorRightActive.b);
        } else {
          colAttr.setXYZ(i, colorDefault.r, colorDefault.g, colorDefault.b);
        }
      }

      posAttr.needsUpdate = true;
      colAttr.needsUpdate = true;

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!mountRef.current) return;
      const newW = mountRef.current.clientWidth;
      camera.aspect = newW / h;
      camera.updateProjectionMatrix();
      renderer.setSize(newW, h);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      container.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      renderer.dispose();
      container.innerHTML = '';
    };
  }, [height]);

  const pNorm = (predictedClass || '').toUpperCase();
  const activeLabel = pNorm.includes('LEFT')
    ? 'RIGHT MOTOR CORTEX (C4 ACTIVATION)'
    : pNorm.includes('RIGHT')
    ? 'LEFT MOTOR CORTEX (C3 ACTIVATION)'
    : 'HOMOGENEOUS SMR REST';

  const labelColor = pNorm.includes('LEFT') ? '#38BDF8' : pNorm.includes('RIGHT') ? '#00F2FE' : '#94A3B8';

  return (
    <div style={{ position: 'relative', width: '100%', borderRadius: '14px', overflow: 'hidden', background: 'radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.9) 0%, rgba(6, 9, 19, 0.98) 100%)', border: '1px solid rgba(56, 189, 248, 0.2)' }}>
      {/* 3D Canvas Mount */}
      <div ref={mountRef} style={{ width: '100%', height: `${height}px`, cursor: 'grab' }} />

      {/* Floating HUD Badges */}
      <div style={{
        position: 'absolute',
        top: '12px',
        left: '14px',
        right: '14px',
        display: 'flex',
        justifyContent: 'space-between',
        pointerEvents: 'none'
      }}>
        <div className="neural-badge badge-cyan" style={{ backdropFilter: 'blur(8px)' }}>
          <span className="pulse-indicator-cyan" /> 3D NEURAL CORTEX
        </div>

        <div style={{
          background: 'rgba(15, 23, 42, 0.85)',
          border: '1px solid rgba(56, 189, 248, 0.3)',
          backdropFilter: 'blur(8px)',
          borderRadius: '8px',
          padding: '4px 10px',
          fontSize: '0.75rem',
          fontWeight: 700,
          color: labelColor,
          letterSpacing: '0.05em'
        }}>
          {activeLabel}
        </div>
      </div>

      <div style={{
        position: 'absolute',
        bottom: '10px',
        left: '14px',
        fontSize: '0.7rem',
        color: '#64748B',
        pointerEvents: 'none'
      }}>
        Interactive WebGL • Drag to Rotate 360° • Synaptic Pulsing
      </div>
    </div>
  );
}
