from typing import Optional


def render_3d_brain_html(
    predicted_class: str = "REST",
    height: int = 380,
    is_processing: bool = False
) -> str:
    """
    Generates a self-contained WebGL 3D Brain & Neural Network HTML component using Three.js.
    Dynamically illuminates the Left or Right Motor Cortex regions depending on model predictions.

    Parameters:
        predicted_class (str): 'REST', 'LEFT_FIST', or 'RIGHT_FIST'.
        height (int): Canvas container height in pixels.
        is_processing (bool): Whether an active EEG window is being processed.

    Returns:
        str: Complete HTML string ready for st.components.v1.html().
    """
    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                overflow: hidden;
                background-color: transparent;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
            }}
            #brain-container {{
                width: 100%;
                height: {height}px;
                position: relative;
                border-radius: 16px;
                background: radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.9) 0%, rgba(6, 9, 19, 0.95) 100%);
                border: 1px solid rgba(56, 189, 248, 0.2);
                box-shadow: inset 0 0 30px rgba(0, 242, 254, 0.05), 0 8px 32px rgba(0, 0, 0, 0.5);
                overflow: hidden;
            }}
            .brain-hud {{
                position: absolute;
                bottom: 12px;
                left: 14px;
                right: 14px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                pointer-events: none;
            }}
            .brain-tag {{
                background: rgba(15, 23, 42, 0.85);
                border: 1px solid rgba(56, 189, 248, 0.3);
                backdrop-filter: blur(8px);
                border-radius: 8px;
                padding: 6px 12px;
                color: #94A3B8;
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            .cortex-indicator {{
                color: #00F2FE;
                font-weight: 700;
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <div id="brain-container">
            <div class="brain-hud">
                <div class="brain-tag">3D NEURAL CORTEX VISUALIZER</div>
                <div class="brain-tag">ACTIVE REGION: <span class="cortex-indicator" id="cortex-label">NEUTRAL REST</span></div>
            </div>
        </div>

        <script>
            const container = document.getElementById('brain-container');
            const scene = new THREE.Scene();

            const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(0, 0.5, 4.2);

            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            // Lighting Setup
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
            scene.add(ambientLight);

            const cyanLight = new THREE.PointLight(0x00F2FE, 2.0, 10);
            cyanLight.position.set(2, 2, 2);
            scene.add(cyanLight);

            const purpleLight = new THREE.PointLight(0x8B5CF6, 2.0, 10);
            purpleLight.position.set(-2, -1, 2);
            scene.add(purpleLight);

            // Brain Neural Network Group
            const brainGroup = new THREE.Group();
            scene.add(brainGroup);

            // Generate Neural Brain Silhouette Point Cloud & Node Graph
            const nodeCount = 380;
            const nodes = [];
            const nodePositions = [];
            const colors = [];

            const colorDefault = new THREE.Color(0x38BDF8);
            const colorLeftActive = new THREE.Color(0x00F2FE);
            const colorRightActive = new THREE.Color(0xF59E0B);

            // Parametric Brain Shape Generator
            for (let i = 0; i < nodeCount; i++) {{
                const u = Math.random() * Math.PI * 2;
                const v = (Math.random() - 0.5) * Math.PI;

                // Ellipsoid base adjusted for brain hemispheres
                let x = 1.35 * Math.cos(v) * Math.cos(u);
                let y = 1.0 * Math.sin(v);
                let z = 1.55 * Math.cos(v) * Math.sin(u);

                // Add longitudinal fissure gap (hemisphere separation)
                if (Math.abs(x) < 0.15) {{
                    x += (x >= 0 ? 0.1 : -0.1);
                }}

                // Add cortical surface folds noise
                const fold = Math.sin(x * 6) * Math.cos(y * 6) * Math.sin(z * 6) * 0.12;
                x += fold;
                y += fold;
                z += fold;

                nodes.push({{ x, y, z, originalX: x, originalY: y, originalZ: z }});
                nodePositions.push(x, y, z);
                colors.push(colorDefault.r, colorDefault.g, colorDefault.b);
            }}

            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(nodePositions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

            // Neural Nodes Material
            const sprite = new THREE.Texture();
            const pointMaterial = new THREE.PointsMaterial({{
                size: 0.08,
                vertexColors: true,
                transparent: true,
                opacity: 0.9,
                blending: THREE.AdditiveBlending
            }});

            const pointCloud = new THREE.Points(geometry, pointMaterial);
            brainGroup.add(pointCloud);

            // Create Synaptic Axon Connections (Lines between nearby nodes)
            const linePositions = [];
            const lineColors = [];
            const maxConnectDist = 0.55;

            for (let i = 0; i < nodeCount; i++) {{
                for (let j = i + 1; j < nodeCount; j++) {{
                    const dx = nodes[i].x - nodes[j].x;
                    const dy = nodes[i].y - nodes[j].y;
                    const dz = nodes[i].z - nodes[j].z;
                    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);

                    if (dist < maxConnectDist && Math.random() < 0.45) {{
                        linePositions.push(nodes[i].x, nodes[i].y, nodes[i].z);
                        linePositions.push(nodes[j].x, nodes[j].y, nodes[j].z);

                        lineColors.push(colorDefault.r, colorDefault.g, colorDefault.b, 0.25);
                        lineColors.push(colorDefault.r, colorDefault.g, colorDefault.b, 0.25);
                    }}
                }}
            }}

            const lineGeometry = new THREE.BufferGeometry();
            lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
            const lineMaterial = new THREE.LineBasicMaterial({{
                color: 0x1E293B,
                transparent: true,
                opacity: 0.35,
                blending: THREE.AdditiveBlending
            }});
            const lineMesh = new THREE.LineSegments(lineGeometry, lineMaterial);
            brainGroup.add(lineMesh);

            // Motor Cortex Highlight Spheres
            const leftCortexMesh = new THREE.Mesh(
                new THREE.SphereGeometry(0.42, 16, 16),
                new THREE.MeshBasicMaterial({{ color: 0x00F2FE, transparent: true, opacity: 0.0, blending: THREE.AdditiveBlending }})
            );
            leftCortexMesh.position.set(-0.65, 0.45, 0.1);
            brainGroup.add(leftCortexMesh);

            const rightCortexMesh = new THREE.Mesh(
                new THREE.SphereGeometry(0.42, 16, 16),
                new THREE.MeshBasicMaterial({{ color: 0xF59E0B, transparent: true, opacity: 0.0, blending: THREE.AdditiveBlending }})
            );
            rightCortexMesh.position.set(0.65, 0.45, 0.1);
            brainGroup.add(rightCortexMesh);

            // State & Prediction Reaction Logic
            const predClass = "{predicted_class}";
            const cortexLabelElem = document.getElementById('cortex-label');

            if (predClass === "LEFT_FIST") {{
                leftCortexMesh.material.opacity = 0.55;
                if (cortexLabelElem) cortexLabelElem.innerText = "RIGHT MOTOR CORTEX (LEFT FIST PRED)";
                if (cortexLabelElem) cortexLabelElem.style.color = "#00F2FE";
            }} else if (predClass === "RIGHT_FIST") {{
                rightCortexMesh.material.opacity = 0.55;
                if (cortexLabelElem) cortexLabelElem.innerText = "LEFT MOTOR CORTEX (RIGHT FIST PRED)";
                if (cortexLabelElem) cortexLabelElem.style.color = "#F59E0B";
            }} else {{
                leftCortexMesh.material.opacity = 0.05;
                rightCortexMesh.material.opacity = 0.05;
                if (cortexLabelElem) cortexLabelElem.innerText = "NEUTRAL REST STATE";
                if (cortexLabelElem) cortexLabelElem.style.color = "#10B981";
            }}

            // Ambient Neural Particles
            const particleGeo = new THREE.BufferGeometry();
            const pCount = 120;
            const pPositions = new Float32Array(pCount * 3);
            for (let i = 0; i < pCount * 3; i += 3) {{
                pPositions[i] = (Math.random() - 0.5) * 6;
                pPositions[i+1] = (Math.random() - 0.5) * 6;
                pPositions[i+2] = (Math.random() - 0.5) * 6;
            }}
            particleGeo.setAttribute('position', new THREE.BufferAttribute(pPositions, 3));
            const particleMat = new THREE.PointsMaterial({{ size: 0.04, color: 0x8B5CF6, transparent: true, opacity: 0.4 }});
            const particles = new THREE.Points(particleGeo, particleMat);
            scene.add(particles);

            // Animation Loop
            let clock = new THREE.Clock();
            function animate() {{
                requestAnimationFrame(animate);
                const elapsedTime = clock.getElapsedTime();

                // Gentle Brain Rotation
                brainGroup.rotation.y = elapsedTime * 0.22;
                brainGroup.rotation.x = Math.sin(elapsedTime * 0.15) * 0.08;

                // Subtle Particle Float
                particles.rotation.y = -elapsedTime * 0.05;

                // Pulsing Cortex Glow if active
                if (predClass === "LEFT_FIST") {{
                    leftCortexMesh.scale.setScalar(1.0 + Math.sin(elapsedTime * 4.0) * 0.12);
                }} else if (predClass === "RIGHT_FIST") {{
                    rightCortexMesh.scale.setScalar(1.0 + Math.sin(elapsedTime * 4.0) * 0.12);
                }}

                renderer.render(scene, camera);
            }}
            animate();

            // Resize handling
            window.addEventListener('resize', () => {{
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }});
        </script>
    </body>
    </html>
    """
    return html_code
