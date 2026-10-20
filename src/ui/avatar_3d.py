from typing import Optional


def render_3d_avatar_html(
    predicted_class: str = "REST",
    actual_class: str = "REST",
    confidence: float = 0.0,
    height: int = 420,
    active_channel: Optional[str] = None
) -> str:
    """
    Generates a self-contained WebGL 3D Cybernetic Humanoid Avatar HTML component using Three.js.
    The 3D avatar dynamically interpolates arm positions based on model predictions.

    Parameters:
        predicted_class (str): 'REST', 'LEFT_FIST', or 'RIGHT_FIST'.
        actual_class (str): Ground truth label.
        confidence (float): Model prediction confidence (0.0 to 1.0).
        height (int): Canvas container height in pixels.
        active_channel (str, optional): Active channel name or group label for display.

    Returns:
        str: Complete HTML string ready for st.components.v1.html().
    """
    match_status = "✓ MATCH" if predicted_class == actual_class else "✗ MISMATCH"
    match_color = "#10B981" if predicted_class == actual_class else "#EF4444"
    conf_pct = f"{confidence * 100:.1f}%" if confidence > 0 else "N/A"
    chan_html = f'<div style="font-size: 0.7rem; color: #94A3B8; margin-top: 2px;">{active_channel}</div>' if active_channel else ''

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
            #canvas-container {{
                width: 100%;
                height: {height}px;
                position: relative;
                border-radius: 16px;
                background: radial-gradient(circle at 50% 40%, rgba(15, 23, 42, 0.95) 0%, rgba(4, 7, 17, 0.98) 100%);
                border: 1px solid rgba(56, 189, 248, 0.25);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
            }}
            .hud-overlay {{
                position: absolute;
                top: 14px;
                left: 14px;
                right: 14px;
                display: flex;
                justify-content: space-between;
                pointer-events: none;
                z-index: 10;
            }}
            .hud-card {{
                background: rgba(11, 17, 32, 0.88);
                border: 1px solid rgba(56, 189, 248, 0.3);
                backdrop-filter: blur(12px);
                border-radius: 10px;
                padding: 8px 14px;
                color: #F8FAFC;
                font-size: 0.82rem;
                box-shadow: 0 4px 16px rgba(0,0,0,0.4);
            }}
            .hud-title {{
                color: #94A3B8;
                font-size: 0.68rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }}
            .hud-val {{
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.15rem;
                font-weight: 800;
                color: #00F2FE;
                margin-top: 2px;
            }}
            .hud-match {{
                font-weight: 700;
                color: {match_color};
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <div id="canvas-container">
            <div class="hud-overlay">
                <div class="hud-card">
                    <div class="hud-title">DECODED MOVEMENT</div>
                    <div class="hud-val">{predicted_class}</div>
                    {chan_html}
                </div>
                <div class="hud-card" style="text-align: center;">
                    <div class="hud-title">CONFIDENCE</div>
                    <div class="hud-val" style="color: #38BDF8;">{conf_pct}</div>
                </div>
                <div class="hud-card" style="text-align: right;">
                    <div class="hud-title">GROUND TRUTH</div>
                    <div class="hud-val" style="color: #F59E0B;">{actual_class} <span class="hud-match">({match_status})</span></div>
                </div>
            </div>
        </div>

        <script>
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();

            const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(0, 1.15, 3.8);
            camera.lookAt(0, 0.45, 0);

            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.shadowMap.enabled = true;
            container.appendChild(renderer.domElement);

            // Futuristic Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);

            const cyanDirLight = new THREE.DirectionalLight(0x00F2FE, 1.2);
            cyanDirLight.position.set(2, 4, 3);
            scene.add(cyanDirLight);

            const purpleLight = new THREE.PointLight(0x8B5CF6, 1.5, 6);
            purpleLight.position.set(-2, 1, 2);
            scene.add(purpleLight);

            // Materials
            const bodyMat = new THREE.MeshStandardMaterial({{
                color: 0x0F172A,
                roughness: 0.25,
                metalness: 0.85
            }});
            const jointMat = new THREE.MeshStandardMaterial({{
                color: 0x00F2FE,
                emissive: 0x00F2FE,
                emissiveIntensity: 0.7
            }});
            const visorMat = new THREE.MeshStandardMaterial({{
                color: 0x38BDF8,
                emissive: 0x38BDF8,
                emissiveIntensity: 0.9
            }});
            const leftGlowMat = new THREE.MeshStandardMaterial({{
                color: 0x00F2FE,
                emissive: 0x00F2FE,
                emissiveIntensity: 0.95
            }});
            const rightGlowMat = new THREE.MeshStandardMaterial({{
                color: 0xF59E0B,
                emissive: 0xF59E0B,
                emissiveIntensity: 0.95
            }});

            // Cybernetic Avatar Assembly
            const avatarGroup = new THREE.Group();
            scene.add(avatarGroup);

            // Torso
            const torsoGeo = new THREE.CylinderGeometry(0.32, 0.23, 0.9, 16);
            const torso = new THREE.Mesh(torsoGeo, bodyMat);
            torso.position.y = 0.45;
            avatarGroup.add(torso);

            // Arc Reactor Chest Core
            const coreGeo = new THREE.SphereGeometry(0.08, 16, 16);
            const core = new THREE.Mesh(coreGeo, jointMat);
            core.position.set(0, 0.65, 0.26);
            avatarGroup.add(core);

            // Head & Visor
            const headGeo = new THREE.SphereGeometry(0.18, 24, 24);
            const head = new THREE.Mesh(headGeo, bodyMat);
            head.position.y = 1.05;
            avatarGroup.add(head);

            const visorGeo = new THREE.BoxGeometry(0.24, 0.07, 0.12);
            const visor = new THREE.Mesh(visorGeo, visorMat);
            visor.position.set(0, 1.07, 0.12);
            avatarGroup.add(visor);

            // Shoulders
            const shoulderL = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 16), jointMat);
            shoulderL.position.set(-0.4, 0.8, 0);
            avatarGroup.add(shoulderL);

            const shoulderR = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 16), jointMat);
            shoulderR.position.set(0.4, 0.8, 0);
            avatarGroup.add(shoulderR);

            // LEFT ARM ASSEMBLY (Pivot at shoulder)
            const leftArmPivot = new THREE.Group();
            leftArmPivot.position.set(-0.4, 0.8, 0);
            avatarGroup.add(leftArmPivot);

            const leftUpperArm = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.05, 0.45, 16), bodyMat);
            leftUpperArm.position.y = -0.225;
            leftArmPivot.add(leftUpperArm);

            const leftHand = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.1, 0.1), leftGlowMat);
            leftHand.position.y = -0.48;
            leftArmPivot.add(leftHand);

            // RIGHT ARM ASSEMBLY (Pivot at shoulder)
            const rightArmPivot = new THREE.Group();
            rightArmPivot.position.set(0.4, 0.8, 0);
            avatarGroup.add(rightArmPivot);

            const rightUpperArm = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.05, 0.45, 16), bodyMat);
            rightUpperArm.position.y = -0.225;
            rightArmPivot.add(rightUpperArm);

            const rightHand = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.1, 0.1), rightGlowMat);
            rightHand.position.y = -0.48;
            rightArmPivot.add(rightHand);

            // Legs
            const legL = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.06, 0.6, 16), bodyMat);
            legL.position.set(-0.16, -0.3, 0);
            avatarGroup.add(legL);

            const legR = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.06, 0.6, 16), bodyMat);
            legR.position.set(0.16, -0.3, 0);
            avatarGroup.add(legR);

            // Movement Animation Target Rotations
            const predClass = "{predicted_class}";
            let targetRotL = {{ x: 0, z: 0.1 }};
            let targetRotR = {{ x: 0, z: -0.1 }};

            if (predClass === "LEFT_FIST") {{
                targetRotL = {{ x: -1.25, z: -0.95 }};
                targetRotR = {{ x: 0, z: -0.1 }};
            }} else if (predClass === "RIGHT_FIST") {{
                targetRotL = {{ x: 0, z: 0.1 }};
                targetRotR = {{ x: -1.25, z: 0.95 }};
            }} else {{
                targetRotL = {{ x: 0, z: 0.1 }};
                targetRotR = {{ x: 0, z: -0.1 }};
            }}

            // Smooth Lerp Animation Loop
            let time = 0;
            function animate() {{
                requestAnimationFrame(animate);
                time += 0.03;

                // Subtle Breathing & Idle Body Sway
                avatarGroup.rotation.y = Math.sin(time * 0.8) * 0.12;
                avatarGroup.position.y = Math.sin(time * 1.5) * 0.02;

                // Smooth Arm Interpolation
                leftArmPivot.rotation.x += (targetRotL.x - leftArmPivot.rotation.x) * 0.12;
                leftArmPivot.rotation.z += (targetRotL.z - leftArmPivot.rotation.z) * 0.12;

                rightArmPivot.rotation.x += (targetRotR.x - rightArmPivot.rotation.x) * 0.12;
                rightArmPivot.rotation.z += (targetRotR.z - rightArmPivot.rotation.z) * 0.12;

                renderer.render(scene, camera);
            }}
            animate();

            // Window resize handler
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
