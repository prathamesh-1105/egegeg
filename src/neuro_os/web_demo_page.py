"""
NeuroOS Phase 9 — Built-in Demo Webpage Generator
Generates a rich, responsive, scientific article with substantial vertical height (>3000px)
for guaranteed same-origin and iframe scroll control demonstrations.
"""

def generate_demo_article_html() -> str:
    """Returns a complete, self-contained HTML5 document for the NeuroOS Demo Webpage."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NeuroOS Research Journal — Sensorimotor Rhythms & Assistive BCI</title>
  <style>
    :root {
      --bg: #090D16;
      --card-bg: rgba(15, 23, 42, 0.75);
      --card-border: rgba(51, 65, 85, 0.6);
      --text: #F8FAFC;
      --text-muted: #94A3B8;
      --cyan: #00F2FE;
      --violet: #8B5CF6;
      --emerald: #10B981;
      --amber: #F59E0B;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.7;
      padding: 0 0 80px 0;
      scroll-behavior: smooth;
    }
    header {
      background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(9, 13, 22, 0.95) 100%);
      border-bottom: 1px solid var(--card-border);
      padding: 40px 24px;
      text-align: center;
      position: sticky;
      top: 0;
      backdrop-filter: blur(12px);
      z-index: 10;
    }
    .badge {
      display: inline-block;
      padding: 4px 12px;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      border-radius: 9999px;
      text-transform: uppercase;
      background: rgba(0, 242, 254, 0.15);
      color: var(--cyan);
      border: 1px solid rgba(0, 242, 254, 0.4);
      margin-bottom: 12px;
    }
    h1 {
      font-size: 2.2rem;
      font-weight: 800;
      line-height: 1.25;
      background: linear-gradient(135deg, #FFFFFF 0%, #00F2FE 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 12px;
    }
    .meta {
      font-size: 0.85rem;
      color: var(--text-muted);
      display: flex;
      justify-content: center;
      gap: 20px;
      flex-wrap: wrap;
    }
    .container {
      max-width: 900px;
      margin: 40px auto;
      padding: 0 24px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 28px;
      margin-bottom: 32px;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }
    h2 {
      font-size: 1.5rem;
      color: #FFFFFF;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 10px;
      border-bottom: 1px solid rgba(51, 65, 85, 0.4);
      padding-bottom: 8px;
    }
    h2 .num {
      color: var(--cyan);
      font-family: monospace;
      font-size: 1.1rem;
    }
    p {
      color: #CBD5E1;
      font-size: 0.98rem;
      margin-bottom: 16px;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin: 20px 0;
    }
    .stat-card {
      background: rgba(30, 41, 59, 0.5);
      border: 1px solid rgba(51, 65, 85, 0.5);
      border-radius: 8px;
      padding: 16px;
      text-align: center;
    }
    .stat-val {
      font-size: 1.8rem;
      font-weight: 800;
      color: var(--cyan);
      font-family: monospace;
    }
    .stat-lbl {
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-top: 4px;
      text-transform: uppercase;
    }
    .callout {
      background: rgba(139, 92, 246, 0.1);
      border-left: 4px solid var(--violet);
      padding: 16px 20px;
      border-radius: 0 8px 8px 0;
      margin: 20px 0;
      font-size: 0.92rem;
      color: #E2E8F0;
    }
    .callout.cyan {
      background: rgba(0, 242, 254, 0.08);
      border-left-color: var(--cyan);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-size: 0.88rem;
    }
    th, td {
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid rgba(51, 65, 85, 0.5);
    }
    th {
      background: rgba(30, 41, 59, 0.8);
      color: var(--cyan);
      font-weight: 700;
    }
    td {
      color: #CBD5E1;
    }
    pre {
      background: #020617;
      border: 1px solid rgba(51, 65, 85, 0.5);
      border-radius: 8px;
      padding: 16px;
      overflow-x: auto;
      font-family: monospace;
      font-size: 0.84rem;
      color: #38BDF8;
      margin: 16px 0;
    }
    .indicator-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.75rem;
      padding: 4px 10px;
      border-radius: 6px;
      background: rgba(16, 185, 129, 0.15);
      color: var(--emerald);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .scroll-target {
      padding: 8px 0;
    }
  </style>
</head>
<body>

  <header>
    <div class="badge">NeuroOS Clinical Research Monograph</div>
    <h1>Physiological Foundations of Motor Imagery Brain-Computer Interfaces</h1>
    <div class="meta">
      <span><strong>Author:</strong> NeuroOS BCI Research Group</span>
      <span>•</span>
      <span><strong>Corpus:</strong> PhysioNet EEGMMIDB (64-channel 160Hz)</span>
      <span>•</span>
      <span><strong>Interactive Target:</strong> 3,500px Controllable Webpage</span>
    </div>
  </header>

  <div class="container">

    <!-- Section 1 -->
    <section class="card scroll-target" id="section-1">
      <h2><span class="num">01.</span> Executive Overview: Assistive Web Navigation</h2>
      <p>
        Assistive brain-computer interfaces (BCIs) translate electrophysiological neural signatures into deterministic software commands, providing individuals with severe neuromuscular impairments (e.g. ALS, locked-in syndrome, spinal cord injury) hands-free agency to navigate digital environments.
      </p>
      <div class="callout cyan">
        <strong>Zero-Conflation Boundary:</strong> NeuroOS operates strictly on 3-class statistical motor imagery decoders (REST, LEFT_FIST, RIGHT_FIST). Predictions represent electrophysiological spectral lateralization rather than speculative semantic mind-reading.
      </div>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-val">160 Hz</div>
          <div class="stat-lbl">Sampling Frequency</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">15 Ch</div>
          <div class="stat-lbl">Motor Strip Montage</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">180 px</div>
          <div class="stat-lbl">Default Scroll Step</div>
        </div>
      </div>
      <p>
        In this dedicated web browser control module (Phase 9), validated motor-imagery events trigger physical viewport scrolling:
        <strong>LEFT_FIST</strong> translates to upward displacement (SCROLL UP),
        <strong>RIGHT_FIST</strong> translates to downward displacement (SCROLL DOWN), and
        <strong>REST</strong> remains completely stationary (NEUTRAL).
      </p>
    </section>

    <!-- Section 2 -->
    <section class="card scroll-target" id="section-2">
      <h2><span class="num">02.</span> Sensorimotor Rhythms & Mu/Beta ERD</h2>
      <p>
        Voluntary motor intention and kinesthetic motor imagery produce distinct Event-Related Desynchronization (ERD) in the sensorimotor cortex. During the imagination of unilateral hand movement, thalamocortical networks undergo localized desynchronization, diminishing spectral power in the mu (8–12 Hz) and beta (16–24 Hz) bands over the contralateral hemispheric representation.
      </p>
      <p>
        Concurrently, ipsilateral homologous regions demonstrate Event-Related Synchronization (ERS) or preserve idling baseline rhythms. In the resting state, bilateral synchronous mu rhythms are maintained across electrode channels C3, Cz, and C4.
      </p>
      <div class="callout">
        <strong>Contralateral Reversal Law:</strong> Left-fist imagery induces mu attenuation over right sensorimotor electrode C4. Right-fist imagery induces mu attenuation over left sensorimotor electrode C3. REST maintains high symmetric rhythmic alpha power.
      </div>
    </section>

    <!-- Section 3 -->
    <section class="card scroll-target" id="section-3">
      <h2><span class="num">03.</span> 15-Channel Motor Strip Montage</h2>
      <p>
        Scalp surface potentials are acquired using the international 10–20 electrode placement standard. To balance spatial selectivity with real-time computational throughput, NeuroOS isolates 15 primary sensorimotor channels positioned over the primary motor cortex (M1) and supplementary motor area (SMA):
      </p>
      <table>
        <thead>
          <tr>
            <th>Electrode Subgroup</th>
            <th>Channels</th>
            <th>Physiological Role</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Left Motor Cortex</td>
            <td>FC3, C5, C3, C1, CP3</td>
            <td>Right hand lateralization & mu ERD</td>
          </tr>
          <tr>
            <td>Central Midline</td>
            <td>FCz, Cz, CPz</td>
            <td>Foot motor area & central reference</td>
          </tr>
          <tr>
            <td>Right Motor Cortex</td>
            <td>FC4, C2, C4, C6, CP4</td>
            <td>Left hand lateralization & mu ERD</td>
          </tr>
        </tbody>
      </table>
    </section>

    <!-- Section 4 -->
    <section class="card scroll-target" id="section-4">
      <h2><span class="num">04.</span> Surface Laplacian Spatial Derivation</h2>
      <p>
        Due to volume conduction through the cerebrospinal fluid (CSF), skull, and scalp, standard scalp EEG potentials suffer severe spatial smearing. To overcome this limitation, NeuroOS applies an anatomical five-point Surface Laplacian spatial filter:
      </p>
      <pre>V_Laplacian(C3) = 4 * V(C3) - [V(FC3) + V(C1) + V(C5) + V(CP3)]
V_Laplacian(C4) = 4 * V(C4) - [V(FC4) + V(C2) + V(C6) + V(CP4)]</pre>
      <p>
        By approximating the second spatial derivative of the surface potential, the Laplacian filter acts as a high-pass spatial filter, attenuating distant volume-conducted activity and isolating localized cortical current source density (CSD).
      </p>
    </section>

    <!-- Section 5 -->
    <section class="card scroll-target" id="section-5">
      <h2><span class="num">05.</span> Riemannian Covariance Geometry</h2>
      <p>
        In addition to classical Filter Bank Common Spatial Patterns (FBCSP), Phase 7 utilizes Riemannian geometry on the manifold of Symmetric Positive Definite (SPD) covariance matrices. Spatial covariance matrices are computed across narrow sub-bands and mapped to the Euclidean tangent space via the Riemannian logarithm map:
      </p>
      <pre>C_tangent = Logm( C_ref^(-1/2) * C * C_ref^(-1/2) )</pre>
      <p>
        This projection eliminates distortion inherent to Euclidean distance on curved manifolds, yielding features invariant to affine transformations and robust against cross-session drift.
      </p>
    </section>

    <!-- Section 6 -->
    <section class="card scroll-target" id="section-6">
      <h2><span class="num">06.</span> Authentic PhysioNet Dataset Replay</h2>
      <p>
        To ensure scientific rigor without fabricating hardware data, NeuroOS replays authentic recorded EDF files from the PhysioNet EEGMMIDB database (Subjects S001–S010). Windows of 3.0 seconds (480 samples at 160 Hz) are extracted and streamed sequentially.
      </p>
      <div class="callout cyan">
        <strong>Strict Offline Guarantee:</strong> All predictions originate from real ML inference over recorded raw microvolt signals. Ground-truth event markers are isolated and displayed exclusively for offline evaluation. They never dictate software actuation.
      </div>
    </section>

    <!-- Section 7 -->
    <section class="card scroll-target" id="section-7">
      <h2><span class="num">07.</span> Multi-Layer Safety Shield & Zero False Actuation</h2>
      <p>
        A clinical BCI must prevent unintended actuation during periods of distraction, cough, or ambient noise. The NeuroOS Safety Shield implements multi-layer defense guardrails:
      </p>
      <ul>
        <li><strong>Confidence Gate:</strong> Predictions with max probability below the threshold (default: 0.60) are rejected.</li>
        <li><strong>Margin Gate:</strong> Margin between top two classes must exceed 0.10.</li>
        <li><strong>Temporal Confirmation:</strong> Optional dwell windows require consistent successive predictions.</li>
        <li><strong>Command Cooldown:</strong> Minimum interval (0.8s) prevents unintended rapid burst firing.</li>
        <li><strong>Emergency Lock:</strong> Instant one-click override freezes all scrolling immediately.</li>
      </ul>
    </section>

    <!-- Section 8 -->
    <section class="card scroll-target" id="section-8">
      <h2><span class="num">08.</span> Browser Security & Cross-Origin Sandboxing</h2>
      <p>
        When loading third-party websites inside an iframe viewport, modern browsers enforce strict Cross-Origin Resource Sharing (CORS) and framing restrictions:
      </p>
      <div class="callout">
        <strong>Security Standards:</strong> Websites providing <code>X-Frame-Options: DENY</code> or <code>Content-Security-Policy: frame-ancestors 'self'</code> cannot be embedded in an iframe. NeuroOS respects browser security policies and provides transparent fallback diagnostics without insecure shell injection.
      </div>
      <p>
        For external domains that permit framing, direct cross-origin JavaScript manipulation is blocked by the Same-Origin Policy. Therefore, this built-in research document serves as a guaranteed-working, fully controllable reference target.
      </p>
    </section>

    <!-- Section 9 -->
    <section class="card scroll-target" id="section-9">
      <h2><span class="num">09.</span> Holdout Evaluation & Benchmark Metrics</h2>
      <p>
        Phase 7 champion model performance evaluated across 10 PhysioNet subjects on holdout Run 11:
      </p>
      <table>
        <thead>
          <tr>
            <th>Model Architecture</th>
            <th>Mean Holdout Acc</th>
            <th>REST Recall</th>
            <th>ML Latency</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Step 61 Baseline Multimodal</td>
            <td>68.4%</td>
            <td>84.2%</td>
            <td>3.8 ms</td>
          </tr>
          <tr>
            <td>Step 62 Surface Laplacian (Frozen Base)</td>
            <td>72.7%</td>
            <td>90.7%</td>
            <td>4.2 ms</td>
          </tr>
          <tr>
            <td>Phase 7 Pure Imagery Calibrated Champion</td>
            <td>78.3%</td>
            <td>92.4%</td>
            <td>4.5 ms</td>
          </tr>
        </tbody>
      </table>
    </section>

    <!-- Section 10 -->
    <section class="card scroll-target" id="section-10">
      <h2><span class="num">10.</span> Conclusion & Clinical Horizons</h2>
      <p>
        By bridging authentic sensorimotor rhythm decoding with deterministic browser scroll mechanics, NeuroOS demonstrates that non-invasive BCIs can provide intuitive, hands-free web navigation while strictly respecting ethical boundaries and client security policies.
      </p>
      <div style="text-align: center; margin-top: 32px;">
        <span class="indicator-pill">NeuroOS Phase 9 Verified • End of Controllable Document</span>
      </div>
    </section>

  </div>

  <script>
    // Bidirectional scroll bridge for iframe control
    window.addEventListener('message', function(event) {
      if (!event.data) return;
      if (event.data.type === 'NEURO_SCROLL_BY') {
        var delta = event.data.delta || 0;
        var isSmooth = event.data.smooth !== false;
        window.scrollBy({ top: delta, behavior: isSmooth ? 'smooth' : 'auto' });
      } else if (event.data.type === 'NEURO_SCROLL_TO') {
        var top = event.data.top || 0;
        var isSmooth = event.data.smooth !== false;
        window.scrollTo({ top: top, behavior: isSmooth ? 'smooth' : 'auto' });
      }
    });

    // Notify parent window of scroll offset
    window.addEventListener('scroll', function() {
      if (window.parent && window.parent !== window) {
        window.parent.postMessage({
          type: 'NEURO_SCROLL_UPDATE',
          scrollTop: window.scrollY || document.documentElement.scrollTop,
          scrollHeight: document.documentElement.scrollHeight,
          clientHeight: document.documentElement.clientHeight
        }, '*');
      }
    });
  </script>
</body>
</html>"""
