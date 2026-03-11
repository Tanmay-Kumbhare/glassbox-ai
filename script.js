// ═══════════════════════════════════════════════════════════════
//  Teachable Machine v1.5
//  Clean v1 base + dynamic classes (up to 5) +
//  training charts + architecture diagram +
//  embedding distance meter + confidence timeline
// ═══════════════════════════════════════════════════════════════

// ── Colour palette for up to 5 classes ───────────────────────
const PALETTE = [
  { color: '#667eea', bg: '#ebf4ff', border: '#bee3f8', text: '#2b6cb0' },
  { color: '#f6ad55', bg: '#fef3c7', border: '#fde68a', text: '#92400e' },
  { color: '#48bb78', bg: '#f0fff4', border: '#c6f6d5', text: '#276749' },
  { color: '#f56565', bg: '#fff5f5', border: '#fed7d7', text: '#c53030' },
  { color: '#9f7aea', bg: '#faf5ff', border: '#e9d8fd', text: '#553c9a' },
];
const MAX_CLASSES = 5;
const EPOCHS      = 25;

// ── State ─────────────────────────────────────────────────────
let mobilenetModel     = null;
let classifier         = null;
let modelTrained       = false;
let classes            = [];      // [{id, name, embeddings:[]}]
let nextClassId        = 0;
let webcamReady        = false;
let collectionInterval = null;
let predInterval       = null;
let activeCollectId    = null;

// ── Chart instances (lightweight — only updated on events) ────
let lossChart      = null;
let accChart       = null;
let timelineChart  = null;
const timelineMax  = 40;   // max points on confidence timeline

// ── DOM ───────────────────────────────────────────────────────
const statusBar      = document.getElementById('status-bar');
const uploadArea     = document.getElementById('uploadArea');
const imageUpload    = document.getElementById('imageUpload');
const preview        = document.getElementById('preview');
const classesWrap    = document.getElementById('classes-wrap');
const addClassBtn    = document.getElementById('addClassBtn');
const statSamples    = document.getElementById('statSamples');
const statClasses    = document.getElementById('statClasses');
const trainBtn       = document.getElementById('trainBtn');
const progressBar    = document.getElementById('progressBar');
const trainLog       = document.getElementById('trainLog');
const webcamEl       = document.getElementById('webcam');
const startWebcamBtn = document.getElementById('startWebcamBtn');
const camBtnRow      = document.getElementById('camBtnRow');
const collectStatus  = document.getElementById('collectStatus');
const predBars       = document.getElementById('predBars');
const predWinner     = document.getElementById('predWinner');
const predictImgBtn  = document.getElementById('predictImgBtn');
const startLiveBtn   = document.getElementById('startLiveBtn');
const stopLiveBtn    = document.getElementById('stopLiveBtn');
const resetBtn       = document.getElementById('resetBtn');
const distPairs      = document.getElementById('distPairs');
const distNote       = document.getElementById('distNote');
const archSvg        = document.getElementById('arch-svg');

const pipeEls = {
  load:    document.getElementById('ps-load'),
  collect: document.getElementById('ps-collect'),
  embed:   document.getElementById('ps-embed'),
  train:   document.getElementById('ps-train'),
  predict: document.getElementById('ps-predict'),
};

// ── Helpers ───────────────────────────────────────────────────
function setStatus(msg, type = '') {
  statusBar.textContent = msg;
  statusBar.className = type;
}

function setPipe(step) {
  const order = ['load','collect','embed','train','predict'];
  const idx = order.indexOf(step);
  order.forEach((k, i) => {
    pipeEls[k].classList.remove('active','done');
    if (i < idx)  pipeEls[k].classList.add('done');
    if (i === idx) pipeEls[k].classList.add('active');
  });
}

// ── MobileNet ─────────────────────────────────────────────────
async function loadMobileNet() {
  setPipe('load');
  setStatus('⏳ Loading MobileNet… first load may take ~10 seconds.');
  try {
    mobilenetModel = await mobilenet.load();
    buildSpatialModel();   // ← pre-build GradCAM sub-model once
    pipeEls.load.classList.replace('active','done');
    setStatus('✅ MobileNet ready. Collect samples for each class to get started.', 'ready');
    addClassBtn.disabled = false;
    // Default 2 classes
    addNewClass('Class A');
    addNewClass('Class B');
    drawArchDiagram();
  } catch(e) {
    setStatus('❌ Failed to load MobileNet. Check your internet connection.', 'error');
  }
}
loadMobileNet();

// ── Classifier builder ────────────────────────────────────────
function buildClassifier(n) {
  if (classifier) classifier.dispose();
  classifier = tf.sequential();
  classifier.add(tf.layers.dense({ inputShape:[1024], units:128, activation:'relu' }));
  classifier.add(tf.layers.dropout({ rate:0.3 }));
  classifier.add(tf.layers.dense({ units:64, activation:'relu' }));
  classifier.add(tf.layers.dense({ units:n, activation:'softmax' }));
  classifier.compile({
    optimizer: tf.train.adam(0.0005),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  modelTrained = false;
}

// ── Embedding extraction ──────────────────────────────────────
function extractEmbedding(src) {
  if (!mobilenetModel) return null;
  if (src instanceof HTMLVideoElement && (!src.videoWidth || !src.videoHeight)) return null;
  return tf.tidy(() => mobilenetModel.infer(src, true));
}

// ── Dynamic class management ──────────────────────────────────
function addNewClass(name) {
  if (classes.length >= MAX_CLASSES) return;
  const id    = nextClassId++;
  const pal   = PALETTE[classes.length % PALETTE.length];
  classes.push({ id, name: name || `Class ${String.fromCharCode(65 + classes.length)}`, embeddings:[], pal });
  renderClasses();
  renderPredBars();
  updateStats();
}

function deleteClass(id) {
  const cls = classes.find(c => c.id === id);
  if (cls) cls.embeddings.forEach(t => t.dispose());
  classes = classes.filter(c => c.id !== id);
  renderClasses();
  renderPredBars();
  updateStats();
  updateDistancePanel();
}

function clearClassSamples(id) {
  const cls = classes.find(c => c.id === id);
  if (!cls) return;
  cls.embeddings.forEach(t => t.dispose());
  cls.embeddings = [];
  updateCountEl(id);
  updateStats();
  checkTrainReady();
  updateDistancePanel();
}

function updateCountEl(id) {
  const el = document.getElementById(`cnt-${id}`);
  const cls = classes.find(c => c.id === id);
  if (el && cls) el.textContent = cls.embeddings.length;
}

function renderClasses() {
  classesWrap.innerHTML = '';
  classes.forEach(cls => {
    const p = cls.pal;
    const div = document.createElement('div');
    div.className = 'class-row';
    div.style.cssText = `background:${p.bg};border-color:${p.border};--cc:${p.color}`;
    div.innerHTML = `
      <div class="class-row-top">
        <div class="cc-dot"></div>
        <span class="cc-name" style="color:${p.text}">${cls.name}</span>
        <span class="cc-count">Samples: <b id="cnt-${cls.id}">${cls.embeddings.length}</b></span>
        ${classes.length > 2 ? `<button class="btn btn-xs btn-red" onclick="deleteClass(${cls.id})" style="margin-left:4px;">✕</button>` : ''}
      </div>
      <div class="class-row-btns">
        <button class="btn btn-xs btn-outline" id="addImgBtn-${cls.id}" onclick="addSampleFromImage(${cls.id})" disabled
          style="border-color:${p.border};color:${p.text}">🖼 Add Image</button>
        <button class="btn btn-xs btn-outline" id="collectBtn-${cls.id}" onclick="startCollection(${cls.id})" disabled
          style="border-color:${p.border};color:${p.text}">⏺ Webcam</button>
        <button class="btn btn-xs" onclick="clearClassSamples(${cls.id})"
          style="background:#f7fafc;border:1.5px solid #e2e8f0;color:#718096;">🗑</button>
      </div>
    `;
    classesWrap.appendChild(div);
    updateAddImgBtn(cls.id);
    updateCollectBtn(cls.id);
  });
  addClassBtn.disabled = classes.length >= MAX_CLASSES;
}

function updateAddImgBtn(id) {
  const btn = document.getElementById(`addImgBtn-${id}`);
  if (btn) btn.disabled = !(preview.src && preview.naturalWidth > 0);
}
function updateCollectBtn(id) {
  const btn = document.getElementById(`collectBtn-${id}`);
  if (btn) btn.disabled = !webcamReady;
}
function updateAllButtons() {
  classes.forEach(c => { updateAddImgBtn(c.id); updateCollectBtn(c.id); });
}

function updateStats() {
  const total = classes.reduce((s,c) => s + c.embeddings.length, 0);
  statSamples.textContent = total;
  statClasses.textContent = classes.length;
}

function checkTrainReady() {
  trainBtn.disabled = !(classes.length >= 2 && classes.every(c => c.embeddings.length >= 2));
}

addClassBtn.addEventListener('click', () => addNewClass());

// ── Upload ────────────────────────────────────────────────────
uploadArea.addEventListener('click', () => imageUpload.click());
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault(); uploadArea.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) readFile(f);
});
imageUpload.addEventListener('change', e => { if (e.target.files[0]) readFile(e.target.files[0]); });

function readFile(file) {
  const reader = new FileReader();
  reader.onload = ev => {
    preview.src = ev.target.result;
    preview.style.display = 'block';
    preview.onload = () => {
      setPipe('collect');
      setStatus('🖼 Image loaded. Click "Add Image" under any class to label it.', 'ready');
      updateAllButtons();
      syncReplayButtons();
    };
  };
  reader.readAsDataURL(file);
}

// ── Add sample from image ─────────────────────────────────────
function addSampleFromImage(id) {
  if (!preview.src || !preview.naturalWidth) return setStatus('Upload an image first.', 'error');
  const cls = classes.find(c => c.id === id);
  if (!cls) return;
  setPipe('embed');
  const emb = extractEmbedding(preview);
  if (!emb) return;
  cls.embeddings.push(emb);
  updateCountEl(id);
  updateStats();
  checkTrainReady();
  setStatus(`✅ Added to "${cls.name}" — ${cls.embeddings.length} sample${cls.embeddings.length > 1 ? 's' : ''}.`, 'ready');
  // Update distance panel after adding
  scheduleDistanceUpdate();
}
window.addSampleFromImage = addSampleFromImage;
window.deleteClass        = deleteClass;
window.clearClassSamples  = clearClassSamples;

// ── Debounce distance update (avoid heavy ops on every click) ─
let distTimer = null;
function scheduleDistanceUpdate() {
  clearTimeout(distTimer);
  distTimer = setTimeout(updateDistancePanel, 600);
}

// ── Epoch Replay — saved weight snapshots ────────────────────
// We save classifier weights every epoch during training.
// Each snapshot = array of Float32Arrays (one per weight tensor).
// Memory: 25 epochs × ~100KB weights = ~2.5MB — totally fine.
let epochSnapshots  = [];   // [{epoch, weights:[], loss, acc}]
let replayTestEmb   = null; // Float32Array — embedding of test image
let replayTestSrc   = null; // 'upload' | 'webcam'
let replayInterval  = null;

// ── Train ─────────────────────────────────────────────────────
trainBtn.addEventListener('click', async () => {
  if (classes.length < 2) return setStatus('Need at least 2 classes.', 'error');
  if (classes.some(c => c.embeddings.length < 2))
    return setStatus('Each class needs at least 2 samples.', 'error');

  setPipe('train');
  trainBtn.disabled = predictImgBtn.disabled = startLiveBtn.disabled = true;
  progressBar.style.width = '0%';
  await checkSampleVariance();   // ← warn if samples are too similar

  // Reset charts
  resetTrainingCharts();
  buildClassifier(classes.length);
  setStatus('🏋️ Training… watch the charts update live!');

  const xs = tf.concat(classes.flatMap(c => c.embeddings));
  const ys = tf.tensor2d(
    classes.flatMap((c, ci) =>
      c.embeddings.map(() => { const a = Array(classes.length).fill(0); a[ci]=1; return a; })
    )
  );

  epochSnapshots = [];   // clear previous run

  try {
    await classifier.fit(xs, ys, {
      epochs: EPOCHS,
      batchSize: Math.min(16, xs.shape[0]),
      shuffle: true,
      validationSplit: 0.1,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const pct = ((epoch+1)/EPOCHS*100).toFixed(0);
          const acc = logs.acc ?? logs.accuracy ?? 0;
          progressBar.style.width = pct + '%';
          trainLog.textContent    = `Epoch ${epoch+1}/${EPOCHS} — Loss: ${logs.loss.toFixed(4)} | Acc: ${(acc*100).toFixed(1)}%`;
          pushTrainingCharts(epoch+1, logs.loss, acc);

          // ── Save weight snapshot for epoch replay ──────────
          // classifier.getWeights() returns tf.Tensor[]. We
          // immediately copy each to a plain Float32Array so we
          // hold no extra GPU tensors between epochs.
          const snap = {
            epoch: epoch + 1,
            loss:  logs.loss,
            acc:   acc,
            weights: classifier.getWeights().map(w => {
              const arr = w.dataSync();           // synchonous read
              const copy = new Float32Array(arr); // detached copy
              // w is still owned by the model — do NOT dispose
              return { data: copy, shape: w.shape };
            })
          };
          epochSnapshots.push(snap);
        }
      }
    });

    modelTrained = true;
    await computeClassMeans();   // ← build mean vectors for "Why" box
    setPipe('predict');
    setStatus('🎉 Training complete! Predict an image or start live prediction.', 'ready');
    trainLog.textContent   = '✅ Model trained successfully!';
    trainBtn.disabled      = false;
    predictImgBtn.disabled = false;
    startLiveBtn.disabled  = false;
    drawArchDiagram();
    updateDistancePanel();
    initReplayCard();     // ← unlock epoch replay UI
  } catch(e) {
    setStatus('❌ Training failed: ' + e.message, 'error');
    trainBtn.disabled = false;
  } finally {
    xs.dispose(); ys.dispose();
  }
});

// ── Class mean embeddings (computed after training for "Why" box) ─
let classMeans = [];   // Float32Array per class, filled after training

async function computeClassMeans() {
  classMeans = [];
  for (const cls of classes) {
    if (!cls.embeddings.length) { classMeans.push(null); continue; }
    const stacked = tf.stack(cls.embeddings);
    const mean    = tf.mean(stacked, 0);
    classMeans.push(await mean.data());
    stacked.dispose(); mean.dispose();
  }
}

function cosineSim(a, b) {
  let dot=0, mA=0, mB=0;
  for (let i=0; i<a.length; i++) { dot+=a[i]*b[i]; mA+=a[i]*a[i]; mB+=b[i]*b[i]; }
  return dot / (Math.sqrt(mA) * Math.sqrt(mB) || 1);
}
function renderPredBars() {
  if (!classes.length) { predBars.innerHTML = '<div style="font-size:0.82rem;color:#a0aec0;">Train the model first, then predict here.</div>'; return; }
  predBars.innerHTML = '';
  classes.forEach(cls => {
    const p = cls.pal;
    const d = document.createElement('div');
    d.className = 'pred-row';
    d.innerHTML = `
      <div class="pred-hdr">
        <span style="display:flex;align-items:center;gap:6px;">
          <span style="width:8px;height:8px;border-radius:50%;background:${p.color};display:inline-block;"></span>
          ${cls.name}
        </span>
        <span id="pct-${cls.id}" style="font-size:0.78rem;font-family:monospace;">—</span>
      </div>
      <div class="pred-track">
        <div class="pred-fill" id="bar-${cls.id}" style="background:${p.color};"></div>
      </div>`;
    predBars.appendChild(d);
  });
}

function showPrediction(probs, currentEmbData) {
  classes.forEach((cls, i) => {
    const pct = (probs[i]*100).toFixed(1);
    const pe  = document.getElementById(`pct-${cls.id}`);
    const be  = document.getElementById(`bar-${cls.id}`);
    if (pe) pe.textContent = pct + '%';
    if (be) be.style.width  = pct + '%';
  });
  const maxI   = Array.from(probs).indexOf(Math.max(...probs));
  const winner = classes[maxI];
  if (winner) {
    predWinner.style.display    = 'block';
    predWinner.textContent      = `🏆 ${winner.name}  (${(probs[maxI]*100).toFixed(1)}%)`;
    predWinner.style.background = winner.pal.bg;
    predWinner.style.color      = winner.pal.text;
    predWinner.style.border     = `1.5px solid ${winner.pal.border}`;
  }
  // Update "Why" explanation if we have current embedding data
  if (currentEmbData) updateWhyBox(probs, currentEmbData, maxI);
}

// ── "Why" explanation box ─────────────────────────────────────
function updateWhyBox(probs, embData, winnerIdx) {
  const box = document.getElementById('whyBox');
  if (!box) return;
  if (!classMeans.length || !classMeans[winnerIdx]) { box.textContent = ''; return; }

  const winner     = classes[winnerIdx];
  const confidence = probs[winnerIdx] * 100;

  // Cosine similarity of current embedding vs each class mean
  const sims = classMeans.map((m, i) => m ? cosineSim(embData, m) : 0);
  const winSim = (sims[winnerIdx] * 100).toFixed(0);

  // Find second best class
  const sorted = probs.map((p,i) => ({p,i})).sort((a,b) => b.p - a.p);
  const second = sorted[1];

  let msg = '';
  if (confidence > 90) {
    msg = `✅ Very confident — your input is <b>${winSim}% similar</b> to the "${winner.name}" training samples.`;
  } else if (confidence > 65) {
    const secondName = classes[second.i]?.name || '';
    msg = `🟡 Moderately confident — leaning toward "${winner.name}" but ${(second.p*100).toFixed(0)}% chance it's "${secondName}". Try moving closer or changing the angle.`;
  } else {
    msg = `⚠️ Uncertain — the input doesn't clearly match either class. The model sees it as <b>between clusters</b>. Add more varied training samples.`;
  }
  box.innerHTML = msg;
}

// ── Pre-training sample variance warning ──────────────────────
async function checkSampleVariance() {
  const warnEl = document.getElementById('varianceWarn');
  if (!warnEl) return;
  const warnings = [];
  for (const cls of classes) {
    if (cls.embeddings.length < 3) continue;
    // Stack all embeddings and compute std dev across samples
    const stacked = tf.stack(cls.embeddings);           // [N, 1024]
    const mean    = tf.mean(stacked, 0, true);           // [1, 1024]
    const diff    = stacked.sub(mean);
    const variance = tf.mean(tf.square(diff)).arraySync();
    stacked.dispose(); mean.dispose(); diff.dispose();
    if (variance < 0.005) {
      warnings.push(`"${cls.name}" (samples look very similar — add more variety)`);
    }
  }
  if (warnings.length) {
    warnEl.innerHTML = `⚠️ Low variety detected in: ${warnings.join(', ')}. Try different angles, distances, or lighting.`;
    warnEl.style.display = 'block';
  } else {
    warnEl.style.display = 'none';
  }
}

// ── Predict image ─────────────────────────────────────────────
predictImgBtn.addEventListener('click', async () => {
  if (!modelTrained) return setStatus('Train the model first.', 'error');
  if (!preview.src || !preview.naturalWidth) return setStatus('Upload an image first.', 'error');
  const emb  = extractEmbedding(preview);
  const pred = classifier.predict(emb);
  const p    = await pred.data();
  emb.dispose(); pred.dispose();
  showPrediction(p);
});

// ── Webcam ────────────────────────────────────────────────────
startWebcamBtn.addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video:{ width:640, height:480 } });
    webcamEl.srcObject = stream;
    webcamEl.onloadedmetadata = () => {
      webcamEl.play();
      webcamReady = true;
      startWebcamBtn.textContent = '✅ Camera On';
      startWebcamBtn.disabled    = true;
      setStatus('📷 Webcam ready! Click a class Webcam button to collect samples.', 'ready');
      updateAllButtons();
      renderCamCollectBtns();
      syncReplayButtons();
    };
  } catch(e) { setStatus('❌ Camera access denied. Allow camera in browser settings.', 'error'); }
});

function renderCamCollectBtns() {
  // Remove existing collect/stop buttons
  camBtnRow.querySelectorAll('.cam-collect, #stopCollectBtn').forEach(b => b.remove());
  // Stop button
  const stopBtn = document.createElement('button');
  stopBtn.className = 'btn btn-stop btn-sm';
  stopBtn.id = 'stopCollectBtn';
  stopBtn.textContent = '⏹ Stop';
  stopBtn.disabled = true;
  stopBtn.onclick = stopCollection;
  camBtnRow.appendChild(stopBtn);
}

function startCollection(id) {
  if (!webcamReady) return setStatus('Start the webcam first.', 'error');
  stopCollection();
  activeCollectId = id;
  const cls = classes.find(c => c.id === id);
  if (!cls) return;

  // Pulse the webcam collect button in the class card
  const btn = document.getElementById(`collectBtn-${id}`);
  if (btn) btn.classList.add('collecting');
  const stopBtn = document.getElementById('stopCollectBtn');
  if (stopBtn) stopBtn.disabled = false;

  collectStatus.textContent = `⏺ Collecting for "${cls.name}"…`;
  setPipe('embed');

  collectionInterval = setInterval(() => {
    if (!webcamEl.videoWidth) return;
    const emb = extractEmbedding(webcamEl);
    if (!emb) return;
    const c = classes.find(c => c.id === id);
    if (!c) { stopCollection(); return; }
    c.embeddings.push(emb);
    updateCountEl(id);
    updateStats();
    checkTrainReady();
    collectStatus.textContent = `⏺ "${c.name}" — ${c.embeddings.length} samples`;
  }, 200); // 5 per second, same as real Teachable Machine
}
window.startCollection = startCollection;

function stopCollection() {
  if (collectionInterval) { clearInterval(collectionInterval); collectionInterval = null; }
  document.querySelectorAll('.collecting').forEach(b => b.classList.remove('collecting'));
  const stopBtn = document.getElementById('stopCollectBtn');
  if (stopBtn) stopBtn.disabled = true;
  if (activeCollectId !== null) {
    const cls = classes.find(c => c.id === activeCollectId);
    if (cls) collectStatus.textContent = `✅ Stopped — "${cls.name}": ${cls.embeddings.length} samples`;
    scheduleDistanceUpdate();
  }
  activeCollectId = null;
}
window.stopCollection = stopCollection;

// ── Live prediction ───────────────────────────────────────────
startLiveBtn.addEventListener('click', () => {
  if (!modelTrained) return setStatus('Train the model first.', 'error');
  if (!webcamReady)  return setStatus('Start the webcam first.', 'error');
  stopLive();
  stopLiveBtn.disabled  = false;
  startLiveBtn.disabled = true;
  predWinner.style.display = 'block';
  const whyBox = document.getElementById('whyBox');
  if (whyBox) whyBox.style.display = 'block';
  setStatus('🔴 Live prediction running…', 'ready');
  initTimelineChart();
  inspectorActivate();   // ← light up inspector panels

  predInterval = setInterval(async () => {
    if (!webcamEl.videoWidth) return;
    const emb  = extractEmbedding(webcamEl);
    const pred = classifier.predict(emb);
    const p    = await pred.data();
    const embData = await emb.data();   // read before dispose for why-box
    emb.dispose(); pred.dispose();
    showPrediction(p, embData);
    pushTimeline(p);
    await runInspector(p);
  }, 200);
});

stopLiveBtn.addEventListener('click', stopLive);
function stopLive() {
  if (predInterval) { clearInterval(predInterval); predInterval = null; }
  stopLiveBtn.disabled  = true;
  startLiveBtn.disabled = !modelTrained;
  inspectorDeactivate();  // ← dim panels when stopped
}

// ── Reset ─────────────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  if (!confirm('Reset everything? All samples and training will be lost.')) return;
  stopCollection(); stopLive();
  stopReplayAuto();
  epochSnapshots = [];
  replayTestEmb  = null;
  if (replayCard) {
    replayCard.style.display = 'none';
    replayBars.innerHTML = '';
    replayInsight.textContent = 'Train the model first, then select a test image to begin.';
  }
  classes.forEach(c => c.embeddings.forEach(t => t.dispose()));
  classes = []; nextClassId = 0;
  modelTrained = false;
  preview.style.display = 'none'; preview.src = '';
  predWinner.style.display = 'none';
  const whyBox = document.getElementById('whyBox');
  if (whyBox) { whyBox.style.display = 'none'; whyBox.textContent = ''; }
  progressBar.style.width = '0%';
  trainLog.textContent = '—';
  predictImgBtn.disabled = startLiveBtn.disabled = true;
  collectStatus.textContent = '';
  distPairs.innerHTML = '<div style="font-size:0.82rem;color:#a0aec0;">Add samples to at least 2 classes to see how separable they are.</div>';
  distNote.textContent = '';
  resetTrainingCharts();
  if (timelineChart) { timelineChart.destroy(); timelineChart = null; }
  setPipe('load'); pipeEls.load.classList.add('done');
  setStatus('🔄 Reset complete. Start collecting samples!', 'ready');
  addClassBtn.disabled = false;
  buildClassifier(2);
  addNewClass('Class A'); addNewClass('Class B');
  drawArchDiagram();
});

// ═══════════════════════════════════════════════════════════════
//  TRAINING CHARTS  (Chart.js — lightweight line charts)
// ═══════════════════════════════════════════════════════════════
const chartOpts = {
  responsive: true,
  maintainAspectRatio: true,
  animation: false,        // no animation = no lag
  plugins: {
    legend: { labels: { color:'#718096', font:{ size:10 } } }
  },
  scales: {
    x: { ticks:{ color:'#a0aec0', font:{size:9} }, grid:{ color:'#f0f4f8' } },
    y: { ticks:{ color:'#a0aec0', font:{size:9} }, grid:{ color:'#f0f4f8' } }
  }
};

function resetTrainingCharts() {
  if (lossChart) lossChart.destroy();
  if (accChart)  accChart.destroy();

  lossChart = new Chart(document.getElementById('lossChart'), {
    type: 'line',
    data: { labels:[], datasets:[{
      label:'Loss', data:[], borderColor:'#f6ad55',
      backgroundColor:'rgba(246,173,85,0.1)', borderWidth:2,
      pointRadius:0, tension:0.3, fill:true
    }]},
    options: { ...chartOpts, scales:{ ...chartOpts.scales, y:{ ...chartOpts.scales.y, min:0 } } }
  });

  accChart = new Chart(document.getElementById('accChart'), {
    type: 'line',
    data: { labels:[], datasets:[{
      label:'Accuracy', data:[], borderColor:'#48bb78',
      backgroundColor:'rgba(72,187,120,0.1)', borderWidth:2,
      pointRadius:0, tension:0.3, fill:true
    }]},
    options: { ...chartOpts, scales:{ ...chartOpts.scales, y:{ ...chartOpts.scales.y, min:0, max:1 } } }
  });
}
resetTrainingCharts();

function pushTrainingCharts(epoch, loss, acc) {
  lossChart.data.labels.push(`E${epoch}`);
  lossChart.data.datasets[0].data.push(+loss.toFixed(4));
  lossChart.update('none');   // 'none' = skip animation, zero lag

  accChart.data.labels.push(`E${epoch}`);
  accChart.data.datasets[0].data.push(+acc.toFixed(4));
  accChart.update('none');
}

// ═══════════════════════════════════════════════════════════════
//  CONFIDENCE TIMELINE  (live scrolling chart)
// ═══════════════════════════════════════════════════════════════
function initTimelineChart() {
  if (timelineChart) timelineChart.destroy();

  const datasets = classes.map(cls => ({
    label: cls.name,
    data: [],
    borderColor: cls.pal.color,
    backgroundColor: 'transparent',
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.3,
  }));

  timelineChart = new Chart(document.getElementById('timelineChart'), {
    type: 'line',
    data: { labels:[], datasets },
    options: {
      ...chartOpts,
      scales: {
        ...chartOpts.scales,
        y: { ...chartOpts.scales.y, min:0, max:100,
          ticks:{ ...chartOpts.scales.y.ticks, callback: v => v+'%' } }
      }
    }
  });
}

let timelineTick = 0;
function pushTimeline(probs) {
  if (!timelineChart) return;
  timelineTick++;
  const lbl = `${timelineTick}`;

  // Slide window
  if (timelineChart.data.labels.length >= timelineMax) {
    timelineChart.data.labels.shift();
    timelineChart.data.datasets.forEach(ds => ds.data.shift());
  }

  timelineChart.data.labels.push(lbl);
  classes.forEach((cls, i) => {
    if (timelineChart.data.datasets[i])
      timelineChart.data.datasets[i].data.push(+(probs[i]*100).toFixed(1));
  });
  timelineChart.update('none');
}

// ═══════════════════════════════════════════════════════════════
//  ARCHITECTURE DIAGRAM  (SVG — static, zero compute)
// ═══════════════════════════════════════════════════════════════
function drawArchDiagram() {
  const n = classes.length || 2;

  // Layer definitions
  const layers = [
    { label:'Your Image',  sub:'224×224px',        neurons: 0,   special:'img',   color:'#667eea' },
    { label:'MobileNet',   sub:'1.0 (frozen)',      neurons: 0,   special:'frozen',color:'#9f7aea' },
    { label:'Embedding',   sub:'1024 features',     neurons: 8,   special:'',      color:'#667eea' },
    { label:'Dense(128)',  sub:'ReLU activation',   neurons: 6,   special:'',      color:'#48bb78' },
    { label:'Dropout',     sub:'30% rate',          neurons: 0,   special:'drop',  color:'#f6ad55' },
    { label:'Dense(64)',   sub:'ReLU activation',   neurons: 5,   special:'',      color:'#48bb78' },
    { label:'Output',      sub:`${n} classes · Softmax`, neurons: Math.min(n, 5), special:'out', color:'#f56565' },
  ];

  const W = 780, H = 130;
  const colW = W / layers.length;

  let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;font-family:Segoe UI,sans-serif;">`;

  // Connector lines between layer columns
  for (let i = 0; i < layers.length-1; i++) {
    const x1 = colW * i + colW * 0.78;
    const x2 = colW * (i+1) + colW * 0.22;
    svg += `<line x1="${x1}" y1="50" x2="${x2}" y2="50" stroke="#e2e8f0" stroke-width="2" stroke-dasharray="${layers[i+1].special==='frozen'?'4,3':'0'}"/>`;
  }

  layers.forEach((l, i) => {
    const cx = colW * i + colW / 2;

    if (l.special === 'img') {
      // Image icon
      svg += `<rect x="${cx-18}" y="26" width="36" height="28" rx="5" fill="${l.color}" opacity="0.15" stroke="${l.color}" stroke-width="1.5"/>`;
      svg += `<text x="${cx}" y="47" text-anchor="middle" font-size="16">🖼</text>`;
    } else if (l.special === 'frozen') {
      // MobileNet block
      svg += `<rect x="${cx-22}" y="22" width="44" height="36" rx="6" fill="${l.color}" opacity="0.15" stroke="${l.color}" stroke-width="1.5"/>`;
      svg += `<text x="${cx}" y="38" text-anchor="middle" font-size="9" font-weight="700" fill="${l.color}">MobileNet</text>`;
      svg += `<text x="${cx}" y="50" text-anchor="middle" font-size="8" fill="#9f7aea">FROZEN</text>`;
    } else if (l.special === 'drop') {
      // Dropout dotted box
      svg += `<rect x="${cx-18}" y="28" width="36" height="24" rx="4" fill="none" stroke="${l.color}" stroke-width="1.5" stroke-dasharray="4,2"/>`;
      svg += `<text x="${cx}" y="44" text-anchor="middle" font-size="9" fill="${l.color}">30%</text>`;
    } else {
      // Neuron circles
      const dots = l.neurons;
      const r    = 5;
      const gap  = 13;
      const totalH = dots * gap - gap * 0.3;
      const startY = 50 - totalH/2;
      for (let d = 0; d < dots; d++) {
        const cy = startY + d * gap;
        svg += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${l.color}" opacity="0.8"/>`;
      }
      if (l.special === 'out') {
        // plus sign if more classes than shown
        if (n > 5) svg += `<text x="${cx}" y="${startY + dots*gap + 6}" text-anchor="middle" font-size="9" fill="#a0aec0">+${n-5} more</text>`;
      }
    }

    // Label below
    svg += `<text x="${cx}" y="96" text-anchor="middle" font-size="9.5" font-weight="700" fill="#4a5568">${l.label}</text>`;
    svg += `<text x="${cx}" y="110" text-anchor="middle" font-size="8" fill="#a0aec0">${l.sub}</text>`;
  });

  svg += '</svg>';
  archSvg.innerHTML = svg;
}

// ═══════════════════════════════════════════════════════════════
//  EMBEDDING DISTANCE PANEL
//  Computes cosine similarity between mean embeddings of pairs.
//  Only runs after collecting stops / training ends — NOT on every frame.
// ═══════════════════════════════════════════════════════════════
async function updateDistancePanel() {
  const valid = classes.filter(c => c.embeddings.length > 0);
  if (valid.length < 2) {
    distPairs.innerHTML = '<div style="font-size:0.82rem;color:#a0aec0;">Add samples to at least 2 classes to see how separable they are.</div>';
    distNote.textContent = '';
    return;
  }

  // Compute mean embedding per class — dispose immediately after
  const means = await Promise.all(valid.map(async cls => {
    const stacked = tf.stack(cls.embeddings);
    const mean    = tf.mean(stacked, 0);
    const data    = await mean.data();
    stacked.dispose(); mean.dispose();
    return { cls, data };
  }));

  // Build all pairs
  const pairs = [];
  for (let i = 0; i < means.length; i++) {
    for (let j = i+1; j < means.length; j++) {
      const a = means[i].data, b = means[j].data;
      // Cosine similarity
      let dot=0, magA=0, magB=0;
      for (let k=0; k<a.length; k++) { dot+=a[k]*b[k]; magA+=a[k]*a[k]; magB+=b[k]*b[k]; }
      const similarity = dot / (Math.sqrt(magA)*Math.sqrt(magB));
      const distance   = ((1 - similarity)*100);  // 0=identical, 100=totally different
      pairs.push({ a: means[i].cls, b: means[j].cls, distance });
    }
  }

  // Render
  distPairs.innerHTML = '';
  let lowestDist = Infinity;
  pairs.forEach(pair => {
    const d    = pair.distance.toFixed(1);
    const pct  = Math.min(100, pair.distance).toFixed(1);
    const hue  = Math.round(pair.distance * 1.2);   // green→red as distance grows
    const fill = `hsl(${hue}, 70%, 52%)`;

    if (pair.distance < lowestDist) lowestDist = pair.distance;

    const div = document.createElement('div');
    div.innerHTML = `
      <div class="dist-row-label">
        <span style="display:flex;align-items:center;gap:6px;">
          <span style="width:8px;height:8px;border-radius:50%;background:${pair.a.pal.color};display:inline-block;"></span>
          ${pair.a.name}
          <span style="color:#a0aec0">vs</span>
          <span style="width:8px;height:8px;border-radius:50%;background:${pair.b.pal.color};display:inline-block;"></span>
          ${pair.b.name}
        </span>
        <strong style="color:${fill}">${d}% different</strong>
      </div>
      <div class="dist-track">
        <div class="dist-fill" style="width:${pct}%;background:${fill};"></div>
      </div>`;
    distPairs.appendChild(div);
  });

  // Advice note
  const avgDist = pairs.reduce((s,p) => s+p.distance, 0) / pairs.length;
  let note = '';
  if (avgDist < 15)
    note = '⚠️ Your classes look <b>very similar</b> to the AI. The model may struggle — try more varied samples, or different subjects.';
  else if (avgDist < 35)
    note = '🟡 Classes are <b>moderately different</b>. Add more diverse samples to improve accuracy.';
  else
    note = '✅ Classes are <b>well separated</b> — the model should learn to distinguish them reliably.';
  distNote.innerHTML = note;
}

// ═══════════════════════════════════════════════════════════════
//  🔬 PIPELINE INSPECTOR
//  Runs inside the 200ms live-prediction interval.
//  5 canvas panels drawn with raw fillRect — no Chart.js, no lag.
//
//  Panel 1 — Raw webcam frame  (drawImage to 112×112 canvas)
//  Panel 2 — Resized 224×224   (tf.browser.toPixels on resized tensor)
//  Panel 3 — Normalised heatmap (pixel/127.5−1, blue→red colormap)
//  Panel 4 — Embedding sparkline (1024 values as tiny bars)
//  Panel 5 — Softmax bars       (one bar per class, coloured)
// ═══════════════════════════════════════════════════════════════

// Canvas refs — grabbed once
const insRawCanvas     = document.getElementById('ins-raw-canvas');
const insResizeCanvas  = document.getElementById('ins-resize-canvas');
const insNormCanvas    = document.getElementById('ins-norm-canvas');
const insEmbedCanvas   = document.getElementById('ins-embed-canvas');
const insSoftmaxCanvas = document.getElementById('ins-softmax-canvas');

const insRawCtx     = insRawCanvas.getContext('2d');
const insResizeCtx  = insResizeCanvas.getContext('2d');
const insNormCtx    = insNormCanvas.getContext('2d');
const insEmbedCtx   = insEmbedCanvas.getContext('2d');
const insSoftmaxCtx = insSoftmaxCanvas.getContext('2d');

const inspectorSub  = document.getElementById('inspectorSub');

function inspectorActivate() {
  ['ins-p1','ins-p2','ins-p3','ins-p4','ins-p5']
    .forEach(id => document.getElementById(id).classList.add('active'));
  inspectorSub.textContent = '🔴 Live — watching every step of the pipeline in real time.';
}

function inspectorDeactivate() {
  ['ins-p1','ins-p2','ins-p3','ins-p4','ins-p5']
    .forEach(id => document.getElementById(id).classList.remove('active'));
  inspectorSub.textContent = 'Start Live Prediction to see every preprocessing step visualised in real time.';
}

// Off-screen 224×224 canvas used for resized pixel readback
const offCanvas = document.createElement('canvas');
offCanvas.width = offCanvas.height = 224;
const offCtx = offCanvas.getContext('2d');

// ── Spatial sub-model for GradCAM (built once after MobileNet loads) ──
let spatialModel = null;   // outputs [1,7,7,1024] last conv feature map

function buildSpatialModel() {
  try {
    const base       = mobilenetModel.model;
    if (!base || !base.layers) return;
    // Find last conv layer before global avg pool — conv_pw_13_relu in MobileNet v1
    const layerNames = base.layers.map(l => l.name);
    const targetName = layerNames.find(n =>
      n.includes('conv_pw_13_relu') || n.includes('conv_pw_13')
    ) || layerNames.find(n => n.includes('conv_pw_') && !n.includes('conv_pw_1_'));
    if (!targetName) return;
    spatialModel = tf.model({
      inputs:  base.inputs,
      outputs: base.getLayer(targetName).output
    });
  } catch(e) { spatialModel = null; }
}

async function runInspector(softmaxProbs) {
  if (!webcamEl.videoWidth || !webcamEl.videoHeight) return;

  // ── Panel 1: Raw frame (always drawn first, immediately) ────
  insRawCtx.drawImage(webcamEl, 0, 0, insRawCanvas.width, insRawCanvas.height);

  let raw, resized, resized255, normalized, batched, normBatch, embedding, spatial;

  try {
    // ── fromPixels → resize ─────────────────────────────────
    raw     = tf.browser.fromPixels(webcamEl);               // [H,W,3] uint8
    resized = tf.image.resizeBilinear(raw, [224, 224]);      // [224,224,3] float
    raw.dispose();

    // ── Panel 2: Resized 224×224 ────────────────────────────
    resized255 = resized.clipByValue(0, 255).cast('int32');
    await tf.browser.toPixels(resized255, offCanvas);        // offCanvas = 224×224
    resized255.dispose();
    insResizeCtx.drawImage(offCanvas, 0, 0,
      insResizeCanvas.width, insResizeCanvas.height);

    // ── Normalise once, reuse for panels 3, 4, GradCAM ──────
    batched   = resized.expandDims(0);                       // [1,224,224,3]
    normBatch = batched.div(127.5).sub(1.0);                 // [-1,+1]
    batched.dispose();
    resized.dispose();

    // ── Panel 3: Normalised heatmap ─────────────────────────
    normalized      = normBatch.squeeze([0]);                // [224,224,3]
    const normData  = await normalized.data();               // Float32 [224*224*3]
    normalized.dispose();
    drawNormPanel(normData);

    // ── Panel 4: Embedding sparkline ────────────────────────
    embedding       = mobilenetModel.infer(normBatch, true); // [1,1024]
    const embData   = await embedding.data();
    embedding.dispose();
    drawEmbeddingPanel(embData);

    // ── GradCAM-lite: spatial feature map ───────────────────
    if (spatialModel) {
      spatial           = spatialModel.predict(normBatch);   // [1,7,7,1024]
      const spatData    = await spatial.data();              // Float32 [7*7*1024]
      const [,fH,fW,fC] = spatial.shape;
      spatial.dispose();
      drawGradCAMOverlay(spatData, fH, fW, fC);
    }

    normBatch.dispose();

  } catch(e) {
    [raw,resized,resized255,normalized,batched,normBatch,embedding,spatial]
      .forEach(t => { try { if (t && !t.isDisposed) t.dispose(); } catch(_){} });
    return;
  }

  // ── Panel 5: Softmax bars ───────────────────────────────────
  drawSoftmaxPanel(softmaxProbs);
}

// ── Panel 3 renderer ─────────────────────────────────────────
// Maps normalised value (-1 to +1) to a blue→white→red heatmap.
// Drawn at 28×28 (stride 8) so it's fast — each "pixel" = one 4×4 block
function drawNormPanel(normData) {
  const DISP = insNormCanvas.width;   // 112
  const GRID = 28;                    // sample every 8th pixel from 224×224
  const step = Math.floor(224 / GRID);
  const cell = DISP / GRID;

  insNormCtx.clearRect(0, 0, DISP, DISP);

  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      // pick one pixel from the 224×224 grid
      const srcRow = row * step;
      const srcCol = col * step;
      // Average R,G,B channels for that pixel → single luminance value
      const idx = (srcRow * 224 + srcCol) * 3;
      const val = (normData[idx] + normData[idx+1] + normData[idx+2]) / 3; // -1 to +1

      // Blue (−1) → white (0) → red (+1)
      let r, g, b;
      if (val < 0) {
        // Negative: blue side. t goes 0(val=-1)→1(val=0)
        const t = val + 1;   // 0..1
        r = Math.round(20  + t * 235);
        g = Math.round(60  + t * 195);
        b = Math.round(220 + t * 35);
      } else {
        // Positive: red side. t goes 0(val=0)→1(val=1)
        const t = val;
        r = Math.round(255);
        g = Math.round(255 - t * 200);
        b = Math.round(255 - t * 235);
      }
      insNormCtx.fillStyle = `rgb(${r},${g},${b})`;
      insNormCtx.fillRect(col * cell, row * cell, cell, cell);
    }
  }
}

// ── Panel 4 renderer ─────────────────────────────────────────
// Draws 1024 embedding values as a filled sparkline polyline.
// Key fix: values are always positive (ReLU output from MobileNet),
// so we normalise against max, not assume a -1..+1 range.
function drawEmbeddingPanel(embData) {
  const W = insEmbedCanvas.width;    // 224
  const H = insEmbedCanvas.height;   // 112

  insEmbedCtx.clearRect(0, 0, W, H);

  // Background
  insEmbedCtx.fillStyle = '#f7fafc';
  insEmbedCtx.fillRect(0, 0, W, H);

  // Find true max across all 1024 values
  let maxVal = 0;
  for (let i = 0; i < embData.length; i++) {
    if (embData[i] > maxVal) maxVal = embData[i];
  }
  if (maxVal === 0) {
    // Guard: nothing to draw
    insEmbedCtx.fillStyle = '#a0aec0';
    insEmbedCtx.font = '9px Segoe UI';
    insEmbedCtx.fillText('No activation', 6, H/2);
    return;
  }

  const n    = embData.length;  // 1024
  const padB = 4;               // bottom padding px

  // Draw as filled area polyline for visibility
  insEmbedCtx.beginPath();
  insEmbedCtx.moveTo(0, H - padB);

  for (let i = 0; i < n; i++) {
    const x    = (i / (n - 1)) * W;
    const norm = embData[i] / maxVal;          // 0..1
    const y    = H - padB - norm * (H - padB - 4);
    if (i === 0) insEmbedCtx.lineTo(x, y);
    else         insEmbedCtx.lineTo(x, y);
  }

  insEmbedCtx.lineTo(W, H - padB);
  insEmbedCtx.closePath();

  // Gradient fill — purple at top, light at bottom
  const grad = insEmbedCtx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0,   'rgba(102,126,234,0.85)');
  grad.addColorStop(1,   'rgba(102,126,234,0.08)');
  insEmbedCtx.fillStyle = grad;
  insEmbedCtx.fill();

  // Stroke line on top
  insEmbedCtx.beginPath();
  for (let i = 0; i < n; i++) {
    const x    = (i / (n - 1)) * W;
    const norm = embData[i] / maxVal;
    const y    = H - padB - norm * (H - padB - 4);
    if (i === 0) insEmbedCtx.moveTo(x, y);
    else         insEmbedCtx.lineTo(x, y);
  }
  insEmbedCtx.strokeStyle = '#667eea';
  insEmbedCtx.lineWidth   = 1.2;
  insEmbedCtx.stroke();

  // Find peak index and label it
  let maxIdx = 0;
  for (let i = 0; i < n; i++) { if (embData[i] > embData[maxIdx]) maxIdx = i; }

  const peakX  = (maxIdx / (n - 1)) * W;
  const peakNorm = embData[maxIdx] / maxVal;
  const peakY  = H - padB - peakNorm * (H - padB - 4);

  // Dot at peak
  insEmbedCtx.beginPath();
  insEmbedCtx.arc(peakX, peakY, 3, 0, Math.PI * 2);
  insEmbedCtx.fillStyle = '#f56565';
  insEmbedCtx.fill();

  // Label — flip to left side if near right edge
  insEmbedCtx.fillStyle = '#4a5568';
  insEmbedCtx.font      = 'bold 8px Segoe UI';
  const labelX = peakX > W - 60 ? peakX - 58 : peakX + 4;
  insEmbedCtx.fillText(`peak f${maxIdx} = ${maxVal.toFixed(2)}`, labelX, Math.max(peakY - 3, 10));
}

// ── Panel 5 renderer ─────────────────────────────────────────
// Draws one horizontal bar per class with the softmax probability.
function drawSoftmaxPanel(probs) {
  const W = insSoftmaxCanvas.width;    // 112
  const H = insSoftmaxCanvas.height;   // 112

  insSoftmaxCtx.clearRect(0, 0, W, H);
  insSoftmaxCtx.fillStyle = '#f7fafc';
  insSoftmaxCtx.fillRect(0, 0, W, H);

  const n       = classes.length;
  const padV    = 8;
  const barH    = Math.min(18, (H - padV * (n + 1)) / n);
  const trackW  = W - 8;

  classes.forEach((cls, i) => {
    const prob = probs[i] || 0;
    const y    = padV + i * (barH + padV);

    // Track background
    insSoftmaxCtx.fillStyle = '#e2e8f0';
    insSoftmaxCtx.beginPath();
    insSoftmaxCtx.roundRect(4, y, trackW, barH, 3);
    insSoftmaxCtx.fill();

    // Filled bar
    const fillW = prob * trackW;
    if (fillW > 0) {
      insSoftmaxCtx.fillStyle = cls.pal.color;
      insSoftmaxCtx.globalAlpha = 0.85;
      insSoftmaxCtx.beginPath();
      insSoftmaxCtx.roundRect(4, y, fillW, barH, 3);
      insSoftmaxCtx.fill();
      insSoftmaxCtx.globalAlpha = 1;
    }

    // Label: class name + %
    insSoftmaxCtx.fillStyle = prob > 0.5 ? 'white' : '#4a5568';
    insSoftmaxCtx.font      = `bold ${Math.min(9, barH - 2)}px Segoe UI`;
    insSoftmaxCtx.fillText(
      `${cls.name}  ${(prob*100).toFixed(1)}%`,
      8, y + barH * 0.68
    );
  });
}

// ═══════════════════════════════════════════════════════════════
//  GradCAM-lite: Spatial Attention Overlay on Panel 1
//
//  HOW IT WORKS:
//  MobileNet's last conv layer (conv_pw_13_relu) outputs a
//  [1, 7, 7, 1024] tensor — a 7×7 grid of 1024-channel activations.
//  Each cell in the 7×7 grid corresponds to a ~32×32px region of
//  the original 224×224 image.
//
//  We compute the MEAN across all 1024 channels per spatial cell
//  → produces a 7×7 "importance map"
//  → ReLU to keep only positive activations
//  → normalise 0..1
//  → bilinear upsample to panel size (112×112)
//  → draw as a semi-transparent jet colormap on top of Panel 1
//
//  Result: bright red = regions MobileNet activated most strongly.
// ═══════════════════════════════════════════════════════════════

// Jet colormap: maps 0..1 → [r,g,b] like MATLAB's jet
function jetColor(t) {
  // piecewise linear jet: blue→cyan→green→yellow→red
  const r = Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4*t - 3), 0), 1));
  const g = Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4*t - 2), 0), 1));
  const b = Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4*t - 1), 0), 1));
  return [r, g, b];
}

// Off-screen canvas for GradCAM upsampling
const gradCanvas = document.createElement('canvas');
const gradCtx    = gradCanvas.getContext('2d');

function drawGradCAMOverlay(spatData, fH, fW, fC) {
  // Step 1: channel-mean per spatial cell → [fH, fW]
  const heatmap = new Float32Array(fH * fW);
  for (let row = 0; row < fH; row++) {
    for (let col = 0; col < fW; col++) {
      let sum = 0;
      const base = (row * fW + col) * fC;
      for (let c = 0; c < fC; c++) sum += Math.max(0, spatData[base + c]); // ReLU
      heatmap[row * fW + col] = sum / fC;
    }
  }

  // Step 2: normalise 0..1
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < heatmap.length; i++) {
    if (heatmap[i] < mn) mn = heatmap[i];
    if (heatmap[i] > mx) mx = heatmap[i];
  }
  const range = mx - mn || 1;
  for (let i = 0; i < heatmap.length; i++) heatmap[i] = (heatmap[i] - mn) / range;

  // Step 3: draw low-res heatmap into gradCanvas at fH×fW
  gradCanvas.width  = fW;
  gradCanvas.height = fH;
  const imgData = gradCtx.createImageData(fW, fH);
  for (let i = 0; i < fH * fW; i++) {
    const [r,g,b] = jetColor(heatmap[i]);
    imgData.data[i*4]   = r;
    imgData.data[i*4+1] = g;
    imgData.data[i*4+2] = b;
    imgData.data[i*4+3] = Math.round(heatmap[i] * 180 + 20); // alpha: 20-200
  }
  gradCtx.putImageData(imgData, 0, 0);

  // Step 4: draw Panel 1 raw frame first, then overlay upsampled heatmap
  // (Panel 1 was already drawn — we composite on top)
  insRawCtx.save();
  insRawCtx.globalAlpha       = 0.52;
  insRawCtx.imageSmoothingEnabled = true;
  insRawCtx.imageSmoothingQuality = 'high';
  insRawCtx.drawImage(gradCanvas, 0, 0,
    insRawCanvas.width, insRawCanvas.height);
  insRawCtx.restore();

  // Step 5: legend strip at bottom — "🔴 = AI attention"
  const lW = insRawCanvas.width;
  const lH = 14;
  const lY  = insRawCanvas.height - lH;
  // gradient legend bar
  const lg = insRawCtx.createLinearGradient(0, 0, lW, 0);
  lg.addColorStop(0,   '#0000ff');
  lg.addColorStop(0.25,'#00ffff');
  lg.addColorStop(0.5, '#00ff00');
  lg.addColorStop(0.75,'#ffff00');
  lg.addColorStop(1,   '#ff0000');
  insRawCtx.fillStyle = 'rgba(0,0,0,0.55)';
  insRawCtx.fillRect(0, lY, lW, lH);
  insRawCtx.fillStyle = lg;
  insRawCtx.fillRect(4, lY + 3, lW - 8, lH - 6);
  insRawCtx.fillStyle = 'white';
  insRawCtx.font      = 'bold 7px Segoe UI';
  insRawCtx.fillText('low attention', 5, lY + lH - 3);
  insRawCtx.textAlign = 'right';
  insRawCtx.fillText('high attention', lW - 4, lY + lH - 3);
  insRawCtx.textAlign = 'left';
}

// ═══════════════════════════════════════════════════════════════
//  🎞 EPOCH REPLAY ENGINE
//
//  After training:
//  1. User picks a test image (uploaded image OR webcam snap)
//  2. We extract its 1024D embedding ONCE (MobileNet, frozen)
//  3. Slider scrubs through epochs 1→25
//  4. For each epoch, we RESTORE saved weights into classifier,
//     run a single predict(embedding), show probability bars
//  5. Auto-play animates through all epochs at 400ms/step
//
//  Key insight: MobileNet is frozen, so the same image always
//  produces the same embedding. Only the classifier weights change.
//  This means replay is instant — no image re-processing needed.
// ═══════════════════════════════════════════════════════════════

// DOM refs for replay card
const replayCard       = document.getElementById('replay-card');
const replayThumb      = document.getElementById('replayThumb');
const replaySourceInfo = document.getElementById('replaySourceInfo');
const replayUseUpload  = document.getElementById('replayUseUpload');
const replaySnap       = document.getElementById('replaySnap');
const epochSlider      = document.getElementById('epochSlider');
const epochLabel       = document.getElementById('epochLabel');
const epochEndLabel    = document.getElementById('epochEndLabel');
const replayPlayBtn    = document.getElementById('replayPlayBtn');
const replayStopBtn    = document.getElementById('replayStopBtn');
const replayResetBtn   = document.getElementById('replayResetBtn');
const replayBars       = document.getElementById('replayBars');
const replayInsight    = document.getElementById('replayInsight');

// ── Show replay card and wire up controls ─────────────────────
function initReplayCard() {
  if (!epochSnapshots.length) return;
  replayCard.style.display = 'block';

  // Update slider max to match actual epochs saved
  epochSlider.max   = epochSnapshots.length;
  epochSlider.value = 1;
  epochEndLabel.textContent = `/ ${epochSnapshots.length}`;

  // Enable source buttons
  replayUseUpload.disabled = !(preview.src && preview.naturalWidth > 0);
  replaySnap.disabled      = !webcamReady;

  // Render empty bars for each class
  renderReplayBars();
  replayInsight.textContent = 'Select a test image above to start the replay.';
}

function renderReplayBars() {
  replayBars.innerHTML = '';
  classes.forEach(cls => {
    const d = document.createElement('div');
    d.className = 'rbar-row';
    d.innerHTML = `
      <div class="rbar-hdr">
        <span style="display:flex;align-items:center;gap:6px;">
          <span style="width:8px;height:8px;border-radius:50%;background:${cls.pal.color};display:inline-block;"></span>
          ${cls.name}
        </span>
        <span id="rpct-${cls.id}" style="font-family:monospace;font-size:0.78rem;">—</span>
      </div>
      <div class="rbar-track">
        <div class="rbar-fill" id="rbar-${cls.id}" style="background:${cls.pal.color};"></div>
      </div>`;
    replayBars.appendChild(d);
  });
}

// ── Set test image from uploaded preview ──────────────────────
replayUseUpload.addEventListener('click', async () => {
  if (!preview.src || !preview.naturalWidth) return;
  await setReplaySource(preview, 'uploaded image');
  replayThumb.src = preview.src;
});

// ── Snap webcam frame as test image ──────────────────────────
replaySnap.addEventListener('click', async () => {
  if (!webcamReady || !webcamEl.videoWidth) return;
  // Draw webcam frame to a temp canvas, use as source
  const snap = document.createElement('canvas');
  snap.width = webcamEl.videoWidth; snap.height = webcamEl.videoHeight;
  snap.getContext('2d').drawImage(webcamEl, 0, 0);
  replayThumb.src = snap.toDataURL('image/jpeg', 0.85);
  await setReplaySource(snap, 'webcam snapshot');
});

async function setReplaySource(src, label) {
  if (!mobilenetModel) return;
  // Extract embedding once — reused for all 25 epoch replays
  const t = tf.tidy(() => mobilenetModel.infer(src, true));  // [1,1024]
  replayTestEmb = await t.data();   // Float32Array
  t.dispose();

  replayThumb.style.display = 'block';
  replaySourceInfo.innerHTML = `Test image: <b>${label}</b>. Drag the slider to scrub through epochs.`;

  // Enable all controls
  epochSlider.disabled  = false;
  replayPlayBtn.disabled = false;
  replayResetBtn.disabled = false;

  // Jump to epoch 1 immediately
  epochSlider.value = 1;
  await scrubToEpoch(1);
}

// ── Core: restore weights for epoch N and predict ─────────────
async function scrubToEpoch(epochNum) {
  if (!replayTestEmb || !epochSnapshots.length) return;

  const snapIdx = Math.min(epochNum - 1, epochSnapshots.length - 1);
  const snap    = epochSnapshots[snapIdx];
  if (!snap) return;

  // Restore saved weights into classifier
  const tensors = snap.weights.map(w => tf.tensor(w.data, w.shape));
  classifier.setWeights(tensors);
  tensors.forEach(t => t.dispose());

  // Run prediction on the saved embedding
  const embTensor  = tf.tensor2d([Array.from(replayTestEmb)], [1, 1024]);
  const predTensor = classifier.predict(embTensor);
  const probs      = await predTensor.data();
  embTensor.dispose(); predTensor.dispose();

  // Update slider label + fill gradient
  epochLabel.textContent = `Epoch ${snap.epoch}`;
  const pct = ((snap.epoch / epochSnapshots.length) * 100).toFixed(0);
  epochSlider.style.setProperty('--pct', pct + '%');

  // Update bars
  classes.forEach((cls, i) => {
    const p = (probs[i] * 100).toFixed(1);
    const pe = document.getElementById(`rpct-${cls.id}`);
    const be = document.getElementById(`rbar-${cls.id}`);
    if (pe) pe.textContent = p + '%';
    if (be) be.style.width  = p + '%';
  });

  // Insight sentence
  const maxI      = Array.from(probs).indexOf(Math.max(...probs));
  const winner    = classes[maxI];
  const conf      = (probs[maxI] * 100).toFixed(1);
  const isEarly   = snap.epoch <= 5;
  const isMiddle  = snap.epoch <= 15;
  const accPct    = (snap.acc * 100).toFixed(1);
  const lossFmt   = snap.loss.toFixed(4);

  let insight = '';
  if (isEarly) {
    insight = `⚡ <b>Epoch ${snap.epoch} — Early training.</b> Loss is ${lossFmt}, accuracy ${accPct}%. Weights are still near random — predictions are mostly guesses.`;
  } else if (isMiddle) {
    insight = `📈 <b>Epoch ${snap.epoch} — Learning.</b> Loss dropping to ${lossFmt}, accuracy ${accPct}%. The model is starting to separate the classes.`;
  } else {
    const converged = snap.acc > 0.9;
    insight = converged
      ? `✅ <b>Epoch ${snap.epoch} — Converged.</b> Loss ${lossFmt}, accuracy ${accPct}%. The model confidently predicts <b>${winner?.name}</b> at ${conf}%.`
      : `🔄 <b>Epoch ${snap.epoch} — Still learning.</b> Loss ${lossFmt}, accuracy ${accPct}%. Not fully converged yet — more samples may help.`;
  }
  replayInsight.innerHTML = insight;

  // Restore final trained weights after scrub so live prediction still works
  // (will be re-applied on next scrub, so no need to re-restore here)
}

// ── Slider input handler ──────────────────────────────────────
epochSlider.addEventListener('input', async () => {
  stopReplayAuto();
  await scrubToEpoch(parseInt(epochSlider.value));
});

// ── Auto-play ─────────────────────────────────────────────────
replayPlayBtn.addEventListener('click', () => {
  if (!replayTestEmb) return;
  stopReplayAuto();
  let e = parseInt(epochSlider.value);
  if (e >= epochSnapshots.length) e = 1;   // wrap around

  replayPlayBtn.disabled = true;
  replayStopBtn.disabled = false;

  replayInterval = setInterval(async () => {
    epochSlider.value = e;
    await scrubToEpoch(e);
    e++;
    if (e > epochSnapshots.length) {
      stopReplayAuto();
      // After auto-play ends, restore final model weights
      restoreFinalWeights();
    }
  }, 420);  // 420ms per epoch — smooth but watchable
});

replayStopBtn.addEventListener('click', () => {
  stopReplayAuto();
  restoreFinalWeights();
});

replayResetBtn.addEventListener('click', async () => {
  stopReplayAuto();
  epochSlider.value = 1;
  await scrubToEpoch(1);
});

function stopReplayAuto() {
  if (replayInterval) { clearInterval(replayInterval); replayInterval = null; }
  replayPlayBtn.disabled = !replayTestEmb;
  replayStopBtn.disabled = true;
}

// Restore the final epoch's weights so live prediction still works after scrubbing
function restoreFinalWeights() {
  if (!epochSnapshots.length) return;
  const last    = epochSnapshots[epochSnapshots.length - 1];
  const tensors = last.weights.map(w => tf.tensor(w.data, w.shape));
  classifier.setWeights(tensors);
  tensors.forEach(t => t.dispose());
}

// Keep source buttons in sync with page state
// (called from readFile and webcam start)
function syncReplayButtons() {
  if (replayUseUpload)
    replayUseUpload.disabled = !(preview.src && preview.naturalWidth > 0);
  if (replaySnap)
    replaySnap.disabled = !webcamReady;
}