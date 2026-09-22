/* app.js — easy-wakeword-trainer frontend
 *
 * No framework. Relative API URLs so Electron can load http://127.0.0.1:8000 later.
 * No LLM or cloud-AI calls — all communication is to the local FastAPI server.
 *
 * Flow:
 *   1. On load → GET /api/health
 *      - If data ready  → show form, enable Train button
 *      - If data missing → show prepare section with "Download Data" button
 *   2. "Download Data" → POST /api/prepare → SSE /api/prepare/events
 *      - Streams live download log
 *      - On "done" → re-check health → show form
 *   3. "Train" → POST /api/train → SSE /api/train/{id}/events
 *      - Streams live training log
 *      - On "done" → show download buttons
 */

const STAGE_ORDER = ["generate", "augment", "train", "export"];

// ── DOM refs ──────────────────────────────────────────────────────────────
const phraseInput     = document.getElementById("phrase-input");
const trainBtn        = document.getElementById("train-btn");
const prepareSection  = document.getElementById("prepare-section");
const prepareTitle    = document.getElementById("prepare-title");
const prepareSubtitle = document.getElementById("prepare-subtitle");
const prepareBtn      = document.getElementById("prepare-btn");
const missingList     = document.getElementById("missing-list");
const prepareLog      = document.getElementById("prepare-log");
const progressSection = document.getElementById("progress-section");
const progressLabel   = document.getElementById("progress-label");
const logPanel        = document.getElementById("log-panel");
const downloadSection = document.getElementById("download-section");
const errorSection    = document.getElementById("error-section");
const errorMsg        = document.getElementById("error-msg");
const modelNameOut    = document.getElementById("model-name-out");
const dlZip           = document.getElementById("dl-zip");
const dlOnnx          = document.getElementById("dl-onnx");
const dlTflite        = document.getElementById("dl-tflite");

const stagePills = {
  generate: document.getElementById("stage-generate"),
  augment:  document.getElementById("stage-augment"),
  train:    document.getElementById("stage-train"),
  export:   document.getElementById("stage-export"),
};

// ── State ─────────────────────────────────────────────────────────────────
let dataReady      = false;
let jobRunning     = false;
let prepareRunning = false;

// ── Init ──────────────────────────────────────────────────────────────────
checkHealth();
phraseInput.addEventListener("input", updateTrainBtn);
trainBtn.addEventListener("click", () => startTraining(phraseInput.value.trim()));
phraseInput.addEventListener("keydown", (e) => { if (e.key === "Enter") trainBtn.click(); });
prepareBtn.addEventListener("click", startPrepare);

// ── Health check ──────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch("/api/health");
    const data = await res.json();
    dataReady = data.ready;

    if (!data.ready) {
      showPrepareSection(data.missing, data.assets);
    } else {
      hidePrepareSection();
    }
  } catch {
    dataReady = false;
    showPrepareSection(["Could not reach the server — is Docker running?"], {});
  }
  updateTrainBtn();
}

// ── Prepare section helpers ───────────────────────────────────────────────
function showPrepareSection(missing, assets) {
  prepareSection.classList.remove("hidden");
  missingList.innerHTML = "";
  missing.forEach((name) => {
    const li = document.createElement("li");
    const desc = assets[name] ? ` — ${assets[name].description}` : "";
    li.textContent = name + desc;
    missingList.appendChild(li);
  });
}

function hidePrepareSection() {
  prepareSection.classList.add("hidden");
}

function setPrepareState(running) {
  prepareRunning = running;
  prepareBtn.disabled = running;
  prepareBtn.textContent = running ? "Downloading…" : "Download Data";
}

// ── First-run data download ───────────────────────────────────────────────
async function startPrepare() {
  setPrepareState(true);
  missingList.innerHTML = "";
  prepareLog.textContent = "";
  prepareLog.classList.remove("hidden");
  prepareTitle.textContent = "Downloading training data…";
  prepareSubtitle.textContent =
    "This is a one-time download (~5–20 GB). Do not close the browser.";

  let jobId;
  try {
    const res  = await fetch("/api/prepare", { method: "POST" });
    const data = await res.json();

    if (data.status === "already_ready") {
      onPrepareComplete();
      return;
    }
    jobId = data.job_id;
  } catch (e) {
    appendPrepareLog("ERROR: " + e.message);
    setPrepareState(false);
    return;
  }

  connectPrepareSSE(jobId);
}

function connectPrepareSSE(jobId) {
  const es = new EventSource("/api/prepare/events");

  es.addEventListener("running", (e) => appendPrepareLog(e.data));
  es.addEventListener("log",     (e) => appendPrepareLog(e.data));

  es.addEventListener("done", () => {
    appendPrepareLog("✓ All assets ready.");
    es.close();
    onPrepareComplete();
  });

  // Server sent an explicit "error" event (training pipeline error)
  es.addEventListener("error", (e) => {
    if (e.data) {
      appendPrepareLog("ERROR: " + e.data);
      setPrepareState(false);
      es.close();
    }
    // if e.data is empty this is an SSE transport error, handled by es.onerror below
  });

  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) return;
    appendPrepareLog("[SSE connection dropped — switching to polling…]");
    es.close();
    pollPrepare();
  };
}

async function pollPrepare(intervalMs = 3000) {
  // Track how many lines we've already shown so we never re-append history.
  // Count current lines in the panel from before SSE dropped.
  let shownLines = prepareLog.textContent
    ? prepareLog.textContent.split("\n").length
    : 0;

  while (true) {
    await new Promise((r) => setTimeout(r, intervalMs));
    try {
      const res  = await fetch("/api/prepare/status");
      const data = await res.json();
      const lines = data.log_lines || [];
      // Only append lines we haven't shown yet
      lines.slice(shownLines).forEach(appendPrepareLog);
      shownLines = lines.length;
      if (data.stage === "done")  { onPrepareComplete(); return; }
      if (data.stage === "error") {
        appendPrepareLog("ERROR: " + (data.error || "Download failed."));
        setPrepareState(false);
        return;
      }
    } catch { /* ignore blips */ }
  }
}

function onPrepareComplete() {
  setPrepareState(false);
  prepareTitle.textContent  = "✓ Training data ready";
  prepareSubtitle.textContent = "";
  prepareBtn.classList.add("hidden");
  dataReady = true;
  updateTrainBtn();
  // Re-verify with the server
  checkHealth();
}

function appendPrepareLog(line) {
  if (!line) return;
  prepareLog.textContent += (prepareLog.textContent ? "\n" : "") + line;
  prepareLog.scrollTop = prepareLog.scrollHeight;
}

// ── Button state ──────────────────────────────────────────────────────────
function updateTrainBtn() {
  const hasText = phraseInput.value.trim().length > 0;
  trainBtn.disabled = !hasText || !dataReady || jobRunning;
}

// ── Phrase validation (mirrors server rules) ──────────────────────────────
function clientValidate(phrase) {
  if (!phrase) return "Phrase must not be empty.";
  if (!/^[a-z0-9 ]+$/.test(phrase.toLowerCase()))
    return "Phrase may only contain letters, numbers, and spaces.";
  if (phrase.trim().split(/\s+/).length > 6)
    return "Phrase must be at most 6 words.";
  return null;
}

// ── Training ──────────────────────────────────────────────────────────────
async function startTraining(phrase) {
  const err = clientValidate(phrase);
  if (err) { showError(err); return; }

  resetTrainUI();
  setJobRunning(true);

  let jobId;
  try {
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phrase }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    jobId = data.job_id;
    progressLabel.textContent = `Training "${data.model_name}"…`;
    modelNameOut.textContent  = data.model_name;
  } catch (e) {
    showError(e.message);
    setJobRunning(false);
    return;
  }

  progressSection.classList.remove("hidden");
  connectTrainSSE(jobId);
}

function connectTrainSSE(jobId) {
  const es = new EventSource(`/api/train/${jobId}/events`);

  es.addEventListener("log",  (e) => appendTrainLog(e.data));
  es.addEventListener("info", (e) => appendTrainLog(e.data));

  STAGE_ORDER.forEach((stage) => {
    es.addEventListener(stage, (e) => {
      activateStage(stage);
      appendTrainLog(e.data);
    });
  });

  es.addEventListener("done", () => {
    markAllStagesDone();
    progressLabel.textContent = "Done!";
    showDownloads(jobId);
    setJobRunning(false);
    es.close();
  });

  es.addEventListener("error", (e) => {
    showError(e.data || "Training failed. See log for details.");
    setJobRunning(false);
    es.close();
  });

  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) return;
    appendTrainLog("[SSE dropped — polling status…]");
    es.close();
    pollTrain(jobId);
  };
}

async function pollTrain(jobId, intervalMs = 3000) {
  while (true) {
    await new Promise((r) => setTimeout(r, intervalMs));
    try {
      const res  = await fetch(`/api/train/${jobId}/status`);
      const data = await res.json();
      progressLabel.textContent = `Stage: ${data.stage}`;
      if (data.stage === "done")  { markAllStagesDone(); showDownloads(jobId); setJobRunning(false); return; }
      if (data.stage === "error") { showError(data.error || "Training failed."); setJobRunning(false); return; }
      if (STAGE_ORDER.includes(data.stage)) activateStage(data.stage);
    } catch { /* ignore */ }
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────
function appendTrainLog(line) {
  logPanel.textContent += (logPanel.textContent ? "\n" : "") + line;
  logPanel.scrollTop = logPanel.scrollHeight;
}

function activateStage(stage) {
  const idx = STAGE_ORDER.indexOf(stage);
  STAGE_ORDER.forEach((s, i) => {
    const pill = stagePills[s];
    if (!pill) return;
    pill.classList.remove("active", "done");
    if (i < idx)      pill.classList.add("done");
    else if (i === idx) pill.classList.add("active");
  });
}

function markAllStagesDone() {
  STAGE_ORDER.forEach((s) => {
    const pill = stagePills[s];
    if (pill) { pill.classList.remove("active"); pill.classList.add("done"); }
  });
}

function showDownloads(jobId) {
  dlZip.href    = `/api/train/${jobId}/download`;
  dlOnnx.href   = `/api/train/${jobId}/download/onnx`;
  dlTflite.href = `/api/train/${jobId}/download/tflite`;
  downloadSection.classList.remove("hidden");
  // Refresh the tester model list so the new model appears immediately
  document.dispatchEvent(new Event("trainingDone"));
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorSection.classList.remove("hidden");
}

function resetTrainUI() {
  progressSection.classList.add("hidden");
  downloadSection.classList.add("hidden");
  errorSection.classList.add("hidden");
  logPanel.textContent      = "";
  progressLabel.textContent = "Starting…";
  STAGE_ORDER.forEach((s) => {
    const pill = stagePills[s];
    if (pill) pill.classList.remove("active", "done");
  });
}

function setJobRunning(running) {
  jobRunning = running;
  phraseInput.disabled = running;
  updateTrainBtn();
}

// =============================================================================
// TESTER — browse models, stream mic audio, show live detection
// =============================================================================

const modelSelect  = document.getElementById("model-select");
const micBtn       = document.getElementById("mic-btn");
const detectorRing = document.getElementById("detector-ring");
const detectorLbl  = document.getElementById("detector-label");
const scoreBar     = document.getElementById("score-bar");
const scoreVal     = document.getElementById("score-val");

let testerWs       = null;
let audioCtx       = null;
let mediaStream    = null;
let scriptNode     = null;
let detectionTimer = null;

// Populate model dropdown on load
async function loadModels() {
  try {
    const res = await fetch("/api/models");
    const { models } = await res.json();
    modelSelect.innerHTML = '<option value="">— select a model —</option>';
    models.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = `${m.name}  (${m.size_kb} KB)`;
      modelSelect.appendChild(opt);
    });
    micBtn.disabled = models.length === 0;
  } catch (e) {
    console.warn("Could not load models:", e);
  }
}
loadModels();
// Refresh model list after a training job completes
document.addEventListener("trainingDone", loadModels);

modelSelect.addEventListener("change", () => {
  if (testerWs) stopListening();
  micBtn.disabled = !modelSelect.value;
  micBtn.textContent = "🎤 Start Listening";
  micBtn.classList.remove("listening");
  resetDetector();
});

micBtn.addEventListener("click", () => {
  if (testerWs) {
    stopListening();
  } else {
    startListening();
  }
});

async function startListening() {
  const modelName = modelSelect.value;
  if (!modelName) return;

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true }
    });
  } catch (e) {
    alert("Microphone access denied: " + e.message);
    return;
  }

  const proto = location.protocol === "https:" ? "wss" : "ws";
  testerWs = new WebSocket(`${proto}://${location.host}/api/test/${encodeURIComponent(modelName)}`);
  testerWs.binaryType = "arraybuffer";

  testerWs.onopen = () => {
    micBtn.textContent = "⏹ Stop Listening";
    micBtn.classList.add("listening");
    detectorLbl.textContent = "Listening…";
    startMicCapture();
  };

  testerWs.onmessage = (evt) => {
    const { score, detected } = JSON.parse(evt.data);
    updateDetector(score, detected);
  };

  testerWs.onerror = (e) => console.error("Tester WS error", e);
  testerWs.onclose = () => stopListening();
}

function startMicCapture() {
  audioCtx = new AudioContext({ sampleRate: 16000 });
  const source = audioCtx.createMediaStreamSource(mediaStream);

  // ScriptProcessorNode gives us raw PCM floats; convert to s16le for OWW
  scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
  scriptNode.onaudioprocess = (e) => {
    if (!testerWs || testerWs.readyState !== WebSocket.OPEN) return;
    const floats = e.inputBuffer.getChannelData(0);
    const s16 = new Int16Array(floats.length);
    for (let i = 0; i < floats.length; i++) {
      s16[i] = Math.max(-32768, Math.min(32767, floats[i] * 32768));
    }
    testerWs.send(s16.buffer);
  };

  source.connect(scriptNode);
  scriptNode.connect(audioCtx.destination);
}

function stopListening() {
  if (scriptNode)   { scriptNode.disconnect(); scriptNode = null; }
  if (audioCtx)     { audioCtx.close();        audioCtx = null; }
  if (mediaStream)  { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (testerWs && testerWs.readyState < 2) testerWs.close();
  testerWs = null;

  micBtn.textContent = "🎤 Start Listening";
  micBtn.classList.remove("listening");
  detectorLbl.textContent = "—";
  resetDetector();
}

function updateDetector(score, detected) {
  const pct = Math.round(score * 100);
  scoreBar.style.width = pct + "%";
  scoreVal.textContent = `score: ${score.toFixed(3)}`;

  if (detected) {
    detectorRing.classList.add("active");
    scoreBar.classList.add("hot");
    detectorLbl.textContent = "✅ Detected!";

    clearTimeout(detectionTimer);
    detectionTimer = setTimeout(() => {
      detectorRing.classList.remove("active");
      scoreBar.classList.remove("hot");
      detectorLbl.textContent = "Listening…";
    }, 1500);
  } else {
    if (!detectorRing.classList.contains("active")) {
      scoreBar.classList.remove("hot");
    }
  }
}

function resetDetector() {
  clearTimeout(detectionTimer);
  detectorRing.classList.remove("active");
  scoreBar.classList.remove("hot");
  scoreBar.style.width = "0%";
  scoreVal.textContent = "";
}
