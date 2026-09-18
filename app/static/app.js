/* app.js — easy-wakeword-trainer frontend
 *
 * No framework. Relative API URLs so Electron can load http://127.0.0.1:8000 later.
 * No LLM or cloud-AI calls — all communication is to the local FastAPI server.
 */

const STAGE_ORDER = ["generate", "augment", "train", "export"];

// ── DOM refs ──────────────────────────────────────────────────────────────
const phraseInput     = document.getElementById("phrase-input");
const trainBtn        = document.getElementById("train-btn");
const healthBanner    = document.getElementById("health-banner");
const healthDetails   = document.getElementById("health-details");
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

// Stage pill elements
const stagePills = {
  generate: document.getElementById("stage-generate"),
  augment:  document.getElementById("stage-augment"),
  train:    document.getElementById("stage-train"),
  export:   document.getElementById("stage-export"),
};

// ── State ─────────────────────────────────────────────────────────────────
let dataReady = false;
let jobRunning = false;
let currentJobId = null;
let currentEventSource = null;

// ── Init ──────────────────────────────────────────────────────────────────
checkHealth();
phraseInput.addEventListener("input", updateTrainBtn);

trainBtn.addEventListener("click", async () => {
  const phrase = phraseInput.value.trim();
  if (!phrase) return;
  await startTraining(phrase);
});

phraseInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") trainBtn.click();
});

// ── Health check ──────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    dataReady = data.ready;
    if (!data.ready) {
      healthDetails.textContent =
        " Missing: " + data.missing.join(", ") + ".";
      healthBanner.classList.remove("hidden");
    } else {
      healthBanner.classList.add("hidden");
    }
  } catch {
    dataReady = false;
    healthDetails.textContent = " Could not reach the server.";
    healthBanner.classList.remove("hidden");
  }
  updateTrainBtn();
}

// ── Button state ──────────────────────────────────────────────────────────
function updateTrainBtn() {
  const hasText = phraseInput.value.trim().length > 0;
  trainBtn.disabled = !hasText || !dataReady || jobRunning;
}

// ── Validation helper (mirrors server rules) ──────────────────────────────
function clientValidate(phrase) {
  if (!phrase) return "Phrase must not be empty.";
  if (!/^[a-z0-9 ]+$/.test(phrase.toLowerCase())) {
    return "Phrase may only contain letters, numbers, and spaces.";
  }
  if (phrase.trim().split(/\s+/).length > 6) {
    return "Phrase must be at most 6 words.";
  }
  return null;
}

// ── Start training ────────────────────────────────────────────────────────
async function startTraining(phrase) {
  const err = clientValidate(phrase);
  if (err) { showError(err); return; }

  resetUI();
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
    currentJobId = jobId;
    progressLabel.textContent = `Training "${data.model_name}"…`;
    modelNameOut.textContent = data.model_name;
  } catch (e) {
    showError(e.message);
    setJobRunning(false);
    return;
  }

  progressSection.classList.remove("hidden");
  connectSSE(jobId);
}

// ── SSE log stream ────────────────────────────────────────────────────────
function connectSSE(jobId) {
  if (currentEventSource) currentEventSource.close();

  const es = new EventSource(`/api/train/${jobId}/events`);
  currentEventSource = es;

  // Generic log lines
  es.addEventListener("log", (e) => appendLog(e.data));

  // Stage transitions
  STAGE_ORDER.forEach((stage) => {
    es.addEventListener(stage, (e) => {
      activateStage(stage);
      appendLog(e.data);
    });
  });

  // Also catch stage name as event from info messages
  es.addEventListener("info", (e) => appendLog(e.data));

  es.addEventListener("done", (e) => {
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

  es.onerror = (e) => {
    // SSE connection error (not a training error)
    if (es.readyState === EventSource.CLOSED) return;
    appendLog("[SSE connection lost, polling status…]");
    es.close();
    pollStatus(jobId);
  };
}

// ── Fallback polling (if SSE drops) ──────────────────────────────────────
async function pollStatus(jobId, intervalMs = 3000) {
  while (true) {
    await new Promise((r) => setTimeout(r, intervalMs));
    try {
      const res = await fetch(`/api/train/${jobId}/status`);
      const data = await res.json();
      progressLabel.textContent = `Stage: ${data.stage}`;
      if (data.stage === "done") {
        markAllStagesDone();
        showDownloads(jobId);
        setJobRunning(false);
        return;
      }
      if (data.stage === "error") {
        showError(data.error || "Training failed.");
        setJobRunning(false);
        return;
      }
      if (STAGE_ORDER.includes(data.stage)) activateStage(data.stage);
    } catch {
      // ignore network blips
    }
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────
function appendLog(line) {
  logPanel.textContent += (logPanel.textContent ? "\n" : "") + line;
  logPanel.scrollTop = logPanel.scrollHeight;
}

function activateStage(stage) {
  const idx = STAGE_ORDER.indexOf(stage);
  STAGE_ORDER.forEach((s, i) => {
    const pill = stagePills[s];
    if (!pill) return;
    pill.classList.remove("active", "done");
    if (i < idx) pill.classList.add("done");
    else if (i === idx) pill.classList.add("active");
  });
}

function markAllStagesDone() {
  STAGE_ORDER.forEach((s) => {
    const pill = stagePills[s];
    if (!pill) return;
    pill.classList.remove("active");
    pill.classList.add("done");
  });
}

function showDownloads(jobId) {
  dlZip.href    = `/api/train/${jobId}/download`;
  dlOnnx.href   = `/api/train/${jobId}/download/onnx`;
  dlTflite.href = `/api/train/${jobId}/download/tflite`;
  downloadSection.classList.remove("hidden");
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorSection.classList.remove("hidden");
}

function resetUI() {
  progressSection.classList.add("hidden");
  downloadSection.classList.add("hidden");
  errorSection.classList.add("hidden");
  logPanel.textContent = "";
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
