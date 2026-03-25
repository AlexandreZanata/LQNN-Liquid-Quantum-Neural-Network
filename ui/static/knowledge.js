/**
 * LQNN Knowledge Base UI
 * Handles file uploads, URL ingestion, real-time progress, system metrics,
 * and knowledge library with absorption percentages.
 */

const WS_URL = `ws://${location.host}/ws`;
let socket = null;
let reconnectTimer = null;

const stats = { files: 0, urls: 0, chunks: 0, images: 0, concepts: 0, rejected: 0, ingested: 0 };

let uploadQueue = [];
let isIngesting = false;
let libraryItems = [];

// ------------------------------------------------------------------ //
// WebSocket                                                           //
// ------------------------------------------------------------------ //

function connect() {
  socket = new WebSocket(WS_URL);
  socket.onopen = () => {
    setStatus(true);
    clearTimeout(reconnectTimer);
    logEntry("sys", "Connected to quantum brain");
    socket.send(JSON.stringify({ action: "get_state" }));
  };
  socket.onclose = () => {
    setStatus(false);
    reconnectTimer = setTimeout(connect, 3000);
  };
  socket.onerror = () => socket.close();
  socket.onmessage = (ev) => {
    try { handleMessage(JSON.parse(ev.data)); }
    catch (e) { console.error("WS parse error:", e); }
  };
}

function sendWS(action, payload = {}) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ action, ...payload }));
  }
}

function setStatus(online) {
  const el = document.getElementById("status-indicator");
  el.textContent = online ? "ONLINE" : "OFFLINE";
  el.className = "status-pill " + (online ? "online" : "offline");
}

// ------------------------------------------------------------------ //
// Message handling                                                    //
// ------------------------------------------------------------------ //

function handleMessage(data) {
  if (data.type === "state") {
    const mem = data.memory || {};
    setText("kb-concepts", `CONCEPTS: ${mem.concepts ?? "--"}`);
    updateKBConcepts(data.recent_concepts || []);
    if (data.system) updateSystemMetrics(data.system);
  } else if (data.type === "live_event") {
    handleLiveEvent(data);
  } else if (data.type === "kb_ingest_result") {
    handleIngestResult(data);
  } else if (data.type === "error") {
    logEntry("error", data.message || "unknown error");
  }
}

function handleLiveEvent(ev) {
  switch (ev.type) {
    case "kb_progress":
      logEntry("progress",
        `[${ev.source_type?.toUpperCase()}] ${ev.source} — ${ev.stage} ${ev.percent}%` +
        (ev.total_chunks ? ` (${ev.total_chunks} chunks)` : "") +
        (ev.stored !== undefined ? ` stored=${ev.stored} rejected=${ev.rejected}` : "")
      );
      updateQueueItemProgress(ev.source, ev.percent);
      break;

    case "kb_chunk_stored":
      logEntry("learn", `chunk stored: "${ev.concept}" [${ev.source}]`);
      break;

    case "kb_image_stored":
      logEntry("image", `image stored: "${ev.concept}" +${ev.associations} assocs`);
      break;

    case "kb_done": {
      const icon = ev.success ? "\u2713" : "\u2717";
      const cls = ev.success ? "learn" : "error";
      logEntry(cls,
        `${icon} ${ev.source} — chunks=${ev.chunks_stored} images=${ev.images_stored} ` +
        `concepts=${ev.concepts_created} (${ev.duration_s}s)` +
        (ev.error ? ` ERR: ${ev.error}` : "")
      );
      if (!ev.error) {
        stats.chunks += ev.chunks_stored || 0;
        stats.images += ev.images_stored || 0;
        stats.concepts += ev.concepts_created || 0;
        stats.rejected += ev.chunks_rejected || 0;
        if (ev.success) stats.ingested++;
        updateStats();
      }
      addLibraryItem({
        source: ev.source,
        source_type: ev.source_type || "file",
        chunks_stored: ev.chunks_stored || 0,
        chunks_total: ev.chunks_total || ev.chunks_stored || 0,
        images_stored: ev.images_stored || 0,
        concepts_created: ev.concepts_created || 0,
        success: ev.success,
        duration_s: ev.duration_s || 0,
      });
      markQueueItemDone(ev.source, ev.success, ev.error);
      break;
    }
  }
}

function handleIngestResult(data) {
  const icon = data.success ? "\u2713" : "\u2717";
  const cls = data.success ? "learn" : "error";
  logEntry(cls,
    `${icon} URL: ${data.source} — chunks=${data.chunks_stored} ` +
    `images=${data.images_stored || 0} concepts=${data.concepts_created} (${data.duration_s}s)` +
    (data.error ? ` ERR: ${data.error}` : "")
  );
  if (!data.error) {
    stats.urls++;
    stats.chunks += data.chunks_stored || 0;
    stats.images += data.images_stored || 0;
    stats.concepts += data.concepts_created || 0;
    stats.rejected += data.chunks_rejected || 0;
    if (data.success) stats.ingested++;
    updateStats();
  }
  addLibraryItem({
    source: data.source,
    source_type: data.source_type || "url",
    chunks_stored: data.chunks_stored || 0,
    chunks_total: data.chunks_total || data.chunks_stored || 0,
    images_stored: data.images_stored || 0,
    concepts_created: data.concepts_created || 0,
    success: data.success,
    duration_s: data.duration_s || 0,
  });
}

// ------------------------------------------------------------------ //
// System Metrics                                                      //
// ------------------------------------------------------------------ //

function updateSystemMetrics(sys) {
  const cpu = document.getElementById("sys-cpu");
  const ram = document.getElementById("sys-ram");
  const gpu = document.getElementById("sys-gpu");
  if (cpu) cpu.textContent = `CPU: ${sys.cpu_percent ?? 0}%`;
  if (ram) {
    const used = sys.ram_used_gb ?? 0;
    const total = sys.ram_total_gb ?? 0;
    ram.textContent = `RAM: ${used}/${total} GB`;
  }
  if (gpu) {
    if (sys.gpu_metric_source === "no_cuda") {
      gpu.textContent = "GPU: unavailable";
    } else {
      const gu = sys.gpu_used_gb ?? 0;
      const gt = sys.gpu_total_gb ?? 0;
      gpu.textContent = `GPU: ${gu}/${gt} GB`;
    }
  }
}

// ------------------------------------------------------------------ //
// File Upload Queue                                                   //
// ------------------------------------------------------------------ //

function addFilesToQueue(files) {
  for (const file of files) {
    const ext = file.name.split(".").pop().toLowerCase();
    const typeLabel = getTypeLabel(ext);
    uploadQueue.push({
      id: `q_${Date.now()}_${Math.random().toString(36).slice(2)}`,
      name: file.name,
      size: file.size,
      ext,
      typeLabel,
      file,
      status: "pending",
      progress: 0,
    });
  }
  renderQueue();
}

function getTypeLabel(ext) {
  if (ext === "pdf") return "pdf";
  if (["jpg", "jpeg", "png", "webp", "gif", "bmp"].includes(ext)) return "img";
  if (["docx", "doc"].includes(ext)) return "doc";
  if (["html", "htm"].includes(ext)) return "html";
  if (ext === "json") return "json";
  return "txt";
}

function renderQueue() {
  const container = document.getElementById("upload-queue");
  container.innerHTML = "";
  setText("kb-queue", `QUEUE: ${uploadQueue.length}`);

  for (const item of uploadQueue) {
    const div = document.createElement("div");
    div.className = "queue-item";
    div.id = `qi_${item.id}`;
    div.innerHTML = `
      <span class="queue-item-type ${item.typeLabel}">${item.typeLabel.toUpperCase()}</span>
      <span class="queue-item-name" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span>
      <span class="queue-item-size">${formatSize(item.size)}</span>
      <span class="queue-item-status ${item.status}" id="qs_${item.id}">${item.status.toUpperCase()}</span>
    `;
    container.appendChild(div);
  }
}

function updateQueueItemProgress(sourceName, pct) {
  const item = uploadQueue.find(i => i.name === sourceName || sourceName.includes(i.name));
  if (!item) return;
  const el = document.getElementById(`qs_${item.id}`);
  if (el) {
    el.textContent = `${pct}%`;
    el.className = `queue-item-status running`;
  }
}

function markQueueItemDone(sourceName, success, error) {
  const item = uploadQueue.find(i => i.name === sourceName || sourceName.includes(i.name));
  if (!item) return;
  item.status = success ? "done" : "error";
  const el = document.getElementById(`qs_${item.id}`);
  if (el) {
    el.textContent = success ? "DONE" : "ERROR";
    el.className = `queue-item-status ${item.status}`;
  }
}

async function ingestAll() {
  if (isIngesting) return;
  const pending = uploadQueue.filter(i => i.status === "pending");
  if (!pending.length) {
    logEntry("sys", "No pending files in queue");
    return;
  }

  isIngesting = true;
  logEntry("sys", `Starting ingestion of ${pending.length} file(s)...`);

  const tags = document.getElementById("upload-tags").value.trim();
  const conceptHint = document.getElementById("upload-concept").value.trim();

  for (const item of pending) {
    item.status = "running";
    renderQueue();

    const formData = new FormData();
    formData.append("file", item.file);
    formData.append("tags", tags);
    formData.append("concept_hint", conceptHint);

    try {
      const resp = await fetch("/api/knowledge/upload", {
        method: "POST",
        body: formData,
      });
      const result = await resp.json();
      item.status = result.success ? "done" : "error";
      stats.files++;
      stats.chunks += result.chunks_stored || 0;
      stats.images += result.images_stored || 0;
      stats.concepts += result.concepts_created || 0;
      stats.rejected += result.chunks_rejected || 0;
      if (result.success) stats.ingested++;
      updateStats();
    } catch (e) {
      item.status = "error";
      logEntry("error", `Upload failed: ${item.name} — ${e.message}`);
    }
    renderQueue();
  }

  isIngesting = false;
  logEntry("sys", "Ingestion batch complete");
}

// ------------------------------------------------------------------ //
// URL Ingestion                                                       //
// ------------------------------------------------------------------ //

function ingestUrl() {
  const url = document.getElementById("url-input").value.trim();
  if (!url) { logEntry("error", "URL is required"); return; }

  const tagsRaw = document.getElementById("url-tags").value.trim();
  const tags = tagsRaw ? tagsRaw.split(",").map(t => t.trim()).filter(Boolean) : [];

  logEntry("url", `Fetching: ${url}`);
  sendWS("kb_ingest_url", { url, tags });
  document.getElementById("url-input").value = "";
}

// ------------------------------------------------------------------ //
// Knowledge Library (absorption %)                                    //
// ------------------------------------------------------------------ //

function addLibraryItem(item) {
  libraryItems.unshift(item);
  if (libraryItems.length > 200) libraryItems = libraryItems.slice(0, 200);
  renderLibrary();
}

function renderLibrary() {
  const container = document.getElementById("kb-library-list");
  if (!container) return;
  container.innerHTML = "";

  for (const item of libraryItems.slice(0, 50)) {
    const total = item.chunks_total || 1;
    const stored = item.chunks_stored + (item.images_stored || 0);
    const absorption = Math.min(100, Math.round((stored / Math.max(total, 1)) * 100));
    const barColor = absorption >= 80 ? "var(--green)" : absorption >= 50 ? "var(--yellow)" : "var(--red)";
    const statusIcon = item.success ? "\u2713" : "\u2717";
    const typeLabel = (item.source_type || "file").toUpperCase();

    const div = document.createElement("div");
    div.className = "library-item";
    div.innerHTML = `
      <div class="library-row">
        <span class="library-type ${item.source_type || ''}">${typeLabel}</span>
        <span class="library-name" title="${escapeHtml(item.source)}">${escapeHtml(truncate(item.source, 40))}</span>
        <span class="library-absorption">${absorption}%</span>
        <span class="library-status ${item.success ? 'ok' : 'err'}">${statusIcon}</span>
      </div>
      <div class="library-bar-bg">
        <div class="library-bar-fill" style="width:${absorption}%;background:${barColor}"></div>
      </div>
      <div class="library-meta">
        chunks=${item.chunks_stored} imgs=${item.images_stored || 0} concepts=${item.concepts_created} (${item.duration_s}s)
      </div>
    `;
    container.appendChild(div);
  }
}

async function loadLibraryHistory() {
  try {
    const resp = await fetch("/api/knowledge/history");
    if (resp.ok) {
      const data = await resp.json();
      if (Array.isArray(data)) {
        libraryItems = data;
        renderLibrary();
      }
    }
  } catch (e) {
    console.debug("Could not load ingestion history:", e);
  }
}

// ------------------------------------------------------------------ //
// Log                                                                 //
// ------------------------------------------------------------------ //

function logEntry(type, msg) {
  const container = document.getElementById("kb-log");
  const div = document.createElement("div");
  div.className = "log-entry";
  const ts = new Date().toLocaleTimeString("en-GB", { hour12: false });
  const typeClass = {
    sys: "cycle", learn: "learn", image: "image", url: "search",
    progress: "consolidation", error: "error",
  }[type] || "";
  div.innerHTML =
    `<span class="log-time">${ts}</span>` +
    `<span class="log-type ${typeClass}">${type.toUpperCase()}</span>` +
    `<span class="log-msg">${escapeHtml(msg)}</span>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

// ------------------------------------------------------------------ //
// Stats & Concepts                                                    //
// ------------------------------------------------------------------ //

function updateStats() {
  setText("stat-files", stats.files);
  setText("stat-urls", stats.urls);
  setText("stat-chunks", stats.chunks);
  setText("stat-images", stats.images);
  setText("stat-concepts", stats.concepts);
  setText("stat-rejected", stats.rejected);
  setText("badge-ingested", `${stats.ingested} ingested`);
}

function updateKBConcepts(concepts) {
  const container = document.getElementById("kb-concepts-list");
  if (!container) return;
  container.innerHTML = "";
  for (const c of concepts.slice(0, 20)) {
    const div = document.createElement("div");
    div.className = "concept-entry";
    const vol = (c.volatility ?? 1).toFixed(2);
    const statusClass = (c.status || "volatile").toLowerCase();
    div.innerHTML =
      `<span class="concept-name">${escapeHtml(c.concept || "")}</span>` +
      `<span class="concept-vol" style="color:${volColor(c.volatility)}">${vol}</span>` +
      `<span class="concept-status ${statusClass}">${c.status || "?"}</span>`;
    container.appendChild(div);
  }
}

// ------------------------------------------------------------------ //
// Utilities                                                           //
// ------------------------------------------------------------------ //

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

function truncate(str, max) {
  return str.length > max ? str.slice(0, max) + "..." : str;
}

function volColor(v) {
  if (v <= 0.2) return "var(--cyan)";
  if (v <= 0.5) return "var(--green)";
  if (v <= 0.8) return "var(--yellow)";
  return "var(--red)";
}

// ------------------------------------------------------------------ //
// Event Listeners                                                     //
// ------------------------------------------------------------------ //

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) {
    addFilesToQueue(e.dataTransfer.files);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    addFilesToQueue(fileInput.files);
    fileInput.value = "";
  }
});

document.getElementById("btn-upload-all").addEventListener("click", ingestAll);

document.getElementById("btn-clear-queue").addEventListener("click", () => {
  uploadQueue = uploadQueue.filter(i => i.status === "running");
  renderQueue();
});

document.getElementById("btn-ingest-url").addEventListener("click", ingestUrl);

document.getElementById("url-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") ingestUrl();
});

document.getElementById("btn-clear-log").addEventListener("click", () => {
  document.getElementById("kb-log").innerHTML = "";
});

// ------------------------------------------------------------------ //
// Init                                                                //
// ------------------------------------------------------------------ //

logEntry("sys", "Knowledge Base terminal ready");
connect();
loadLibraryHistory();
