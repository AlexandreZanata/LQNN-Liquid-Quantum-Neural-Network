const ws = new WebSocket(`ws://${location.host}/ws`);

const LANGUAGES = {
  ja: { name: "Japanese",   flag: "\u{1F1EF}\u{1F1F5}" },
  ru: { name: "Russian",    flag: "\u{1F1F7}\u{1F1FA}" },
  it: { name: "Italian",    flag: "\u{1F1EE}\u{1F1F9}" },
  zh: { name: "Chinese",    flag: "\u{1F1E8}\u{1F1F3}" },
  ko: { name: "Korean",     flag: "\u{1F1F0}\u{1F1F7}" },
  fr: { name: "French",     flag: "\u{1F1EB}\u{1F1F7}" },
  de: { name: "German",     flag: "\u{1F1E9}\u{1F1EA}" },
  es: { name: "Spanish",    flag: "\u{1F1EA}\u{1F1F8}" },
  pt: { name: "Portuguese", flag: "\u{1F1E7}\u{1F1F7}" },
  ar: { name: "Arabic",     flag: "\u{1F1F8}\u{1F1E6}" },
};

const DATASETS = [
  {
    id: "tatoeba",
    name: "TATOEBA",
    license: "CC BY 2.0",
    size: "13M+ sentences",
    desc: "Human-verified sentence pairs across 429 languages. Ideal for language learning.",
    langs: ["ja", "ru", "it", "zh", "ko", "fr", "de", "es", "pt", "ar"],
  },
  {
    id: "opus100",
    name: "OPUS-100",
    license: "Open",
    size: "55M pairs",
    desc: "English-centric multilingual corpus covering 100 languages, up to 1M pairs each.",
    langs: ["ja", "ru", "it", "zh", "ko", "fr", "de", "es", "pt", "ar"],
  },
  {
    id: "paracrawl",
    name: "PARACRAWL v8",
    license: "CC0 / Open",
    size: "Varies per lang",
    desc: "Large-scale parallel web crawl. EN-RU (2.8M), EN-IT (14M), EN-DE, EN-FR and more.",
    langs: ["ru", "it", "fr", "de", "es", "pt"],
  },
  {
    id: "jparacrawl",
    name: "JPARACRAWL v3",
    license: "Open / NTT",
    size: "53M EN-JA",
    desc: "Largest public English-Japanese parallel corpus from NTT web crawling.",
    langs: ["ja"],
  },
];

let langStatus = {};
let eventCount = 0;

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function sendWS(action, data) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action, ...data }));
  }
}

/* ---------- WebSocket ---------- */

ws.onopen = () => {
  document.getElementById("status-indicator").textContent = "ONLINE";
  document.getElementById("status-indicator").className = "status-pill online";
  sendWS("get_state");
  sendWS("get_language_status");
};

ws.onclose = () => {
  document.getElementById("status-indicator").textContent = "OFFLINE";
  document.getElementById("status-indicator").className = "status-pill offline";
};

ws.onmessage = (ev) => {
  let data;
  try { data = JSON.parse(ev.data); } catch { return; }
  handleMessage(data);
};

function handleMessage(data) {
  const t = data.type;

  if (t === "state") {
    updateSystemMetrics(data);
  } else if (t === "language_status") {
    langStatus = data.languages || {};
    updateLangCards();
    updateLangStats(data);
  } else if (t === "live_event") {
    const et = data.event_type || "";
    if (et.startsWith("lang_")) {
      appendLogEntry(data);
      if (et === "lang_pair_stored" && data.lang_code) {
        addRecentConcept(data);
      }
      eventCount++;
      setText("badge-lang-events", `${eventCount} events`);
    }
    if (et === "lang_status_update") {
      langStatus = data.languages || langStatus;
      updateLangCards();
      updateLangStats(data);
    }
  } else if (t === "language_training_started") {
    appendLogEntry({ event_type: "lang_start", message: "Language training started" });
  } else if (t === "language_training_stopped") {
    appendLogEntry({ event_type: "lang_stop", message: "Language training stopped" });
  }
}

function updateSystemMetrics(data) {
  const sys = data.system || {};
  if (sys.cpu_percent !== undefined) setText("sys-cpu", `CPU: ${sys.cpu_percent}%`);
  if (sys.ram_used_gb !== undefined) setText("sys-ram", `RAM: ${sys.ram_used_gb}/${sys.ram_total_gb}GB`);
  if (sys.gpu_used_gb !== undefined) setText("sys-gpu", `GPU: ${sys.gpu_used_gb}/${sys.gpu_total_gb}GB`);
}

/* ---------- Language Cards ---------- */

function buildLangGrid() {
  const grid = document.getElementById("lang-grid");
  grid.innerHTML = "";

  for (const [code, info] of Object.entries(LANGUAGES)) {
    const st = langStatus[code] || {};
    const pairs = st.pairs_stored || 0;
    const total = st.total_pairs || 0;
    const pct = total > 0 ? Math.round((pairs / total) * 100) : 0;
    const status = st.status || "idle";

    const card = document.createElement("div");
    card.className = `lang-card ${status !== "idle" ? status : ""}`;
    card.dataset.code = code;

    card.innerHTML = `
      <div class="lang-card-header">
        <span class="lang-card-flag">${info.flag}</span>
        <span class="lang-card-name">${info.name}</span>
        <span class="lang-card-code">${code}</span>
        <span class="lang-card-status ${status}">${status.toUpperCase()}</span>
      </div>
      <div class="lang-card-stats">
        <span>Pairs: <span class="val">${pairs.toLocaleString()}</span></span>
        <span>Concepts: <span class="val">${(st.concepts || 0).toLocaleString()}</span></span>
        <span>${pct}%</span>
      </div>
      <div class="lang-card-bar">
        <div class="lang-card-bar-fill" style="width:${pct}%"></div>
      </div>
      <div class="lang-card-actions">
        <button class="action-btn" onclick="trainLang('${code}')">TRAIN</button>
        <button class="action-btn danger" onclick="stopLang('${code}')">STOP</button>
      </div>
    `;

    grid.appendChild(card);
  }
}

function updateLangCards() {
  for (const [code, st] of Object.entries(langStatus)) {
    const card = document.querySelector(`.lang-card[data-code="${code}"]`);
    if (!card) continue;

    const pairs = st.pairs_stored || 0;
    const total = st.total_pairs || 0;
    const pct = total > 0 ? Math.round((pairs / total) * 100) : 0;
    const status = st.status || "idle";

    card.className = `lang-card ${status !== "idle" ? status : ""}`;

    const statusEl = card.querySelector(".lang-card-status");
    if (statusEl) {
      statusEl.textContent = status.toUpperCase();
      statusEl.className = `lang-card-status ${status}`;
    }

    const statsEl = card.querySelector(".lang-card-stats");
    if (statsEl) {
      statsEl.innerHTML = `
        <span>Pairs: <span class="val">${pairs.toLocaleString()}</span></span>
        <span>Concepts: <span class="val">${(st.concepts || 0).toLocaleString()}</span></span>
        <span>${pct}%</span>
      `;
    }

    const barFill = card.querySelector(".lang-card-bar-fill");
    if (barFill) barFill.style.width = `${pct}%`;
  }

  const activeCount = Object.values(langStatus).filter(s => s.status === "training").length;
  setText("badge-lang-count", `${activeCount} active`);
  setText("lang-active", `ACTIVE: ${activeCount}`);
}

function updateLangStats(data) {
  let totalPairs = 0;
  let activeLangs = 0;
  let totalAssoc = 0;
  let totalDatasets = 0;
  let trainingRunning = false;

  for (const st of Object.values(langStatus)) {
    totalPairs += st.pairs_stored || 0;
    totalAssoc += st.associations || 0;
    if (st.status === "training") { activeLangs++; trainingRunning = true; }
    if (st.datasets_downloaded) totalDatasets += st.datasets_downloaded;
  }

  setText("stat-total-pairs", totalPairs.toLocaleString());
  setText("stat-langs-active", activeLangs);
  setText("stat-datasets-dl", totalDatasets);
  setText("stat-assoc-gen", totalAssoc.toLocaleString());
  setText("lang-total-pairs", `PAIRS: ${totalPairs.toLocaleString()}`);

  const statusEl = document.getElementById("stat-training-status");
  if (statusEl) {
    statusEl.textContent = trainingRunning ? "RUNNING" : "STOPPED";
    statusEl.className = "stat-value " + (trainingRunning ? "info" : "warn");
  }
}

/* ---------- Datasets ---------- */

function buildDatasetList() {
  const list = document.getElementById("dataset-list");
  list.innerHTML = "";

  for (const ds of DATASETS) {
    const item = document.createElement("div");
    item.className = "dataset-item";
    item.innerHTML = `
      <div class="dataset-row">
        <span class="dataset-name">${ds.name}</span>
        <span class="dataset-license">${ds.license}</span>
        <span class="dataset-size">${ds.size}</span>
      </div>
      <div class="dataset-desc">${ds.desc}</div>
      <div class="dataset-langs">
        ${ds.langs.map(l => `<span class="dataset-lang-tag">${l.toUpperCase()}: ${LANGUAGES[l]?.name || l}</span>`).join("")}
      </div>
      <div class="dataset-actions">
        <button class="action-btn" onclick="downloadDataset('${ds.id}')">DOWNLOAD & INGEST</button>
      </div>
    `;
    list.appendChild(item);
  }
}

/* ---------- Training Log ---------- */

function appendLogEntry(data) {
  const log = document.getElementById("lang-log");
  const entry = document.createElement("div");
  entry.className = "log-entry";

  const ts = new Date().toLocaleTimeString("en-US", { hour12: false });
  const et = data.event_type || "info";
  const typeClass = et.replace(/_/g, "-");

  let msg = data.message || "";
  if (!msg) {
    if (et === "lang_pair_stored") {
      msg = `[${(data.lang_code || "").toUpperCase()}] "${(data.source_text || "").substring(0, 40)}" -> "${(data.target_text || "").substring(0, 40)}"`;
    } else if (et === "lang_progress") {
      msg = `[${(data.lang_code || "").toUpperCase()}] ${data.pairs_processed || 0} pairs processed`;
    } else if (et === "lang_dataset_downloaded") {
      msg = `Dataset ${data.dataset || ""} downloaded for ${(data.lang_code || "").toUpperCase()}`;
    } else {
      msg = JSON.stringify(data).substring(0, 120);
    }
  }

  entry.innerHTML = `
    <span class="log-time">${ts}</span>
    <span class="log-type ${typeClass}">${et.replace("lang_", "").toUpperCase()}</span>
    <span class="log-msg">${msg}</span>
  `;

  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

/* ---------- Recent Concepts ---------- */

function addRecentConcept(data) {
  const list = document.getElementById("lang-concepts-list");
  const entry = document.createElement("div");
  entry.className = "lang-concept-entry";
  entry.innerHTML = `
    <span class="lang-concept-lang">${data.lang_code || "??"}</span>
    <span class="lang-concept-source">${(data.source_text || "").substring(0, 50)}</span>
    <span class="lang-concept-arrow">\u2192</span>
    <span class="lang-concept-target">${(data.target_text || "").substring(0, 50)}</span>
  `;

  list.insertBefore(entry, list.firstChild);
  while (list.children.length > 50) list.removeChild(list.lastChild);
}

/* ---------- Actions ---------- */

function trainLang(code) {
  sendWS("start_language_training", { languages: [code] });
  appendLogEntry({ event_type: "lang_start", message: `Starting training for ${LANGUAGES[code]?.name || code}` });
}

function stopLang(code) {
  sendWS("stop_language_training", { languages: [code] });
  appendLogEntry({ event_type: "lang_stop", message: `Stopping training for ${LANGUAGES[code]?.name || code}` });
}

function downloadDataset(datasetId) {
  sendWS("download_dataset", { dataset_id: datasetId });
  appendLogEntry({ event_type: "lang_download", message: `Requested download: ${datasetId}` });
}

/* ---------- Button Listeners ---------- */

document.getElementById("btn-train-all").addEventListener("click", () => {
  const allCodes = Object.keys(LANGUAGES);
  sendWS("start_language_training", { languages: allCodes });
  appendLogEntry({ event_type: "lang_start", message: "Starting training for ALL languages" });
});

document.getElementById("btn-stop-all").addEventListener("click", () => {
  sendWS("stop_language_training", { languages: Object.keys(LANGUAGES) });
  appendLogEntry({ event_type: "lang_stop", message: "Stopping ALL language training" });
});

document.getElementById("btn-clear-lang-log").addEventListener("click", () => {
  document.getElementById("lang-log").innerHTML = "";
  eventCount = 0;
  setText("badge-lang-events", "0 events");
});

/* ---------- Init ---------- */

buildLangGrid();
buildDatasetList();
