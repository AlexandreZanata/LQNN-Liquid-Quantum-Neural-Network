const WS_URL = `ws://${location.host}/ws`;
let socket = null;
let reconnectTimer = null;
let lastModelRuntimeState = null;

function connect() {
  socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    setStatus(true);
    clearTimeout(reconnectTimer);
    appendSystem("WebSocket connected to quantum brain");
  };

  socket.onclose = () => {
    setStatus(false);
    reconnectTimer = setTimeout(connect, 3000);
  };

  socket.onerror = () => {
    socket.close();
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleMessage(data);
    } catch (e) {
      console.error("WS parse error:", e);
    }
  };
}

function sendAction(action, payload = {}) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ action, ...payload }));
  }
}

function setStatus(online) {
  const el = document.getElementById("status-indicator");
  el.textContent = online ? "ONLINE" : "OFFLINE";
  el.className = "status-pill " + (online ? "online" : "offline");
}

function handleMessage(data) {
  if (data.type === "state") {
    updateStats(data);
    updateTrainingLog(data.training_log || []);
    updateAgentLog(data.agent_activity || []);
    updateConcepts(data.recent_concepts || []);
    updateSystemMetrics(data.system || {});
  } else if (data.type === "chat_response") {
    appendChat("assistant", data.response, data);
  } else if (data.type === "search_result") {
    appendSystem(
      `SEARCH complete: ${data.concepts_learned} concepts, ` +
      `${data.images_processed} images (${data.duration_s}s)`
    );
  } else if (data.type === "learn_result") {
    appendSystem(`LEARNED "${data.concept}" +${data.associations} associations`);
  } else if (data.type === "consolidation_result") {
    appendSystem(
      `CONSOLIDATE: pruned=${data.pruned} crystallized=${data.crystallized}`
    );
  } else if (data.type === "self_play_result") {
    appendSystem(`SELF-PLAY: ${data.action} (${data.concept || "n/a"})`);
  } else if (data.type === "live_event") {
    handleLiveEvent(data);
  } else if (data.type === "training_started") {
    appendSystem("Training STARTED");
  } else if (data.type === "training_stopped") {
    appendSystem("Training STOPPED");
  } else if (data.type === "error") {
    appendError(data.message);
  }
}

function handleLiveEvent(ev) {
  const evType = ev.type === "live_event" ? (ev.type_detail || ev.type) : ev.type;
  addAgentEntry(ev);
}

// -- STATS UPDATE --

function updateStats(data) {
  const mem = data.memory || {};
  const train = data.training || {};
  const agents = data.agents || {};

  setText("st-concepts", mem.concepts ?? "--");
  setText("st-assoc", mem.associations ?? "--");
  setText("st-learned", mem.learn_count ?? "--");
  setText("st-queries", mem.query_count ?? "--");

  const clipEl = document.getElementById("st-clip");
  if (clipEl) {
    clipEl.textContent = mem.clip_ready ? "READY" : "LOADING";
    clipEl.className = "stat-value " + (mem.clip_ready ? "info" : "warn");
  }

  const llmEl = document.getElementById("st-llm");
  if (llmEl) {
    llmEl.textContent = mem.llm_ready ? "READY" : "LOADING";
    llmEl.className = "stat-value " + (mem.llm_ready ? "info" : "warn");
  }

  const badgeLlm = document.getElementById("badge-llm");
  if (badgeLlm) {
    badgeLlm.textContent = "LLM: " + (mem.llm_ready ? "READY" : "LOADING");
  }

  const trainEl = document.getElementById("st-training");
  if (trainEl) {
    trainEl.textContent = train.running ? "RUNNING" : "STOPPED";
    trainEl.className = "stat-value " + (train.running ? "info" : "warn");
  }

  setText("badge-cycle", `cycle: ${train.cycle ?? 0}`);
  setText("badge-agent", `cycle: ${agents.cycle ?? 0}`);

  const phaseEl = document.getElementById("sys-phase");
  if (phaseEl) {
    const phase = (train.phase || "").replace(/_/g, " ").toUpperCase();
    phaseEl.textContent = `PHASE: ${phase || "--"}`;
  }

  const uptimeEl = document.getElementById("sys-uptime");
  if (uptimeEl) {
    uptimeEl.textContent = `UP: ${formatUptime(train.uptime_s)}`;
  }
}

function updateSystemMetrics(sys) {
  setText("sys-cpu", `CPU: ${sys.cpu_percent ?? 0}%`);
  const used = sys.gpu_used_gb;
  const total = sys.gpu_total_gb;
  const source = sys.gpu_metric_source || "";
  if (source === "no_cuda") {
    setText("sys-gpu", "GPU: unavailable");
  } else if ((used ?? 0) === 0 && (total ?? 0) > 0 && source !== "nvidia_smi") {
    setText("sys-gpu", `GPU: --/${total}GB (metric timeout)`);
  } else {
    setText("sys-gpu", `GPU: ${used ?? 0}/${total ?? 0}GB`);
  }
  setText("sys-ram", `RAM: ${sys.ram_used_gb ?? 0}/${sys.ram_total_gb ?? 0}GB`);

  const runtimeState = sys.model_runtime_state || "unknown";
  if (runtimeState !== lastModelRuntimeState) {
    appendSystem(`MODEL STATE: ${runtimeState.toUpperCase()}`);
    lastModelRuntimeState = runtimeState;
  }
}

// -- TRAINING LOG --

let lastTrainLogLen = 0;
function updateTrainingLog(entries) {
  if (entries.length === lastTrainLogLen) return;
  lastTrainLogLen = entries.length;
  const container = document.getElementById("training-log");
  if (!container) return;
  container.innerHTML = "";
  const recent = entries.slice(-40);
  for (const e of recent) {
    const div = document.createElement("div");
    div.className = "log-entry";
    const ts = formatTimestamp(e.timestamp);
    const typeClass = getLogTypeClass(e.type);
    const msg = formatTrainEvent(e);
    div.innerHTML =
      `<span class="log-time">${ts}</span>` +
      `<span class="log-type ${typeClass}">${(e.type || "").toUpperCase()}</span>` +
      `<span class="log-msg">${msg}</span>`;
    container.appendChild(div);
  }
  container.scrollTop = container.scrollHeight;
}

function formatTrainEvent(e) {
  switch (e.type) {
    case "warmup":
      if (e.stage === "cache_check") return `[CACHE] ${e.message || ""}`;
      if (e.stage === "clip_load") return `[LOAD] ${e.message || ""}`;
      if (e.stage === "clip_ready") return `[READY] ${e.message || ""}`;
      if (e.stage === "llm_load") return `[DOWNLOAD/LOAD] ${e.message || ""}`;
      if (e.stage === "llm_ready") return `[READY] ${e.message || ""}`;
      if (e.stage === "inference_ready") return `[INFERENCE] ${e.message || ""}`;
      return e.message || "";
    case "cycle_start":
      return `Cycle ${e.cycle} [${e.phase || ""}]`;
    case "cycle_end":
      return `+${e.learned || 0} concepts +${e.images || 0} imgs (${e.duration_s || 0}s)`;
    case "cycle_crawl":
      return `crawled: ${e.concepts_learned || 0} concepts, ${e.images_processed || 0} images`;
    case "consolidation":
      return `pruned=${e.pruned || 0} crystal=${e.crystallized || 0} decay=${e.decayed || 0}`;
    case "self_play":
      return `${e.action || ""} "${e.concept || ""}"`;
    case "trainer_start":
    case "trainer_stop":
      return e.message || "";
    case "error":
      return e.message || "unknown error";
    default:
      return e.message || JSON.stringify(e).substring(0, 80);
  }
}

// -- AGENT LOG --

let lastAgentLogLen = 0;
function updateAgentLog(entries) {
  if (entries.length === lastAgentLogLen) return;
  lastAgentLogLen = entries.length;
  const container = document.getElementById("agent-log");
  if (!container) return;
  container.innerHTML = "";
  const recent = entries.slice(-40);
  for (const e of recent) {
    addAgentEntryToContainer(container, e);
  }
  container.scrollTop = container.scrollHeight;
}

function addAgentEntry(e) {
  const container = document.getElementById("agent-log");
  if (!container) return;
  addAgentEntryToContainer(container, e);
  container.scrollTop = container.scrollHeight;
}

function addAgentEntryToContainer(container, e) {
  const div = document.createElement("div");
  div.className = "log-entry";
  const ts = formatTimestamp(e.timestamp);
  const typeClass = getLogTypeClass(e.type);
  const msg = formatAgentEvent(e);
  div.innerHTML =
    `<span class="log-time">${ts}</span>` +
    `<span class="log-type ${typeClass}">${(e.type || "").toUpperCase()}</span>` +
    `<span class="log-msg">${msg}</span>`;
  container.appendChild(div);
}

function formatAgentEvent(e) {
  switch (e.type) {
    case "search":
      return `"${e.concept || ""}" -> query: "${e.query || ""}"`;
    case "learn":
      return `"${(e.concept || "").substring(0, 50)}" +${e.associations || 0} assocs [${e.source || ""}]`;
    case "learn_image":
      return `"${e.concept || ""}" relevance=${e.relevance_score || 0} +${e.associations || 0} assocs`;
    case "judge_reject":
      return `REJECTED "${e.concept || ""}" (${e.reason || ""})`;
    case "agent_error":
      return `ERROR "${e.concept || ""}" : ${e.error || ""}`;
    default:
      return e.message || JSON.stringify(e).substring(0, 80);
  }
}

// -- CONCEPTS --

function updateConcepts(concepts) {
  const container = document.getElementById("concepts-list");
  if (!container) return;
  container.innerHTML = "";
  setText("badge-concepts", `${concepts.length} concepts`);
  for (const c of concepts) {
    const div = document.createElement("div");
    const statusClass = (c.status || "volatile").toLowerCase();
    div.className = `concept-entry ${statusClass}`;
    const vol = (c.volatility ?? 1).toFixed(2);
    div.innerHTML =
      `<span class="concept-name">${escapeHtml(c.concept || "")}</span>` +
      `<span class="concept-vol" style="color:${volColor(c.volatility)}">${vol}</span>` +
      `<span class="concept-count">x${c.access_count || 0}</span>` +
      `<span class="concept-status ${statusClass}">${c.status || "?"}</span>`;
    container.appendChild(div);
  }
}

function volColor(v) {
  if (v <= 0.2) return "var(--cyan)";
  if (v <= 0.5) return "var(--green)";
  if (v <= 0.8) return "var(--yellow)";
  return "var(--red)";
}

// -- CHAT OUTPUT --

function appendChat(role, text, meta = null) {
  const output = document.getElementById("chat-output");
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.textContent = text;

  if (meta && role === "assistant") {
    const metaDiv = document.createElement("div");
    metaDiv.className = "meta";
    const parts = [];
    if (meta.confidence !== undefined)
      parts.push(`confidence: ${meta.confidence}`);
    if (meta.concepts && meta.concepts.length)
      parts.push(`concepts: ${meta.concepts.join(", ")}`);
    if (meta.duration_ms !== undefined)
      parts.push(`${meta.duration_ms}ms`);
    metaDiv.textContent = parts.join(" | ");
    div.appendChild(metaDiv);
  }

  output.appendChild(div);
  output.scrollTop = output.scrollHeight;
}

function appendSystem(text) {
  const output = document.getElementById("chat-output");
  const div = document.createElement("div");
  div.className = "chat-msg system";
  div.textContent = text;
  output.appendChild(div);
  output.scrollTop = output.scrollHeight;
}

function appendError(text) {
  const output = document.getElementById("chat-output");
  const div = document.createElement("div");
  div.className = "chat-msg error";
  div.textContent = text;
  output.appendChild(div);
  output.scrollTop = output.scrollHeight;
}

// -- UTILITIES --

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function formatUptime(s) {
  if (!s) return "--";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
}

function formatTimestamp(ts) {
  if (!ts) return "--:--:--";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-GB", { hour12: false });
}

function getLogTypeClass(type) {
  const map = {
    warmup: "start",
    search: "search",
    learn: "learn",
    learn_image: "image",
    judge_reject: "judge",
    consolidation: "consolidation",
    self_play: "self-play",
    cycle_start: "cycle",
    cycle_end: "cycle",
    cycle_crawl: "learn",
    trainer_start: "start",
    trainer_stop: "stop",
    error: "error",
    agent_error: "error",
  };
  return map[type] || "";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// -- EVENT LISTENERS --

document.getElementById("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const input = document.getElementById("chat-input");
    const text = input.value.trim();
    if (!text) return;
    appendChat("user", text);
    sendAction("chat", { text });
    input.value = "";
  }
});

document.getElementById("btn-consolidate").addEventListener("click", () => {
  sendAction("consolidate");
});

document.getElementById("btn-self-play").addEventListener("click", () => {
  sendAction("self_play");
});

document.getElementById("btn-learn").addEventListener("click", () => {
  const concept = prompt("Enter concept to learn:");
  if (concept) sendAction("learn", { concept });
});

document.getElementById("btn-search").addEventListener("click", () => {
  const query = prompt("Enter search query:");
  if (query) sendAction("search", { query });
});

document.getElementById("btn-start-train").addEventListener("click", () => {
  sendAction("start_training");
});

document.getElementById("btn-stop-train").addEventListener("click", () => {
  sendAction("stop_training");
});

// -- INIT --
appendSystem("Initializing quantum brain terminal...");
connect();
