const WS_URL = `ws://${location.host}/ws`;
let socket = null;
let reconnectTimer = null;
let lastModelRuntimeState = null;
let chatSessions = [];
let currentSessionId = null;
let currentMessages = [];
let streamingDiv = null;
let streamingText = "";
let reasoningDiv = null;
let pendingDeleteId = null;
let sidebarOpen = true;
let isStreaming = false;

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
  } else if (data.type === "chat_reasoning") {
    handleReasoning(data.step);
  } else if (data.type === "chat_token") {
    handleStreamToken(data.token);
  } else if (data.type === "chat_response") {
    finalizeStream(data);
  } else if (data.type === "stream_cancelled") {
    finalizeCancelledStream();
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
  const resolved = { ...ev, type: ev.event_type || ev.type };
  addAgentEntry(resolved);
}

// -- REASONING + STREAMING --
// The reasoning block and streaming body live INSIDE a single wrapper div
// (streamingDiv) so each Q&A is fully self-contained and no orphan
// reasoning blocks leak between messages.

function cleanupPreviousStream() {
  if (reasoningDiv) {
    reasoningDiv.classList.add("reasoning-complete");
    reasoningDiv = null;
  }
  if (streamingDiv) {
    const body = streamingDiv.querySelector(".streaming-cursor");
    if (body) body.classList.remove("streaming-cursor");
    streamingDiv = null;
    streamingText = "";
  }
}

function _ensureStreamingDiv() {
  if (streamingDiv) return;
  const output = document.getElementById("chat-output");
  streamingDiv = document.createElement("div");
  streamingDiv.className = "chat-msg assistant";

  const header = document.createElement("div");
  header.className = "msg-header";
  header.innerHTML =
    `<span class="msg-avatar assistant-avatar">$</span>` +
    `<span class="msg-timestamp">${nowTimestamp()}</span>`;
  streamingDiv.appendChild(header);

  output.appendChild(streamingDiv);
  streamingText = "";
}

function handleReasoning(step) {
  _ensureStreamingDiv();

  if (!reasoningDiv) {
    reasoningDiv = document.createElement("div");
    reasoningDiv.className = "reasoning-block";

    const header = document.createElement("div");
    header.className = "reasoning-header";
    header.innerHTML =
      '<span class="reasoning-icon">⚡</span> ' +
      '<span class="reasoning-label">QUANTUM REASONING</span>' +
      '<span class="reasoning-toggle">▶ show</span>';
    reasoningDiv.appendChild(header);

    header.addEventListener("click", () => {
      if (!reasoningDiv.classList.contains("reasoning-complete")) return;
      const expanded = reasoningDiv.classList.toggle("reasoning-expanded");
      const toggle = header.querySelector(".reasoning-toggle");
      if (toggle) toggle.textContent = expanded ? "▼ hide" : "▶ show";
    });

    const steps = document.createElement("div");
    steps.className = "reasoning-steps";
    reasoningDiv._stepsEl = steps;
    reasoningDiv.appendChild(steps);

    streamingDiv.appendChild(reasoningDiv);
  }

  const stepsEl = reasoningDiv._stepsEl;
  if (stepsEl) {
    const stepEl = document.createElement("div");
    stepEl.className = "reasoning-step";
    stepEl.textContent = step;
    stepsEl.appendChild(stepEl);
  }

  const output = document.getElementById("chat-output");
  output.scrollTop = output.scrollHeight;
}

function handleStreamToken(token) {
  if (reasoningDiv && !reasoningDiv.classList.contains("reasoning-complete")) {
    reasoningDiv.classList.add("reasoning-complete");
  }

  _ensureStreamingDiv();

  if (!streamingDiv._bodyEl) {
    const body = document.createElement("div");
    body.className = "msg-body streaming-cursor";
    streamingDiv._bodyEl = body;
    streamingDiv.appendChild(body);
  }

  streamingText += token;
  const body = streamingDiv._bodyEl;
  if (body) {
    renderRichContent(body, streamingText);
  }

  const output = document.getElementById("chat-output");
  output.scrollTop = output.scrollHeight;
}

function finalizeStream(data) {
  isStreaming = false;
  updateStopButton();

  if (reasoningDiv) {
    reasoningDiv.classList.add("reasoning-complete");
    reasoningDiv = null;
  }

  if (streamingDiv) {
    const body = streamingDiv._bodyEl;
    if (body) {
      body.classList.remove("streaming-cursor");
      const fullText = data.response || streamingText;
      renderRichContent(body, fullText);
    }

    if (data.confidence !== undefined || data.concepts) {
      const metaDiv = document.createElement("div");
      metaDiv.className = "meta";
      const parts = [];
      if (data.confidence !== undefined) parts.push(`confidence: ${data.confidence}`);
      if (data.concepts && data.concepts.length) parts.push(`concepts: ${data.concepts.join(", ")}`);
      if (data.duration_ms !== undefined) parts.push(`${data.duration_ms}ms`);
      metaDiv.textContent = parts.join(" | ");
      streamingDiv.appendChild(metaDiv);
    }

    const fullText = data.response || streamingText;
    const footer = buildMsgFooter(fullText);
    streamingDiv.appendChild(footer);

    currentMessages.push({
      role: "assistant",
      text: fullText,
      meta: { confidence: data.confidence, concepts: data.concepts, duration_ms: data.duration_ms },
      timestamp: Date.now(),
    });
    saveCurrentSession().catch(() => {});

    streamingDiv = null;
    streamingText = "";
  } else {
    appendChat("assistant", data.response, data);
  }
}

function finalizeCancelledStream() {
  isStreaming = false;
  updateStopButton();

  if (reasoningDiv) {
    reasoningDiv.classList.add("reasoning-complete");
    reasoningDiv = null;
  }

  if (streamingDiv) {
    const body = streamingDiv._bodyEl;
    if (body) {
      body.classList.remove("streaming-cursor");
      if (streamingText) {
        renderRichContent(body, streamingText + "\n\n*[ Generation stopped ]*");
      } else {
        body.textContent = "[ Generation cancelled ]";
      }
    }

    if (streamingText) {
      currentMessages.push({
        role: "assistant",
        text: streamingText,
        meta: { cancelled: true },
        timestamp: Date.now(),
      });
      saveCurrentSession().catch(() => {});
    }

    streamingDiv = null;
    streamingText = "";
  }
}

function updateStopButton() {
  const btn = document.getElementById("btn-stop-stream");
  if (btn) {
    btn.style.display = isStreaming ? "inline-flex" : "none";
  }
}

function buildMsgFooter(text) {
  const footer = document.createElement("div");
  footer.className = "msg-footer";

  const copyBtn = document.createElement("button");
  copyBtn.className = "msg-copy-btn-footer";
  copyBtn.innerHTML = '<span class="copy-icon">⧉</span> Copy';
  copyBtn.addEventListener("click", () => copyFullResponse(text, copyBtn));
  footer.appendChild(copyBtn);

  return footer;
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

function appendChat(role, text, meta = null, persist = true) {
  const output = document.getElementById("chat-output");
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;

  const header = document.createElement("div");
  header.className = "msg-header";

  if (role === "user") {
    header.innerHTML =
      `<span class="msg-avatar user-avatar">&gt;</span>` +
      `<span class="msg-timestamp">${nowTimestamp()}</span>`;
  } else if (role === "assistant") {
    header.innerHTML =
      `<span class="msg-avatar assistant-avatar">$</span>` +
      `<span class="msg-timestamp">${nowTimestamp()}</span>`;
  }

  div.appendChild(header);

  const body = document.createElement("div");
  body.className = "msg-body";

  if (role === "assistant") {
    renderRichContent(body, text);
  } else {
    body.textContent = text;
  }
  div.appendChild(body);

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

  if (role === "assistant") {
    div.appendChild(buildMsgFooter(text));
  }

  output.appendChild(div);
  output.scrollTop = output.scrollHeight;
  if (persist && (role === "user" || role === "assistant")) {
    currentMessages.push({
      role,
      text,
      meta: meta || {},
      timestamp: Date.now(),
    });
    saveCurrentSession().catch(() => {});
  }
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

// -- RICH CONTENT RENDERING --

function renderRichContent(container, text) {
  container.innerHTML = "";
  const fenceRe = /```([a-zA-Z0-9_+-]*)\n?([\s\S]*?)```/g;
  let last = 0;
  let m;
  while ((m = fenceRe.exec(text)) !== null) {
    const plain = text.slice(last, m.index);
    if (plain.trim()) {
      renderMarkdownBlock(container, plain.trim());
    }
    const lang = (m[1] || "code").trim();
    const code = m[2] || "";
    container.appendChild(buildCodeBlock(lang, code));
    last = fenceRe.lastIndex;
  }
  const tail = text.slice(last);
  if (tail.trim() || !container.children.length) {
    renderMarkdownBlock(container, tail.trim() || text);
  }
}

function renderMarkdownBlock(container, text) {
  const lines = text.split("\n");
  let html = "";
  for (const line of lines) {
    let processed = escapeHtml(line);

    if (/^#{1,3}\s+/.test(line)) {
      const level = line.match(/^(#{1,3})/)[1].length;
      const content = processed.replace(/^#{1,3}\s+/, '');
      processed = `<span class="md-h${level}">${content}</span>`;
      html += processed + "\n";
      continue;
    }

    processed = processed.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    processed = processed.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
    processed = processed.replace(/\\\((.+?)\\\)/g, '<span class="math-inline">$1</span>');
    processed = processed.replace(/\\\[(.+?)\\\]/g, '<div class="math-block">$1</div>');

    if (/^\s*\d+\.\s+/.test(line)) {
      const content = processed.replace(/^\s*\d+\.\s+/, '');
      const num = line.match(/^\s*(\d+)\./)[1];
      processed = `<span class="md-num-item"><span class="md-num">${num}.</span> ${content}</span>`;
    } else if (/^\s*[-*]\s+/.test(line)) {
      processed = '<span class="md-list-item">' +
        processed.replace(/^\s*[-*]\s+/, '') + '</span>';
    }

    if (/^\s*---\s*$/.test(line)) {
      processed = '<hr class="md-hr">';
    }

    html += processed + "\n";
  }
  const p = document.createElement("div");
  p.className = "chat-text";
  p.innerHTML = html;
  container.appendChild(p);
}

function buildCodeBlock(lang, code) {
  const wrap = document.createElement("div");
  wrap.className = "code-block";

  const header = document.createElement("div");
  header.className = "code-header";
  const langSpan = document.createElement("span");
  langSpan.textContent = lang.toUpperCase();
  const btn = document.createElement("button");
  btn.className = "code-copy-btn";
  btn.textContent = "COPY";
  btn.addEventListener("click", async () => {
    await copyCode(code, btn);
  });
  header.appendChild(langSpan);
  header.appendChild(btn);

  const pre = document.createElement("pre");
  pre.className = "code-pre";
  const c = document.createElement("code");
  c.textContent = code;
  pre.appendChild(c);

  wrap.appendChild(header);
  wrap.appendChild(pre);
  return wrap;
}

async function copyCode(code, button) {
  try {
    await navigator.clipboard.writeText(code);
    const old = button.textContent;
    button.textContent = "COPIED";
    setTimeout(() => { button.textContent = old; }, 1200);
  } catch (_) {
    button.textContent = "FAILED";
    setTimeout(() => { button.textContent = "COPY"; }, 1200);
  }
}

async function copyFullResponse(text, button) {
  const original = button.innerHTML;
  try {
    await navigator.clipboard.writeText(text);
    button.innerHTML = '<span class="copy-icon">✓</span> Copied';
    setTimeout(() => { button.innerHTML = original; }, 1500);
  } catch (_) {
    button.textContent = "Failed";
    setTimeout(() => { button.innerHTML = original; }, 1500);
  }
}

// -- CHAT SESSIONS --

async function loadSessions() {
  const resp = await fetch("/api/chat/sessions");
  const data = await resp.json();
  chatSessions = Array.isArray(data) ? data : [];
  if (!chatSessions.length) {
    await createSession("New chat");
    return;
  }
  currentSessionId = chatSessions[0].id;
  renderSessionList();
  await openSession(currentSessionId);
}

function renderSessionList() {
  const container = document.getElementById("session-list");
  if (!container) return;
  container.innerHTML = "";
  for (const s of chatSessions) {
    const item = document.createElement("div");
    item.className = "session-item" + (s.id === currentSessionId ? " active" : "");
    item.dataset.id = s.id;

    const content = document.createElement("div");
    content.className = "session-item-content";

    const title = document.createElement("div");
    title.className = "session-item-title";
    title.textContent = truncate(s.title || "New chat", 24);

    const meta = document.createElement("div");
    meta.className = "session-item-meta";
    const count = s.message_count || 0;
    const dateStr = s.updated_at ? new Date(s.updated_at * 1000).toLocaleDateString("en-GB", {
      day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit"
    }) : "";
    meta.textContent = `${count} msgs ${dateStr ? "· " + dateStr : ""}`;

    content.appendChild(title);
    content.appendChild(meta);

    const delBtn = document.createElement("button");
    delBtn.className = "session-item-delete";
    delBtn.textContent = "✕";
    delBtn.title = "Delete session";
    delBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      showDeleteModal(s.id, s.title || "New chat");
    });

    item.appendChild(content);
    item.appendChild(delBtn);

    item.addEventListener("click", () => {
      if (s.id !== currentSessionId) {
        openSession(s.id);
      }
    });

    container.appendChild(item);
  }
}

async function createSession(title = "New chat") {
  const resp = await fetch("/api/chat/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  const created = await resp.json();
  await refreshSessions();
  currentSessionId = created.id;
  renderSessionList();
  await openSession(currentSessionId);
}

async function openSession(sessionId) {
  const resp = await fetch(`/api/chat/sessions/${sessionId}`);
  const data = await resp.json();
  currentSessionId = sessionId;
  currentMessages = Array.isArray(data.messages) ? data.messages : [];
  reasoningDiv = null;
  streamingDiv = null;
  streamingText = "";
  const output = document.getElementById("chat-output");
  output.innerHTML = "";
  for (const msg of currentMessages) {
    appendChat(msg.role, msg.text, msg.meta || null, false);
  }
  renderSessionList();
}

async function saveCurrentSession() {
  if (!currentSessionId) return;
  const firstUser = currentMessages.find((m) => m.role === "user");
  const title = firstUser ? firstUser.text.slice(0, 40) : "New chat";
  await fetch(`/api/chat/sessions/${currentSessionId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title,
      messages: currentMessages,
    }),
  });
  await refreshSessions();
  renderSessionList();
}

async function refreshSessions() {
  const resp = await fetch("/api/chat/sessions");
  const data = await resp.json();
  chatSessions = Array.isArray(data) ? data : [];
}

async function deleteSession(sessionId) {
  await fetch(`/api/chat/sessions/${sessionId}`, { method: "DELETE" });
  await refreshSessions();
  if (!chatSessions.length) {
    await createSession("New chat");
    return;
  }
  if (currentSessionId === sessionId) {
    currentSessionId = chatSessions[0].id;
    await openSession(currentSessionId);
  }
  renderSessionList();
}

// -- DELETE MODAL --

function showDeleteModal(sessionId, sessionTitle) {
  pendingDeleteId = sessionId;
  const modal = document.getElementById("delete-modal");
  const body = document.getElementById("delete-modal-body");
  body.innerHTML = `Permanently delete session <span class="modal-session-name">"${escapeHtml(truncate(sessionTitle, 30))}"</span>?<br>This cannot be undone.`;
  modal.classList.add("visible");
}

function hideDeleteModal() {
  pendingDeleteId = null;
  document.getElementById("delete-modal").classList.remove("visible");
}

document.getElementById("modal-cancel").addEventListener("click", hideDeleteModal);
document.getElementById("modal-confirm").addEventListener("click", async () => {
  if (pendingDeleteId) {
    await deleteSession(pendingDeleteId);
  }
  hideDeleteModal();
});
document.getElementById("delete-modal").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) hideDeleteModal();
});

// -- SIDEBAR TOGGLE --

function toggleSidebar() {
  const sidebar = document.getElementById("chat-sidebar");
  sidebarOpen = !sidebarOpen;
  if (sidebarOpen) {
    sidebar.classList.remove("collapsed");
  } else {
    sidebar.classList.add("collapsed");
  }
}

// -- UTILITIES --

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function truncate(str, n) {
  return str.length > n ? str.slice(0, n) + "…" : str;
}

function nowTimestamp() {
  return new Date().toLocaleTimeString("en-GB", { hour12: false });
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
    if (!currentSessionId) return;
    cleanupPreviousStream();
    appendChat("user", text);
    isStreaming = true;
    updateStopButton();
    sendAction("chat_stream", { text, session_id: currentSessionId });
    input.value = "";
  }
});

document.getElementById("btn-toggle-sidebar").addEventListener("click", toggleSidebar);

document.getElementById("btn-new-session").addEventListener("click", async () => {
  await createSession("New chat");
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

const stopStreamBtn = document.getElementById("btn-stop-stream");
if (stopStreamBtn) {
  stopStreamBtn.addEventListener("click", () => {
    sendAction("cancel_stream");
    isStreaming = false;
    updateStopButton();
  });
}

// -- INIT --
appendSystem("Initializing quantum brain terminal...");
connect();
loadSessions().catch((e) => {
  appendError(`Failed to load chat sessions: ${e.message || e}`);
});
