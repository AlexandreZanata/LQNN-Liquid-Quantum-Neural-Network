const WS_URL = `ws://${location.host}/ws`;
let socket = null;
let reconnectTimer = null;

function connect() {
  socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    setStatus(true);
    clearTimeout(reconnectTimer);
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
  el.className = "status " + (online ? "online" : "offline");
}

function handleMessage(data) {
  if (data.type === "state") {
    updateStats(data);
  } else if (data.type === "chat_response") {
    appendChat("assistant", data.response, data);
  } else if (data.type === "search_result") {
    appendChat("assistant",
      `Search done: ${data.concepts_learned} concepts learned, ` +
      `${data.images_processed} images processed (${data.duration_s}s)`);
  } else if (data.type === "learn_result") {
    appendChat("assistant",
      `Learned "${data.concept}" with ${data.associations} associations`);
  } else if (data.type === "consolidation_result") {
    appendChat("assistant",
      `Consolidation: pruned=${data.pruned}, crystallized=${data.crystallized}`);
  } else if (data.type === "self_play_result") {
    appendChat("assistant",
      `Self-play: ${data.action} (concept: ${data.concept || "n/a"})`);
  } else if (data.type === "error") {
    appendChat("assistant", `Error: ${data.message}`);
  }
}

function updateStats(data) {
  const mem = data.memory || {};
  const train = data.training || {};
  const agents = data.agents || {};

  setText("stat-concepts", `Concepts: ${mem.concepts ?? "--"}`);
  setText("stat-associations", `Associations: ${mem.associations ?? "--"}`);
  setText("stat-learn-count", `Learned: ${mem.learn_count ?? "--"}`);
  setText("stat-query-count", `Queries: ${mem.query_count ?? "--"}`);

  setText("stat-training-cycle", `Cycle: ${train.cycle ?? "--"}`);
  setText("stat-training-uptime", `Uptime: ${formatUptime(train.uptime_s)}`);
  setText("stat-training-running", `Status: ${train.running ? "RUNNING" : "STOPPED"}`);

  setText("stat-agent-cycle", `Cycle: ${agents.cycle ?? "--"}`);
  setText("stat-agent-online", `Online: ${agents.online ? "YES" : "NO"}`);
  setText("stat-agent-gaps", `Gaps: ${agents.gap_queue_size ?? "--"}`);

  const list = document.getElementById("concept-list");
  if (list && data.recent_concepts) {
    list.innerHTML = "";
    for (const c of data.recent_concepts.slice(0, 15)) {
      const li = document.createElement("li");
      li.textContent = `${c.concept} (v:${c.volatility?.toFixed(2)}, x${c.access_count})`;
      list.appendChild(li);
    }
  }
}

function appendChat(role, text, meta = null) {
  const output = document.getElementById("chat-output");
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.textContent = text;

  if (meta && role === "assistant") {
    const metaDiv = document.createElement("div");
    metaDiv.className = "meta";
    const parts = [];
    if (meta.confidence !== undefined) parts.push(`confidence: ${meta.confidence}`);
    if (meta.concepts && meta.concepts.length) parts.push(`concepts: ${meta.concepts.join(", ")}`);
    if (meta.duration_ms !== undefined) parts.push(`${meta.duration_ms}ms`);
    metaDiv.textContent = parts.join(" | ");
    div.appendChild(metaDiv);
  }

  output.appendChild(div);
  output.scrollTop = output.scrollHeight;
}

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

// -- Event listeners --

document.getElementById("btn-send").addEventListener("click", () => {
  const input = document.getElementById("chat-input");
  const text = input.value.trim();
  if (!text) return;
  appendChat("user", text);
  sendAction("chat", { text });
  input.value = "";
});

document.getElementById("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("btn-send").click();
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

// -- Init --
connect();
