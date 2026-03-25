const POLL_INTERVAL = 3000;
const conceptHistory = [];
let polling = null;

async function fetchStatus() {
  try {
    const [brainRes, trainingRes, agentRes] = await Promise.all([
      fetch("/api/brain/status").then(r => r.json()).catch(() => null),
      fetch("/api/training/status").then(r => r.json()).catch(() => null),
      fetch("/api/agents/status").then(r => r.json()).catch(() => null),
    ]);

    if (brainRes && brainRes.memory) {
      setText("m-concepts", brainRes.memory.concepts ?? "--");
      setText("m-associations", brainRes.memory.associations ?? "--");
      setText("m-learn-count", brainRes.memory.learn_count ?? "--");
      setText("m-query-count", brainRes.memory.query_count ?? "--");
      setText("m-clip", brainRes.memory.clip_ready ? "LOADED" : "NOT LOADED");
      setText("m-llm", brainRes.memory.llm_ready ? "LOADED" : "NOT LOADED");

      conceptHistory.push(brainRes.memory.concepts || 0);
      if (conceptHistory.length > 60) conceptHistory.shift();
      drawChart();
    }

    if (trainingRes) {
      const running = trainingRes.running;
      const badge = document.getElementById("t-running");
      badge.textContent = running ? "RUNNING" : "STOPPED";
      badge.className = "badge " + (running ? "running" : "stopped");

      setText("t-cycle", trainingRes.cycle ?? "--");
      setText("t-uptime", formatUptime(trainingRes.uptime_s));

      const lm = trainingRes.latest_metrics;
      if (lm) {
        setText("t-concepts-cycle", lm.concepts_this_cycle ?? "--");
        setText("t-images-cycle", lm.images_this_cycle ?? "--");
        setText("t-pruned", lm.consolidation_pruned ?? "--");
        setText("t-crystallized", lm.consolidation_crystallized ?? "--");
        setText("t-duration", (lm.cycle_duration_s ?? 0).toFixed(1) + "s");
        addLogEntry(lm);
      }
    }

    if (agentRes) {
      setText("a-cycle", agentRes.cycle ?? "--");
      setText("a-online", agentRes.online ? "YES" : "NO");
      setText("a-gaps", agentRes.gap_queue_size ?? "--");
    }

    setStatus(true);
  } catch (e) {
    setStatus(false);
  }
}

function addLogEntry(metrics) {
  const log = document.getElementById("training-log");
  const line = document.createElement("div");
  line.textContent =
    `[Cycle ${metrics.cycle}] concepts=${metrics.total_concepts} ` +
    `assoc=${metrics.total_associations} learned=${metrics.concepts_this_cycle} ` +
    `imgs=${metrics.images_this_cycle} pruned=${metrics.consolidation_pruned} ` +
    `crystal=${metrics.consolidation_crystallized} (${metrics.cycle_duration_s?.toFixed(1)}s)`;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;

  while (log.children.length > 100) log.removeChild(log.firstChild);
}

function drawChart() {
  const canvas = document.getElementById("chart-concepts");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (conceptHistory.length < 2) return;

  const max = Math.max(...conceptHistory, 1);
  const step = w / (conceptHistory.length - 1);

  ctx.strokeStyle = "#3b82f6";
  ctx.lineWidth = 2;
  ctx.beginPath();
  conceptHistory.forEach((val, i) => {
    const x = i * step;
    const y = h - (val / max) * (h - 20) - 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#64748b";
  ctx.font = "10px monospace";
  ctx.fillText(`max: ${max}`, 5, 14);
  ctx.fillText(`now: ${conceptHistory[conceptHistory.length - 1]}`, 5, h - 4);
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setStatus(online) {
  const el = document.getElementById("status-indicator");
  el.textContent = online ? "ONLINE" : "OFFLINE";
  el.className = "status " + (online ? "online" : "offline");
}

function formatUptime(s) {
  if (!s) return "--";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
}

document.getElementById("btn-manual-cycle").addEventListener("click", async () => {
  try {
    await fetch("/api/training/cycle", { method: "POST" });
    fetchStatus();
  } catch (e) {
    console.error(e);
  }
});

document.getElementById("btn-agent-cycle").addEventListener("click", async () => {
  try {
    await fetch("/api/agents/cycle", { method: "POST" });
    fetchStatus();
  } catch (e) {
    console.error(e);
  }
});

polling = setInterval(fetchStatus, POLL_INTERVAL);
fetchStatus();
