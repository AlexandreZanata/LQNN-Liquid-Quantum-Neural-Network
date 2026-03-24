const statusEl = document.getElementById("status");
const canvas = document.getElementById("network-canvas");
const ctx = canvas.getContext("2d");
const metricsEl = document.getElementById("metrics");
const branchesEl = document.getElementById("branches");

const textInput = document.getElementById("text-input");
const stimulateBtn = document.getElementById("stimulate-btn");
const sleepBtn = document.getElementById("sleep-btn");
const resetBtn = document.getElementById("reset-btn");
const quantumBtn = document.getElementById("quantum-btn");
const autoToggle = document.getElementById("auto-toggle");

let ws = null;
let latestState = null;

function connect() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${window.location.host}/ws`);

  ws.onopen = () => {
    statusEl.textContent = "Conectado (stream ativo)";
    statusEl.style.color = "#2a9d8f";
  };

  ws.onclose = () => {
    statusEl.textContent = "Desconectado. Reconectando...";
    statusEl.style.color = "#e76f51";
    setTimeout(connect, 1500);
  };

  ws.onerror = () => {
    statusEl.textContent = "Erro no WebSocket";
    statusEl.style.color = "#e76f51";
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "error") {
        statusEl.textContent = data.message;
        statusEl.style.color = "#e76f51";
        return;
      }
      latestState = data;
      render(data);
    } catch (err) {
      statusEl.textContent = "Falha ao decodificar estado";
      statusEl.style.color = "#e76f51";
    }
  };
}

function sendAction(payload) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    return;
  }
  ws.send(JSON.stringify(payload));
}

function colorForNeuron(neuron) {
  if (neuron.energy < 0.2) {
    return "#df4d4d";
  }
  if (neuron.activation_probability > 0.6) {
    return "#1ea36b";
  }
  return "#2e86c1";
}

function render(state) {
  const { neurons, synapses, metrics, quantum } = state;
  const scaleX = canvas.width / 100;
  const scaleY = canvas.height / 100;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const syn of synapses) {
    const pre = neurons.find((n) => n.id === syn.pre_id);
    const post = neurons.find((n) => n.id === syn.post_id);
    if (!pre || !post) continue;

    const alpha = 0.08 + syn.weight * 0.65;
    ctx.strokeStyle = syn.strong ? `rgba(95, 138, 255, ${alpha})` : `rgba(87, 111, 139, ${alpha})`;
    ctx.lineWidth = 0.5 + syn.weight * 3.2;
    ctx.beginPath();
    ctx.moveTo(pre.x * scaleX, pre.y * scaleY);
    ctx.lineTo(post.x * scaleX, post.y * scaleY);
    ctx.stroke();
  }

  for (const neuron of neurons) {
    const x = neuron.x * scaleX;
    const y = neuron.y * scaleY;
    const radius = 3 + neuron.activation_probability * 8;
    const color = colorForNeuron(neuron);

    ctx.beginPath();
    ctx.fillStyle = color;
    if (neuron.activation_probability > 0.75) {
      ctx.shadowBlur = 16;
      ctx.shadowColor = color;
    } else {
      ctx.shadowBlur = 0;
    }
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.shadowBlur = 0;

  metricsEl.innerHTML = "";
  const metricItems = [
    `tick: ${metrics.tick}`,
    `neuronios: ${metrics.neuron_count}`,
    `sinapses: ${metrics.synapse_count}`,
    `sinapses fortes: ${metrics.strong_synapses}`,
    `peso medio: ${metrics.mean_weight.toFixed(3)}`,
    `energia media: ${metrics.mean_energy.toFixed(3)}`,
    `energia min-max: ${metrics.min_energy.toFixed(3)} .. ${metrics.max_energy.toFixed(3)}`,
    `razao ativos: ${(metrics.active_ratio * 100).toFixed(1)}%`,
    `modo quantico: ${quantum.enabled ? "on" : "off"}`,
  ];
  for (const item of metricItems) {
    const li = document.createElement("li");
    li.textContent = item;
    metricsEl.appendChild(li);
  }

  branchesEl.innerHTML = "";
  if (quantum.branch_energies.length === 0) {
    const li = document.createElement("li");
    li.textContent = "Sem dados de ramos";
    branchesEl.appendChild(li);
  } else {
    quantum.branch_energies.forEach((energy, idx) => {
      const li = document.createElement("li");
      const selected = idx === quantum.selected_branch ? " <- selecionado" : "";
      li.textContent = `ramo ${idx}: ${energy.toFixed(4)}${selected}`;
      branchesEl.appendChild(li);
    });
  }
}

stimulateBtn.addEventListener("click", () => {
  sendAction({ action: "stimulate", text: textInput.value.trim() });
});

sleepBtn.addEventListener("click", () => {
  sendAction({ action: "sleep", cycles: 1 });
});

resetBtn.addEventListener("click", () => {
  sendAction({ action: "reset" });
});

quantumBtn.addEventListener("click", () => {
  sendAction({ action: "toggle_quantum" });
});

autoToggle.addEventListener("change", () => {
  sendAction({ action: "toggle_auto", enabled: autoToggle.checked });
});

connect();
