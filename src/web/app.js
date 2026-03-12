const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const messages = document.getElementById("messages");
const showSources = document.getElementById("show-sources");
const healthStatus = document.getElementById("health-status");
const healthPill = document.getElementById("health-pill");
const docList = document.getElementById("doc-list");
const docSearch = document.getElementById("doc-search");

const documents = [
  {
    name: "Product Documentation.pdf",
    size: "2.4 MB",
    time: "2 hours ago",
    status: "ready",
    chunks: 156,
  },
  {
    name: "Technical Specifications.docx",
    size: "1.8 MB",
    time: "5 hours ago",
    status: "ready",
    chunks: 89,
  },
  {
    name: "User Manual.pdf",
    size: "3.2 MB",
    time: "1 day ago",
    status: "ready",
    chunks: 201,
  },
  {
    name: "API Reference.md",
    size: "512 KB",
    time: "2 days ago",
    status: "processing",
    chunks: 45,
  },
];

function statusLabel(status) {
  return status === "ready" ? "Ready" : "Processing";
}

function renderDocuments(filter = "") {
  const query = filter.trim().toLowerCase();
  const visible = documents.filter((doc) => doc.name.toLowerCase().includes(query));
  docList.innerHTML = "";

  visible.forEach((doc) => {
    const card = document.createElement("article");
    card.className = "doc-card";
    card.innerHTML = `
      <div class="doc-icon" aria-hidden="true"></div>
      <div class="doc-main">
        <h3>${doc.name}</h3>
        <div class="doc-meta">${doc.size} &nbsp;&bull;&nbsp; ${doc.time}</div>
        <div class="doc-status-row">
          <span class="status-badge ${doc.status}">${statusLabel(doc.status)}</span>
          <span class="chunk-meta">${doc.chunks} chunks</span>
        </div>
      </div>
      <div class="trash-btn" aria-hidden="true"></div>
    `;
    docList.appendChild(card);
  });
}

function formatTime() {
  return new Date().toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
  });
}

function scorePercent(score) {
  if (typeof score !== "number" || Number.isNaN(score)) {
    return 0;
  }
  return Math.max(8, Math.min(100, Math.round(score * 95)));
}

function buildSourcesMarkup(sources) {
  if (!Array.isArray(sources) || sources.length === 0) {
    return "";
  }

  const items = sources
    .map((source) => {
      const label = source.source_path
        ? source.source_path.split(/[/\\]/).pop()
        : source.category || `Source #${source.rank}`;
      const score = scorePercent(source.score);
      const preview = source.preview || "";

      return `
        <div class="source-item">
          <div class="source-main">
            <span class="source-chip">${label}</span>
            <span class="source-rank">Rank ${source.rank}${source.chunk_id ? ` • ${source.chunk_id}` : ""}</span>
            <div class="source-preview">${preview}</div>
          </div>
          <div class="source-score">
            ${score}%
            <div class="score-bar"><div class="score-fill" style="width:${score}%"></div></div>
          </div>
        </div>
      `;
    })
    .join("");

  return `
    <div class="sources-panel">
      <div class="sources-title">Sources</div>
      ${items}
    </div>
  `;
}

function addMessage({ text, role, sources = null, time = formatTime() }) {
  const row = document.createElement("section");
  row.className = `message-row ${role}`;

  const avatarClass = role === "user" ? "user" : "bot";
  row.innerHTML = `
    <div class="bubble-wrap">
      <div class="avatar ${avatarClass}" aria-hidden="true"></div>
      <div class="bubble">${text}${role === "bot" ? buildSourcesMarkup(sources) : ""}</div>
    </div>
    <div class="message-meta">${time}</div>
  `;

  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
  return row;
}

function addTyping() {
  const node = document.createElement("div");
  node.className = "typing";
  node.textContent = "Assistant is analyzing your knowledge base...";
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
  return node;
}

async function askQuestion(question) {
  const payload = {
    question,
    index_dir: "data/index_business",
    ollama_model: "llama3.1:8b",
    top_k: 5,
    include_sources: showSources.checked,
    strict_answer: true,
  };

  const response = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Erreur API (${response.status}) ${err}`);
  }

  return response.json();
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    if (!res.ok) {
      throw new Error("Health check failed");
    }
    healthStatus.textContent = "System Active";
    healthPill.classList.remove("offline");
  } catch (_) {
    healthStatus.textContent = "System Offline";
    healthPill.classList.add("offline");
  }
}

function autoResizeTextarea() {
  input.style.height = "0px";
  input.style.height = `${Math.min(input.scrollHeight, 220)}px`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage({ text: question, role: "user" });
  input.value = "";
  autoResizeTextarea();
  sendBtn.disabled = true;

  const typing = addTyping();

  try {
    const data = await askQuestion(question);
    typing.remove();
    addMessage({
      text: data.answer || "No answer returned.",
      role: "bot",
      sources: data.sources || null,
    });
  } catch (error) {
    typing.remove();
    addMessage({ text: `Erreur: ${error.message}`, role: "bot" });
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});

input.addEventListener("input", autoResizeTextarea);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

docSearch.addEventListener("input", (event) => {
  renderDocuments(event.target.value);
});

renderDocuments();
checkHealth();
autoResizeTextarea();
addMessage({
  text: "Bonjour. Pose une question sur tes documents et je reponds a partir de la base RAG locale.",
  role: "bot",
  time: "Now",
});
