const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const messages = document.getElementById("messages");
const showSources = document.getElementById("show-sources");
const healthStatus = document.getElementById("health-status");
const healthPill = document.getElementById("health-pill");
const docList = document.getElementById("doc-list");
const docSearch = document.getElementById("doc-search");
const uploadBtn = document.getElementById("upload-btn");
const fileInput = document.getElementById("file-input");
const docCategory = document.getElementById("doc-category");
const uploadFeedback = document.getElementById("upload-feedback");
const summaryBtn = document.getElementById("summary-btn");
const flashcardsBtn = document.getElementById("flashcards-btn");
const quizBtn = document.getElementById("quiz-btn");
const engineMode = document.getElementById("engine-mode");

let documents = [];
let selectedDocIds = [];

function statusLabel(status) {
  if (status === "ready") return "Ready";
  if (status === "empty") return "Empty";
  return "Processing";
}

function relativeTime(isoDate) {
  if (!isoDate) return "unknown";
  const then = new Date(isoDate);
  const diffMs = Date.now() - then.getTime();
  const diffHours = Math.floor(diffMs / 3600000);
  if (diffHours < 1) return "just now";
  if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? "s" : ""} ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays} day${diffDays > 1 ? "s" : ""} ago`;
}

function renderDocuments(filter = "") {
  const query = filter.trim().toLowerCase();
  const visible = documents.filter((doc) => {
    const haystack = `${doc.name} ${doc.category}`.toLowerCase();
    return haystack.includes(query);
  });

  docList.innerHTML = "";
  visible.forEach((doc) => {
    const isSelected = selectedDocIds.includes(doc.doc_id);
    const card = document.createElement("article");
    card.className = `doc-card ${isSelected ? "selected" : ""}`;
    card.dataset.docId = doc.doc_id;
    card.innerHTML = `
      <div class="doc-icon" aria-hidden="true"></div>
      <div class="doc-main">
        <h3>${doc.name}</h3>
        <div class="doc-meta">${doc.size_label || "0 KB"} &nbsp;&bull;&nbsp; ${relativeTime(doc.uploaded_at)}</div>
        <div class="doc-status-row">
          <span class="status-badge ${doc.status}">${statusLabel(doc.status)}</span>
          <span class="chunk-meta">${doc.chunks} chunks</span>
          <span class="doc-tag">${doc.category}</span>
        </div>
      </div>
      <div class="select-indicator">${isSelected ? "•" : ""}</div>
    `;
    card.addEventListener("click", () => toggleDocumentSelection(doc.doc_id));
    docList.appendChild(card);
  });

  if (visible.length === 0) {
    docList.innerHTML = `<div class="empty-state">Aucun document trouvé. Upload tes fichiers de cours ou de projet.</div>`;
  }
}

function toggleDocumentSelection(docId) {
  if (selectedDocIds.includes(docId)) {
    selectedDocIds = selectedDocIds.filter((id) => id !== docId);
  } else {
    selectedDocIds = [docId];
  }
  renderDocuments(docSearch.value);
}

function formatTime() {
  return new Date().toLocaleTimeString("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function scorePercent(score) {
  if (typeof score !== "number" || Number.isNaN(score)) {
    return 0;
  }
  return Math.max(8, Math.min(100, Math.round(score * 95)));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
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
            <span class="source-chip">${escapeHtml(label)}</span>
            <span class="source-rank">Rank ${source.rank}${source.chunk_id ? ` • ${escapeHtml(source.chunk_id)}` : ""}</span>
            <div class="source-preview">${escapeHtml(preview)}</div>
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
}

function addTyping(text = "Analyse en cours...") {
  const node = document.createElement("div");
  node.className = "typing";
  node.textContent = text;
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
  return node;
}

async function fetchLibrary() {
  const response = await fetch("/api/library");
  if (!response.ok) {
    throw new Error("Impossible de charger la bibliothèque.");
  }
  const data = await response.json();
  documents = data.documents || [];
  renderDocuments(docSearch.value);
}

async function askWorkspace(question) {
  const endpoint = "/api/workspace/ask/haystack";
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      top_k: 8,
      include_sources: showSources.checked,
      doc_ids: selectedDocIds,
      anti_hallucination: true,
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Erreur API (${response.status}) ${err}`);
  }

  return response.json();
}

async function callStudyEndpoint(endpoint) {
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_ids: selectedDocIds }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Erreur API (${response.status}) ${err}`);
  }

  return response.json();
}

function renderSummary(data) {
  const bullets = (data.bullets || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  return `
    <strong>${escapeHtml(data.title)}</strong>

    ${escapeHtml(data.paragraph || "")}

    <ul>${bullets}</ul>
  `;
}

function renderFlashcards(data) {
  const cards = (data.flashcards || []).map((card, index) => {
    return `<div class="study-card"><strong>Fiche ${index + 1}</strong><br>${escapeHtml(card.question)}<br><br>${escapeHtml(card.answer)}</div>`;
  });
  return cards.join("");
}

function renderQuiz(data) {
  const blocks = (data.quiz || []).map((item) => {
    const options = item.options.map((option, index) => `<li>${String.fromCharCode(65 + index)}. ${escapeHtml(option)}</li>`).join("");
    return `<div class="study-card"><strong>Question ${item.id}</strong><br>${escapeHtml(item.question)}<ul>${options}</ul><em>Bonne réponse: ${escapeHtml(item.answer)}</em></div>`;
  });
  return blocks.join("");
}

async function uploadFiles(files) {
  if (!files.length) return;
  uploadFeedback.textContent = "Upload en cours...";

  for (const file of files) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("category", docCategory.value);

    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`${file.name}: ${err}`);
    }
  }

  uploadFeedback.textContent = "Documents ajoutés à ton espace personnel.";
  await fetchLibrary();
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

  addMessage({ text: escapeHtml(question), role: "user" });
  input.value = "";
  autoResizeTextarea();
  sendBtn.disabled = true;

  const typing = addTyping("Recherche via Haystack...");

  try {
    const data = await askWorkspace(question);
    typing.remove();
    addMessage({
      text: escapeHtml(data.answer || "Aucune réponse."),
      role: "bot",
      sources: data.sources || null,
    });
  } catch (error) {
    typing.remove();
    addMessage({ text: `Erreur: ${escapeHtml(error.message)}`, role: "bot" });
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async (event) => {
  try {
    await uploadFiles([...event.target.files]);
  } catch (error) {
    uploadFeedback.textContent = `Erreur upload: ${error.message}`;
  } finally {
    fileInput.value = "";
  }
});

summaryBtn.addEventListener("click", async () => {
  const typing = addTyping("Génération du résumé...");
  try {
    const data = await callStudyEndpoint("/api/study/summary");
    typing.remove();
    addMessage({ text: renderSummary(data), role: "bot", time: "Maintenant" });
  } catch (error) {
    typing.remove();
    addMessage({ text: `Erreur: ${escapeHtml(error.message)}`, role: "bot" });
  }
});

flashcardsBtn.addEventListener("click", async () => {
  const typing = addTyping("Création des fiches de révision...");
  try {
    const data = await callStudyEndpoint("/api/study/flashcards");
    typing.remove();
    addMessage({ text: renderFlashcards(data), role: "bot", time: "Maintenant" });
  } catch (error) {
    typing.remove();
    addMessage({ text: `Erreur: ${escapeHtml(error.message)}`, role: "bot" });
  }
});

quizBtn.addEventListener("click", async () => {
  const typing = addTyping("Construction du quiz...");
  try {
    const data = await callStudyEndpoint("/api/study/quiz");
    typing.remove();
    addMessage({ text: renderQuiz(data), role: "bot", time: "Maintenant" });
  } catch (error) {
    typing.remove();
    addMessage({ text: `Erreur: ${escapeHtml(error.message)}`, role: "bot" });
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

async function init() {
  await fetchLibrary();
  await checkHealth();
  autoResizeTextarea();
  addMessage({
    text: "Bonjour. Upload tes documents de cours, de projet ou de révision directement ici. Ensuite tu peux poser des questions, générer un résumé, créer des fiches ou lancer un quiz.",
    role: "bot",
    time: "Now",
  });
}

init().catch((error) => {
  addMessage({ text: `Erreur initialisation: ${escapeHtml(error.message)}`, role: "bot" });
});
