const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const messages = document.getElementById("messages");
const showSources = document.getElementById("show-sources");
const healthStatus = document.getElementById("health-status");

function addBubble(text, role, sources = null) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;

  if (role === "bot" && Array.isArray(sources) && sources.length > 0) {
    const sourceDiv = document.createElement("div");
    sourceDiv.className = "sources";
    sourceDiv.textContent = sources
      .map((s) => `#${s.rank} ${s.category || ""} | ${s.chunk_id || ""}`)
      .join("\n");
    div.appendChild(sourceDiv);
  }

  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function addTyping() {
  const node = document.createElement("div");
  node.className = "typing";
  node.textContent = "Assistant est en train de repondre...";
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
    const ok = res.ok;
    healthStatus.textContent = ok ? "En ligne" : "Indisponible";
    healthStatus.style.color = ok ? "#0a7f72" : "#b42318";
  } catch (_) {
    healthStatus.textContent = "Indisponible";
    healthStatus.style.color = "#b42318";
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addBubble(question, "user");
  input.value = "";
  sendBtn.disabled = true;
  sendBtn.textContent = "Envoi...";

  const typing = addTyping();

  try {
    const data = await askQuestion(question);
    typing.remove();
    addBubble(data.answer || "Aucune reponse", "bot", data.sources || null);
  } catch (err) {
    typing.remove();
    addBubble(`Erreur: ${err.message}`, "bot");
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Envoyer";
    input.focus();
  }
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

checkHealth();
addBubble("Bonjour. Pose ta question, je reponds a partir du corpus local.", "bot");
