// drives the chat UI: sends questions to POST /ask and renders the ChatbotResponse
const chat = document.getElementById("chat");
const welcome = document.getElementById("welcome");
const form = document.getElementById("form");
const input = document.getElementById("question");
const send = document.getElementById("send");

// one id per browser tab load, so every question in this tab shares conversation memory
// (the backend keys its LangGraph checkpointer on this) - a reload starts a fresh session
const sessionId = crypto.randomUUID();

// inline icons, so avatars render as crisp vector shapes instead of mismatched emoji/text
const ICONS = {
  user: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M12 12c2.7 0 4.9-2.2 4.9-4.9S14.7 2.2 12 2.2 7.1 4.4 7.1 7.1 9.3 12 12 12Zm0 2.4c-3.5 0-10.5 1.8-10.5 5.3v2.1h21v-2.1c0-3.5-7-5.3-10.5-5.3Z" fill="currentColor"/></svg>',
  bot: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M12 2 3 6.5V12c0 5 3.8 8.7 9 10 5.2-1.3 9-5 9-10V6.5L12 2Z" fill="currentColor"/></svg>',
};

// removes the welcome/suggestions screen the first time a message is sent
function clearWelcome() {
  if (welcome) welcome.remove();
}

// appends a user or assistant message row (avatar + a body column holding the text, and
// later the sources line stacked underneath it) and returns the whole row
function addMessage(text, role, extraClass) {
  const row = document.createElement("div");
  row.className = `message ${role} ${extraClass || ""}`.trim();

  const avatar = document.createElement("div");
  avatar.className = "message__avatar";
  avatar.innerHTML = role === "user" ? ICONS.user : ICONS.bot;

  const body = document.createElement("div");
  body.className = "message__body";

  const content = document.createElement("div");
  content.className = "message__content";
  content.textContent = text;
  body.appendChild(content);

  if (role === "user") {
    row.append(body, avatar);
  } else {
    row.append(avatar, body);
  }

  chat.appendChild(row);
  row.scrollIntoView({ behavior: "smooth", block: "end" });
  return row;
}

// shows a bouncing-dots row while waiting for the API, returns it so it can be removed later
function addTypingIndicator() {
  const row = addMessage("", "assistant");
  row.querySelector(".message__content").innerHTML = "<div class=\"typing\"><span></span><span></span><span></span></div>";
  return row;
}

async function askQuestion(question) {
  clearWelcome();
  addMessage(question, "user");
  send.disabled = true;

  const typingRow = addTypingIndicator();

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
    });

    typingRow.remove();

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      addMessage(detail.detail || `Request failed (${res.status})`, "assistant", "out-of-scope");
      return;
    }

    const data = await res.json();
    const row = addMessage(data.answer, "assistant", data.is_scope ? "" : "out-of-scope");
    // source_labels are pre-formatted server-side (e.g. "Chunk 45 (~Sec. 25)") - the "~"
    // matters: it's a best-effort section guess, not a confirmed citation, so the label
    // is never rendered as a bare "§N" that would misrepresent a chunk id as a section number
    if (data.source_labels && data.source_labels.length) {
      const sources = document.createElement("div");
      sources.className = "message__sources";
      sources.innerHTML =
        "Sources " + data.source_labels.map((label) => `<span class="source-pill">${label}</span>`).join("");
      row.querySelector(".message__body").appendChild(sources);
    }
  } catch (err) {
    typingRow.remove();
    addMessage("Something went wrong talking to the server.", "assistant", "out-of-scope");
  } finally {
    send.disabled = false;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  askQuestion(question);
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => askQuestion(button.textContent));
});
