"use strict";

const API_URL = "http://127.0.0.1:5000/api/analyse";
const MAX_CHARS = 30000;
const TIMEOUT_MS = 120000; // LIME can take time on a laptop CPU.

const ui = {
  read: document.getElementById("read-email"),
  clear: document.getElementById("clear-email"),
  analyse: document.getElementById("analyse-email"),
  text: document.getElementById("email-text"),
  status: document.getElementById("status"),
  result: document.getElementById("result"),
  verdict: document.getElementById("verdict"),
  score: document.getElementById("score"),
  note: document.getElementById("explanation-note"),
  terms: document.getElementById("terms"),
};

function setStatus(message, kind = "") {
  ui.status.textContent = message;
  ui.status.className = `status ${kind}`.trim();
}

function setBusy(busy) {
  ui.read.disabled = busy;
  ui.clear.disabled = busy;
  ui.analyse.disabled = busy;
  ui.analyse.textContent = busy ? "Analysing…" : "Analyse email";
}

function clearResult() {
  ui.result.hidden = true;
  ui.terms.replaceChildren();
}

// This function runs inside the Gmail tab when the user clicks Read.
// Gmail may change its page structure, so manual paste is also supported.
function readVisibleGmailMessage() {
  const main = document.querySelector('[role="main"]') || document;
  const subject = main.querySelector("h2.hP")?.innerText?.trim() || "";

  const bodies = [...main.querySelectorAll(".a3s")].filter(
    element => element.getClientRects().length > 0 && element.innerText.trim()
  );

  const body = bodies.at(-1)?.innerText?.trim() || "";

  if (!body) {
    return {
      error: "No open Gmail message body was found. Open a message, or paste text manually.",
    };
  }

  return {
    text: [subject, body].filter(Boolean).join("\n\n"),
    messageCount: bodies.length,
  };
}

ui.read.addEventListener("click", async () => {
  clearResult();
  setStatus("Reading the open email…");

  try {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true,
    });

    if (!tab?.id || !/^https:\/\/mail\.google\.com\/mail\//.test(tab.url || "")) {
      throw new Error(
        "Open a Gmail message in the active tab, or paste email text manually."
      );
    }

    const injected = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: readVisibleGmailMessage,
    });

    const message = injected?.[0]?.result;

    if (!message || message.error) {
      throw new Error(message?.error || "Unable to read the open message.");
    }

    if (message.text.length > MAX_CHARS) {
      throw new Error(
        "Email is too long; paste a shorter de-identified sample."
      );
    }

    ui.text.value = message.text;

    setStatus(
      message.messageCount > 1
        ? "Loaded the last visible message in this conversation. Check the text before analysing."
        : "Open message loaded. Check the text before analysing.",
      "success"
    );
  } catch (error) {
    setStatus(error.message || "Could not read Gmail.", "error");
  }
});

ui.clear.addEventListener("click", () => {
  ui.text.value = "";
  clearResult();
  setStatus("Text cleared.");
  ui.text.focus();
});

function showResult(data) {
  if (
    !["phishing", "legitimate"].includes(data?.verdict) ||
    !Number.isFinite(Number(data.score))
  ) {
    throw new Error("The API returned an incomplete result.");
  }

  ui.result.hidden = false;

  const phishing = data.verdict === "phishing";
  ui.verdict.textContent = phishing
    ? "Potential phishing"
    : "Predicted legitimate";
  ui.verdict.className = `verdict ${data.verdict}`;
  ui.score.textContent = `${(Number(data.score) * 100).toFixed(1)}%`;

  const explanation = data.explanation;

  if (
    explanation?.method !== "LIME" ||
    !Array.isArray(explanation.terms)
  ) {
    ui.note.textContent =
      "A LIME explanation was not returned. Check the API before using this result in the demo.";
    ui.terms.replaceChildren();
    return;
  }

  ui.note.textContent =
    `LIME explains the ${data.verdict} verdict. ` +
    "Positive weights support it; negative weights oppose it. " +
    "Weights are not percentages.";

  ui.terms.replaceChildren();

  if (!explanation.terms.length) {
    const item = document.createElement("li");
    item.textContent = "No influential terms were returned for this message.";
    ui.terms.append(item);
  }

  for (const term of explanation.terms.slice(0, 8)) {
    const weight = Number(term.weight);

    if (typeof term.word !== "string" || !Number.isFinite(weight)) {
      continue;
    }

    const row = document.createElement("li");

    const word = document.createElement("span");
    word.className = "term";
    word.textContent = term.word;

    const value = document.createElement("span");
    value.className = weight >= 0 ? "positive" : "negative";
    value.textContent =
      `${weight >= 0 ? "+" : ""}${weight.toFixed(3)}`;

    row.append(word, value);
    ui.terms.append(row);
  }
}

ui.analyse.addEventListener("click", async () => {
  const text = ui.text.value.trim();
  clearResult();

  if (!text) {
    setStatus("Open a Gmail message or paste email text first.", "error");
    return;
  }

  if (text.length > MAX_CHARS) {
    setStatus("Text is too long for this demo.", "error");
    return;
  }

  setBusy(true);
  setStatus("Analysing locally. LIME may take a while…");

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `API error ${response.status}`);
    }

    showResult(data);
    setStatus("Analysis complete.", "success");
  } catch (error) {
    let message = error.message || "Analysis failed.";

    if (error.name === "AbortError") {
      message =
        "Analysis took too long. Check whether Flask is running and reduce LIME samples for testing.";
    } else if (error.message === "Failed to fetch") {
      message =
        "Cannot reach the local API. Start app.py and check the extension's localhost permission.";
    }

    setStatus(message, "error");
  } finally {
    clearTimeout(timeout);
    setBusy(false);
  }
});