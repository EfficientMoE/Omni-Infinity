const video = document.querySelector("#video");
const panel = document.querySelector("#prompt-panel");
const status = document.querySelector("#status");
const form = document.querySelector("#stream-form");
const keyActions = {
  ArrowUp: "forward",
  ArrowDown: "back",
  ArrowLeft: "left",
  ArrowRight: "right",
  " ": "jump",
};

let socket = null;
let sourceBuffer = null;
let appendQueue = [];
let cues = [];

function bytes(encoded) {
  const raw = atob(encoded);
  return Uint8Array.from(raw, (character) => character.charCodeAt(0));
}

function activeCue(items, time) {
  return items.find(
    (cue) => cue.pts <= time && time < cue.pts + cue.duration,
  );
}

function appendNext() {
  if (!sourceBuffer || sourceBuffer.updating || appendQueue.length === 0) {
    return;
  }
  sourceBuffer.appendBuffer(appendQueue.shift());
}

function renderCues() {
  panel.replaceChildren(
    ...cues.map((cue) => {
      const item = document.createElement("article");
      item.className = "cue";
      item.dataset.index = String(cue.index);
      const label = cue.instruction || cue.prompt;
      item.innerHTML = `<time>${cue.pts.toFixed(3)}s</time><p></p>`;
      item.querySelector("p").textContent = label;
      return item;
    }),
  );
}

function highlightCue() {
  const active = activeCue(cues, video.currentTime);
  for (const item of panel.querySelectorAll(".cue")) {
    item.classList.toggle(
      "active",
      active !== undefined && item.dataset.index === String(active.index),
    );
  }
}

function openMedia(codec, init) {
  const mediaSource = new MediaSource();
  video.src = URL.createObjectURL(mediaSource);
  mediaSource.addEventListener(
    "sourceopen",
    () => {
      sourceBuffer = mediaSource.addSourceBuffer(codec);
      sourceBuffer.addEventListener("updateend", appendNext);
      appendQueue.push(bytes(init));
      appendNext();
    },
    { once: true },
  );
}

function receive(message) {
  if (message.type === "init") {
    openMedia(message.codec, message.init_b64);
    return;
  }
  if (message.type === "chunk") {
    appendQueue.push(bytes(message.video_b64));
    cues.push(message);
    renderCues();
    appendNext();
    status.textContent = `chunk ${message.index}`;
    return;
  }
  if (message.type === "end") {
    status.textContent = "complete";
    return;
  }
  if (message.type === "error") {
    status.textContent = message.detail;
  }
}

async function createStream(event) {
  event.preventDefault();
  status.textContent = "opening";
  cues = [];
  renderCues();
  const response = await fetch("/v1/streams", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      type: "fl2va",
      prompt: document.querySelector("#prompt").value,
      optimizations: [],
      source: document.querySelector("#source").value,
    }),
  });
  if (!response.ok) {
    status.textContent = `HTTP ${response.status}`;
    return;
  }
  const { stream_id: streamId } = await response.json();
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(
    `${protocol}://${location.host}/v1/streams/${streamId}/ws`,
  );
  socket.addEventListener("open", () => {
    status.textContent = "connected";
  });
  socket.addEventListener("message", (item) => {
    receive(JSON.parse(item.data));
  });
  socket.addEventListener("close", () => {
    if (status.textContent !== "complete") {
      status.textContent = "closed";
    }
  });
}

function sendKey(event, down) {
  const action = keyActions[event.key];
  if (!action || !socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }
  event.preventDefault();
  socket.send(
    JSON.stringify({
      type: "input",
      key: event.key,
      down,
      action,
    }),
  );
}

form.addEventListener("submit", createStream);
video.addEventListener("timeupdate", highlightCue);
window.addEventListener("keydown", (event) => sendKey(event, true));
window.addEventListener("keyup", (event) => sendKey(event, false));
