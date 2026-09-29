const video = document.querySelector("#video");
const panel = document.querySelector("#prompt-panel");
const status = document.querySelector("#status");
const form = document.querySelector("#stream-form");
const prompt = document.querySelector("#prompt");
const demoPrompt = [
  "[Shot 1] 2D-animated wide shot in an infinite black void. Thick grey smoke, black ink haze, and glowing red embers drift around a pale ronin assassin in layered black silk robes with red waist cords. A black katana rests at her hip. She stands still, right hand on the hilt, while cold white light cuts through the smoke. Low wind, ember crackle, distant taiko.",
  "[Shot 2] At 00:01.500 the camera pushes into a close-up. Her eyes narrow and she quickdraws the katana. A crimson slash tears the smoke. Robes and hair snap backward. A metallic scrape and a tearing whoosh.",
  "[Shot 3] At 00:03.000 a medium-wide shot pans as she leaps and spins. Overlapping crimson arcs scatter embers. Fabric snaps and the blade whistles.",
  "[Shot 4] At 00:04.200 she lands and cuts a jagged crimson arc that freezes the smoke and debris in slow motion. A deep impact boom rings out, then a sustained high tone over a shamisen hit.",
].join(" ");
const demoScript = [
  { t: 0.2, action: "hold", instruction: "Still stance in rolling smoke, hand on the katana." },
  { t: 1.2, action: "draw", instruction: "Eyes narrow; quickdraw leaves a crimson slash." },
  { t: 2.2, action: "dash", instruction: "Robes and hair snap back as she drives forward." },
  { t: 3.2, action: "leap", instruction: "Airborne spins scatter red embers." },
  { t: 4.2, action: "finish", instruction: "Finishing arc freezes smoke and debris." },
];
prompt.value = demoPrompt;
const keyActions = {
  ArrowUp: "forward",
  ArrowDown: "back",
  ArrowLeft: "left",
  ArrowRight: "right",
  " ": "jump",
};

let socket = null;
let mediaSource = null;
let sourceBuffer = null;
let appendQueue = [];
let cues = [];
let endRequested = false;

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
    if (
      endRequested &&
      sourceBuffer &&
      !sourceBuffer.updating &&
      mediaSource?.readyState === "open"
    ) {
      mediaSource.endOfStream();
    }
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
  mediaSource = new MediaSource();
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
    endRequested = true;
    appendNext();
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
  if (socket) {
    socket.close();
  }
  sourceBuffer = null;
  appendQueue = [];
  endRequested = false;
  cues = [];
  renderCues();
  const firstFrame = await fetch("/demo-first.png");
  if (!firstFrame.ok) {
    status.textContent = `HTTP ${firstFrame.status}`;
    return;
  }
  const firstFrameBytes = new Uint8Array(await firstFrame.arrayBuffer());
  let firstFrameBase64 = "";
  for (let offset = 0; offset < firstFrameBytes.length; offset += 0x8000) {
    firstFrameBase64 += String.fromCharCode(
      ...firstFrameBytes.subarray(offset, offset + 0x8000),
    );
  }
  const response = await fetch("/v1/streams", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      type: "fl2va",
      prompt: prompt.value,
      source: document.querySelector("#source").value,
      model_arch: "h3-dense",
      optimizations: [
        "adaln-host-cache",
        "block-stream",
        "text-encoder-stream",
      ],
      seed: 0,
      num_inference_steps: 8,
      resolution: "256p",
      num_frames: 120,
      first_frame_base64: btoa(firstFrameBase64),
      action_script: demoScript,
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
