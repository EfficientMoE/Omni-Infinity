const video = document.querySelector("#video");
const panel = document.querySelector("#prompt-panel");
const status = document.querySelector("#status");
const form = document.querySelector("#stream-form");
const prompt = document.querySelector("#prompt");
const source = document.querySelector("#source");
const duration = document.querySelector("#duration");
const promptStack = document.querySelector("#prompt-stack");
const frameChip = document.querySelector("#frame-chip");
const artifactLink = document.querySelector("#artifact");
const cueCount = document.querySelector("#cue-count");
const demoShots = [
  "[Shot 1] 2D-animated wide shot in an infinite black void. Thick grey smoke, black ink haze, and glowing red embers drift around a pale ronin assassin in layered black silk robes with red waist cords. A black katana rests at her hip. She stands still, right hand on the hilt, while cold white light cuts through the smoke. Low wind, ember crackle, distant taiko.",
  "[Shot 2] At 00:01.500 the camera pushes into a close-up. Her eyes narrow and she quickdraws the katana. A crimson slash tears the smoke. Robes and hair snap backward. A metallic scrape and a tearing whoosh.",
  "[Shot 3] At 00:03.000 a medium-wide shot pans as she leaps and spins. Overlapping crimson arcs scatter embers. Fabric snaps and the blade whistles.",
  "[Shot 4] At 00:04.200 she lands and cuts a jagged crimson arc that freezes the smoke and debris in slow motion. A deep impact boom rings out, then a sustained high tone over a shamisen hit.",
];
const demoPrompt = demoShots.join(" ");
const SEGMENT_SECONDS = 124 / 24;
const DEMO_CLIPS = { "15s": 3, "1min": 12, "2min": 24, "5min": 59 };
const DEMO_SECONDS = { "15s": 15, "1min": 60, "2min": 120, "5min": 300 };
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
let generation = 0;
let activeIndex = null;
let schedule = [];
let modelRunning = false;
let playbackStarted = false;

function bytes(encoded) {
  const raw = atob(encoded);
  return Uint8Array.from(raw, (character) => character.charCodeAt(0));
}

function activeCue(items, time) {
  return items.find(
    (cue) => cue.pts <= time && time < cue.pts + cue.duration,
  );
}

function streamState(text) {
  if (text === "offline") return "idle";
  if (text === "opening") return "connecting";
  if (text === "connected") return "waiting";
  if (text.startsWith("chunk ") || text.startsWith("live")) return "live";
  if (text === "complete") return "done";
  return "alert";
}

function setStatus(text) {
  status.textContent = text;
  document.body.dataset.state = streamState(text);
}

// Resubmission replaces the MediaSource; a detached buffer must not append.
function bufferIsLive() {
  return (
    mediaSource &&
    sourceBuffer &&
    Array.prototype.includes.call(mediaSource.sourceBuffers, sourceBuffer)
  );
}

function appendNext() {
  if (!sourceBuffer || sourceBuffer.updating || appendQueue.length === 0) {
    if (
      endRequested &&
      sourceBuffer &&
      !sourceBuffer.updating &&
      mediaSource &&
      mediaSource.readyState === "open"
    ) {
      try {
        mediaSource.endOfStream();
      } catch (error) {
        /* already ended, or the MediaSource was replaced — ignore */
      }
    }
    return;
  }
  if (!bufferIsLive()) {
    return;
  }
  const segment = appendQueue.shift();
  try {
    sourceBuffer.appendBuffer(segment);
  } catch (error) {
    /* stale SourceBuffer from a superseded stream — drop the append */
  }
}

function renderCues() {
  panel.replaceChildren(
    ...cues.map((cue) => {
      const item = document.createElement("article");
      item.className = "cue";
      if (cue.ready === false) {
        item.classList.add("rendering");
      }
      item.dataset.index = String(cue.index);
      const label = cue.instruction || cue.prompt;
      const stamp = document.createElement("time");
      stamp.textContent = `${cue.pts.toFixed(3)}s`;
      const body = document.createElement("p");
      body.textContent = label;
      item.append(stamp, body);
      return item;
    }),
  );
  if (cueCount) {
    cueCount.textContent = cues.length
      ? `${String(cues.length).padStart(2, "0")} cues`
      : "";
  }
  highlightCue();
}

function highlightCue() {
  const active = activeCue(schedule.length ? schedule : cues, video.currentTime);
  for (const item of panel.querySelectorAll(".cue")) {
    item.classList.toggle(
      "active",
      active !== undefined && item.dataset.index === String(active.index),
    );
  }
  const nextIndex = active === undefined ? null : active.index;
  if (nextIndex !== activeIndex) {
    activeIndex = nextIndex;
    if (nextIndex !== null) {
      const target = panel.querySelector(`.cue[data-index="${nextIndex}"]`);
      if (target) {
        target.scrollIntoView({ block: "nearest" });
      }
    }
  }
}

function openMedia(codec, init) {
  const myGeneration = generation;
  const media = new MediaSource();
  mediaSource = media;
  video.src = URL.createObjectURL(media);
  media.addEventListener(
    "sourceopen",
    () => {
      if (myGeneration !== generation) {
        return;
      }
      sourceBuffer = media.addSourceBuffer(codec);
      sourceBuffer.addEventListener("updateend", () => {
        if (myGeneration !== generation) {
          return;
        }
        appendNext();
      });
      // Server sends init + every chunk + end in one burst, so chunk
      // payloads may already be queued. The init segment must be appended
      // first — put it at the head of the queue before draining.
      appendQueue.unshift(bytes(init));
      appendNext();
      startPlayback();
    },
    { once: true },
  );
}

function startPlayback() {
  if (playbackStarted) {
    return;
  }
  playbackStarted = true;
  const pending = video.play();
  if (pending && pending.catch) {
    pending.catch(() => {});
  }
}

function slipValue(index) {
  return demoShots[index] || `Clip ${index + 1}. Continue the same scene and subject.`;
}

function renderPromptStack() {
  const count = DEMO_CLIPS[duration.value];
  const previous = [...promptStack.querySelectorAll("textarea")].map(
    (node) => node.value,
  );
  promptStack.replaceChildren();
  for (let index = 0; index < count; index += 1) {
    const label = document.createElement("label");
    label.className = "slip";
    const marker = document.createElement("span");
    marker.className = "slip__index";
    marker.textContent = String(index + 1).padStart(2, "0");
    const field = document.createElement("textarea");
    field.rows = 2;
    field.required = true;
    field.spellcheck = false;
    field.value = previous[index] || slipValue(index);
    field.setAttribute("aria-label", `Prompt ${index + 1}`);
    label.append(marker, field);
    promptStack.append(label);
  }
  frameChip.textContent = `256p / 124f × ${count}`;
}

function syncMode() {
  form.dataset.mode = source.value;
  const demo = source.value === "demo";
  prompt.disabled = demo;
  prompt.required = !demo;
  if (demo) {
    renderPromptStack();
    return;
  }
  frameChip.textContent = "256p / 120f";
}

function armSchedule(prompts, seconds) {
  schedule = prompts.map((text, index) => {
    const pts = index * SEGMENT_SECONDS;
    const end = Math.min((index + 1) * SEGMENT_SECONDS, seconds);
    return {
      index,
      pts,
      duration: end - pts,
      prompt: text,
      instruction: null,
      ready: false,
    };
  });
  cues = schedule;
  renderCues();
}

function receive(message) {
  if (message.type === "init") {
    openMedia(message.codec, message.init_b64);
    return;
  }
  if (message.type === "chunk") {
    appendQueue.push(bytes(message.video_b64));
    if (schedule.length) {
      const row = schedule.find(
        (item) => item.pts <= message.pts && message.pts < item.pts + item.duration,
      );
      if (row) {
        row.ready = true;
      }
      cues = schedule;
    } else {
      cues.push(message);
    }
    renderCues();
    appendNext();
    startPlayback();
    setStatus(
      schedule.length && modelRunning
        ? "live · model running"
        : `chunk ${message.index}`,
    );
    return;
  }
  if (message.type === "end") {
    endRequested = true;
    modelRunning = false;
    // Latch the terminal status first; a late append/flush must not stop
    // the monitor from reading "complete".
    setStatus("complete");
    if (message.artifact_url && artifactLink) {
      artifactLink.href = message.artifact_url;
      artifactLink.hidden = false;
    }
    appendNext();
    return;
  }
  if (message.type === "error") {
    setStatus(message.detail);
  }
}

async function createStream(event) {
  event.preventDefault();
  // Drop focus off the submit button so the Space "jump" key can never
  // re-trigger a hidden submit once the stream has ended.
  if (document.activeElement && document.activeElement.blur) {
    document.activeElement.blur();
  }
  generation += 1;
  setStatus("opening");
  if (socket) {
    socket.close();
  }
  sourceBuffer = null;
  appendQueue = [];
  endRequested = false;
  cues = [];
  schedule = [];
  modelRunning = false;
  playbackStarted = false;
  activeIndex = null;
  renderCues();
  if (artifactLink) {
    artifactLink.hidden = true;
    artifactLink.removeAttribute("href");
  }
  const firstFrame = await fetch("/demo-first.png");
  if (!firstFrame.ok) {
    setStatus(`HTTP ${firstFrame.status}`);
    return;
  }
  const firstFrameBytes = new Uint8Array(await firstFrame.arrayBuffer());
  let firstFrameBase64 = "";
  for (let offset = 0; offset < firstFrameBytes.length; offset += 0x8000) {
    firstFrameBase64 += String.fromCharCode(
      ...firstFrameBytes.subarray(offset, offset + 0x8000),
    );
  }
  const demo = source.value === "demo";
  const prompts = demo
    ? [...promptStack.querySelectorAll("textarea")].map((node) => node.value)
    : [];
  if (demo) {
    armSchedule(prompts, DEMO_SECONDS[duration.value]);
    modelRunning = true;
  }
  const response = await fetch(demo ? "/v1/demos" : "/v1/streams", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(
      demo
        ? {
            prompts,
            duration: duration.value,
            model_arch: "h3-dense",
            optimizations: [
              "adaln-host-cache",
              "block-stream",
              "text-encoder-stream",
            ],
            seed: 0,
            num_inference_steps: 8,
            resolution: "256p",
            first_frame_base64: btoa(firstFrameBase64),
          }
        : {
            type: "fl2va",
            prompt: prompt.value,
            source: source.value,
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
          },
    ),
  });
  if (!response.ok) {
    modelRunning = false;
    setStatus(`HTTP ${response.status}`);
    return;
  }
  const payload = await response.json();
  const streamId = payload.stream_id || payload.id;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const connection = new WebSocket(
    `${protocol}://${location.host}${demo ? "/v1/demos" : "/v1/streams"}/${streamId}/ws`,
  );
  socket = connection;
  connection.addEventListener("open", () => {
    setStatus("connected");
  });
  connection.addEventListener("message", (item) => {
    receive(JSON.parse(item.data));
  });
  connection.addEventListener("close", () => {
    if (socket === connection && status.textContent !== "complete") {
      setStatus("closed");
    }
  });
}

function sendKey(event, down) {
  // Never steal keys or preventDefault while the prompt or source has focus.
  const focused = document.activeElement;
  if (
    focused === prompt ||
    focused === source ||
    focused === duration ||
    (promptStack && promptStack.contains(focused))
  ) {
    return;
  }
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

setStatus("offline");
source.addEventListener("change", syncMode);
duration.addEventListener("change", () => {
  if (source.value === "demo") {
    renderPromptStack();
  }
});
form.addEventListener("submit", createStream);
video.addEventListener("timeupdate", highlightCue);
window.addEventListener("keydown", (event) => sendKey(event, true));
window.addEventListener("keyup", (event) => sendKey(event, false));
