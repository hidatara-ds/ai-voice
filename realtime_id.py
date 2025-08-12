# -*- coding: utf-8 -*-
# app_voice_web.py — Localhost web UI for Indonesian real-time STT (Deepgram) + VAD logs
# Requirements: flask, deepgram-sdk, sounddevice, numpy

import os
import sys
import json
import time
import wave
import queue
import threading
import traceback
from collections import deque

import numpy as np
import sounddevice as sd
from flask import Flask, Response, request, jsonify, render_template_string

from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

# ---------- Simple PubSub for SSE ----------
class PubSub:
    def __init__(self):
        self.subs = set()
        self.lock = threading.Lock()

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            self.subs.add(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.subs.discard(q)

    def publish(self, msg):
        with self.lock:
            targets = list(self.subs)
        for q in targets:
            try:
                q.put(msg, timeout=0.01)
            except Exception:
                pass

pubsub = PubSub()

def emit(msg, level="info"):
    # msg can be str or dict; wrap to JSON-serializable
    if isinstance(msg, dict):
        data = {"level": level, **msg}
    else:
        data = {"level": level, "msg": str(msg)}
    pubsub.publish(data)
    # Also print to console for debugging
    if level == "error":
        print("[ERROR]", data, file=sys.stderr)
    else:
        print("[LOG]", data)

# ---------- Voice Worker (Deepgram + VAD) ----------
class VoiceWorker:
    def __init__(self):
        self.thread = None
        self.stop_evt = threading.Event()
        self.running = False
        self.state_lock = threading.Lock()
        self.last_config = {}

    def is_running(self):
        with self.state_lock:
            return self.running

    def start(self, device=None, rate=48000, model="nova-2", lang="id",
              vad_th=None, save_wav=False, rms_spam=False):
        with self.state_lock:
            if self.running:
                emit("Already running.", "warn")
                return False
            self.running = True

        self.stop_evt.clear()
        self.thread = threading.Thread(
            target=self._run,
            kwargs=dict(
                device=device, rate=rate, model=model, lang=lang,
                vad_th=vad_th, save_wav=save_wav, rms_spam=rms_spam
            ),
            daemon=True
        )
        self.thread.start()
        self.last_config = {
            "device": device, "rate": rate, "model": model,
            "lang": lang, "vad_th": vad_th, "save_wav": save_wav,
            "rms_spam": rms_spam
        }
        return True

    def stop(self):
        self.stop_evt.set()
        emit("Stopping…", "info")
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        with self.state_lock:
            self.running = False
        emit("Stopped.", "info")

    def _run(self, device, rate, model, lang, vad_th, save_wav, rms_spam):
        # Check API key
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            emit("DEEPGRAM_API_KEY belum diset. (CMD) set DEEPGRAM_API_KEY=dg_xxx...", "error")
            with self.state_lock:
                self.running = False
            return

        # Configure audio device
        try:
            if device is not None:
                sd.default.device = (device, None)
                emit(f"Use input device index: {device}", "info")
        except Exception as e:
            emit(f"Failed to set device: {e}", "error")

        emit(f"Sample rate: {rate}, Model: {model}, Lang: {lang}", "info")

        # Init Deepgram
        try:
            dg = DeepgramClient(api_key)
            conn = dg.listen.websocket.v("1")
        except Exception as e:
            emit(f"Gagal init Deepgram: {e}", "error")
            with self.state_lock:
                self.running = False
            return

        def first_arg_or_kw(args, kwargs, key="result"):
            r = kwargs.get(key)
            if r is None and args:
                r = args[0]
            return r

        # Event handlers
        def on_open(*a, **k):
            emit("WS OPEN", "info")

        def on_close(*a, **k):
            emit("WS CLOSE", "info")

        def on_error(*a, **k):
            info = first_arg_or_kw(a, k) or k or a
            try:
                s = json.dumps(info, default=str)[:800]
            except Exception:
                s = str(info)
            emit({"event": "WS_ERROR", "detail": s}, "error")

        def on_metadata(*a, **k):
            meta = first_arg_or_kw(a, k) or k or a
            try:
                s = json.dumps(meta, default=str)[:800]
            except Exception:
                s = str(meta)
            emit({"event": "METADATA", "detail": s}, "info")

        def on_transcript(*a, **k):
            result = first_arg_or_kw(a, k)
            if not result:
                return
            try:
                alt = result.channel.alternatives[0]
                if alt.transcript:
                    final = getattr(result, "is_final", False)
                    emit({"event": "TRANSCRIPT", "final": bool(final), "text": alt.transcript}, "info")
            except Exception as e:
                emit(f"Transcript handler error: {e}", "error")

        # Register events
        conn.on(LiveTranscriptionEvents.Open, on_open)
        conn.on(LiveTranscriptionEvents.Close, on_close)
        conn.on(LiveTranscriptionEvents.Error, on_error)
        conn.on(LiveTranscriptionEvents.Metadata, on_metadata)
        conn.on(LiveTranscriptionEvents.Transcript, on_transcript)

        # Start Deepgram
        try:
            opts = LiveOptions(
                model=model,
                language=lang,
                encoding="linear16",
                sample_rate=rate,
                channels=1,
            )
            conn.start(opts)
        except Exception as e:
            emit(f"Start Deepgram failed: {e}", "error")
            with self.state_lock:
                self.running = False
            return

        # Audio capture + VAD (event-driven)
        q = queue.Queue()
        wav = None
        if save_wav:
            try:
                wav = wave.open("debug_capture.wav", "wb")
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(rate)
                emit("Saving raw audio to debug_capture.wav", "info")
            except Exception as e:
                emit(f"Open WAV failed: {e}", "error")

        vad_state = "silence"   # "silence" | "speech"
        speech_start_ts = None

        calib_until = time.time() + 0.7
        calib_vals = []
        base_th = vad_th  # manual override if provided
        TH_HYST = 0.7
        MA_N = 5
        rms_hist = deque(maxlen=MA_N)

        def on_audio(indata, frames, time_info, status):
            if status:
                emit(f"[AUDIO status] {status}", "warn")
            q.put(bytes(indata))

        emit("Diam 0.7s untuk kalibrasi noise… lalu bicara natural.", "info")

        stream = None
        try:
            stream = sd.RawInputStream(
                samplerate=rate, channels=1, dtype="int16",
                callback=on_audio, blocksize=0, device=device
            )
            stream.start()

            while not self.stop_evt.is_set():
                try:
                    chunk = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if not chunk:
                    continue

                if wav:
                    try:
                        wav.writeframes(chunk)
                    except Exception:
                        pass

                arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
                if rms_spam:
                    emit({"event": "RMS", "value": round(rms, 1)}, "info")

                now = time.time()
                if base_th is None and now <= calib_until:
                    calib_vals.append(rms)
                    conn.send(chunk)
                    continue
                elif base_th is None and calib_vals:
                    base_noise = float(np.median(calib_vals)) if calib_vals else 0.0
                    base_th = max(300.0, base_noise * 3.0 + 200.0)
                    emit({"event": "VAD_THRESHOLD_AUTO", "threshold": round(base_th, 1), "noise": round(base_noise, 1)}, "info")

                rms_hist.append(rms)
                rms_smooth = float(np.mean(rms_hist)) if rms_hist else rms

                th_on = base_th
                th_off = base_th * TH_HYST if base_th is not None else None

                if th_on is not None:
                    if vad_state == "silence" and rms_smooth >= th_on:
                        vad_state = "speech"
                        speech_start_ts = now
                        emit({"event": "VAD_START", "rms": round(rms_smooth, 0)}, "info")
                    elif vad_state == "speech" and rms_smooth <= th_off:
                        vad_state = "silence"
                        dur = (now - speech_start_ts) if speech_start_ts else 0.0
                        emit({"event": "VAD_END", "duration": round(dur, 2), "rms": round(rms_smooth, 0)}, "info")
                        speech_start_ts = None

                # Always send audio to Deepgram
                conn.send(chunk)

        except Exception as e:
            emit(f"Audio loop error: {e}", "error")
            traceback.print_exc()
        finally:
            try:
                conn.finish()
            except Exception:
                pass
            if stream:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            if wav:
                try:
                    wav.close()
                except Exception:
                    pass
            with self.state_lock:
                self.running = False
            emit("Worker finished.", "info")


worker = VoiceWorker()

# ---------- Flask app ----------
app = Flask(__name__)

INDEX_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ID Voice Debug (Deepgram)</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 0; background: #0b0f14; color: #e5e7eb; }
  header { padding: 16px 20px; background: #0f172a; border-bottom: 1px solid #1f2937; }
  h1 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: .2px; }
  main { max-width: 960px; margin: 0 auto; padding: 20px; }
  .card { background: #0f172a; border: 1px solid #1f2937; border-radius: 14px; padding: 16px; box-shadow: 0 8px 30px rgba(0,0,0,.25); }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: end; }
  label { font-size: 12px; color: #94a3b8; display: block; margin-bottom: 6px; }
  select,input { background: #0b1220; color: #e5e7eb; border: 1px solid #233146; border-radius: 10px; padding: 10px 12px; width: 100%; }
  input[type=number] { width: 120px; }
  .btn { background: #2563eb; color: white; border: 0; border-radius: 10px; padding: 10px 14px; font-weight: 600; cursor: pointer; }
  .btn.stop { background: #ef4444; }
  .btn:disabled { opacity: .5; cursor: not-allowed; }
  .log { background: #0b1220; border: 1px solid #233146; border-radius: 12px; padding: 12px; height: 360px; overflow: auto; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.3; }
  .tag { display: inline-block; font-size: 10px; padding: 2px 6px; border-radius: 999px; margin-right: 8px; }
  .t-info { background: #1e293b; color: #cbd5e1; }
  .t-warn { background: #92400e; color: #fde68a; }
  .t-error{ background: #7f1d1d; color: #fecaca; }
  .muted { color: #94a3b8; }
  footer { opacity: .6; font-size: 12px; padding-top: 10px; }
</style>
</head>
<body>
  <header><h1>Indonesian Voice Debug (Deepgram) — Localhost</h1></header>
  <main>
    <div class="card">
      <div class="grid">
        <div>
          <label>Input Device</label>
          <select id="device"></select>
        </div>
        <div>
          <label>Sample Rate</label>
          <select id="rate">
            <option value="16000">16000</option>
            <option value="44100">44100</option>
            <option value="48000" selected>48000</option>
          </select>
        </div>
      </div>
      <div class="grid">
        <div>
          <label>Model</label>
          <select id="model">
            <option value="nova-2" selected>nova-2</option>
            <option value="base">base</option>
          </select>
        </div>
        <div>
          <label>Language</label>
          <select id="lang">
            <option value="id" selected>id (Bahasa Indonesia)</option>
            <option value="en">en</option>
          </select>
        </div>
      </div>

      <div class="row" style="margin-bottom:12px;">
        <div>
          <label>VAD Threshold (auto jika kosong)</label>
          <input id="vadth" type="number" placeholder="e.g. 900" />
        </div>
        <div>
          <label class="muted">Options</label>
          <label style="display:inline-flex;align-items:center;gap:8px;">
            <input id="savewav" type="checkbox" /> Simpan debug_capture.wav
          </label>
          <label style="display:inline-flex;align-items:center;gap:8px;margin-left:16px;">
            <input id="rmsspam" type="checkbox" /> Log RMS tiap chunk (verbose)
          </label>
        </div>
      </div>

      <div class="row" style="margin-bottom:12px;">
        <button class="btn" id="startBtn">Start</button>
        <button class="btn stop" id="stopBtn" disabled>Stop</button>
        <div id="status" class="muted">Idle</div>
      </div>

      <div class="log" id="log"></div>
      <footer>Tips: kalau tidak ada VAD START saat bicara, coba ganti device atau set threshold (900–1500).</footer>
    </div>
  </main>

<script>
  const logEl = document.getElementById('log');
  const deviceSel = document.getElementById('device');
  const rateSel = document.getElementById('rate');
  const modelSel = document.getElementById('model');
  const langSel  = document.getElementById('lang');
  const vadThInp = document.getElementById('vadth');
  const saveWav  = document.getElementById('savewav');
  const rmsSpam  = document.getElementById('rmsspam');
  const statusEl = document.getElementById('status');
  const startBtn = document.getElementById('startBtn');
  const stopBtn  = document.getElementById('stopBtn');

  let es = null;

  function addLog(obj) {
    const level = obj.level || 'info';
    const tagClass = level === 'error' ? 't-error' : (level === 'warn' ? 't-warn' : 't-info');
    const tag = `<span class="tag ${tagClass}">${level.toUpperCase()}</span>`;
    const msg = obj.msg ? obj.msg
              : obj.event ? (`${obj.event}${obj.text ? ' — ' + (obj.final ? '[FINAL] ' : '[PART] ') + obj.text : ''}`)
              : JSON.stringify(obj);
    const line = document.createElement('div');
    const ts = new Date().toLocaleTimeString();
    line.innerHTML = `${tag}<span class="muted">${ts}</span> ${msg}`;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  }

  async function loadDevices() {
    try {
      const res = await fetch('/devices');
      const data = await res.json();
      deviceSel.innerHTML = '';
      data.devices.forEach(d => {
        const opt = document.createElement('option');
        opt.value = d.index;
        opt.textContent = `[${d.index}] ${d.name} (inputs=${d.inputs}, outputs=${d.outputs})`;
        if (d.inputs > 0 && deviceSel.options.length === 0) opt.selected = true;
        deviceSel.appendChild(opt);
      });
    } catch (e) {
      addLog({level:'error', msg:`Load devices failed: ${e}`});
    }
  }

  function connectSSE() {
    if (es) es.close();
    es = new EventSource('/events');
    es.onmessage = (e) => {
      try { addLog(JSON.parse(e.data)); } catch { addLog({level:'info', msg:e.data}); }
    };
    es.onerror = () => { /* keep alive */ };
  }

  async function start() {
    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusEl.textContent = 'Running…';
    const payload = {
      device: Number(deviceSel.value),
      rate: Number(rateSel.value),
      model: modelSel.value,
      lang: langSel.value,
      vad_th: vadThInp.value ? Number(vadThInp.value) : null,
      save_wav: !!saveWav.checked,
      rms_spam: !!rmsSpam.checked
    };
    const res = await fetch('/start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    const data = await res.json();
    addLog({level: data.ok ? 'info' : 'error', msg: data.msg});
    if (!data.ok) { startBtn.disabled = false; stopBtn.disabled = true; statusEl.textContent = 'Idle'; }
  }

  async function stop() {
    stopBtn.disabled = true;
    const res = await fetch('/stop', {method:'POST'});
    const data = await res.json();
    addLog({level: data.ok ? 'info' : 'error', msg: data.msg});
    startBtn.disabled = false;
    statusEl.textContent = 'Idle';
  }

  document.getElementById('startBtn').onclick = start;
  document.getElementById('stopBtn').onclick = stop;

  loadDevices();
  connectSSE();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/devices")
def devices():
    devs = []
    try:
        for i, d in enumerate(sd.query_devices()):
            devs.append({
                "index": i,
                "name": d.get("name", ""),
                "inputs": d.get("max_input_channels", 0),
                "outputs": d.get("max_output_channels", 0),
            })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "devices": []})
    return jsonify({"ok": True, "devices": devs})

@app.route("/start", methods=["POST"])
def start_route():
    data = request.get_json(force=True)
    device = data.get("device", None)
    rate = int(data.get("rate", 48000))
    model = data.get("model", "nova-2")
    lang = data.get("lang", "id")
    vad_th = data.get("vad_th", None)
    save_wav = bool(data.get("save_wav", False))
    rms_spam = bool(data.get("rms_spam", False))

    if worker.is_running():
        return jsonify({"ok": False, "msg": "Worker already running."})

    ok = worker.start(device=device, rate=rate, model=model, lang=lang,
                      vad_th=vad_th, save_wav=save_wav, rms_spam=rms_spam)
    return jsonify({"ok": ok, "msg": "Started." if ok else "Failed to start."})

@app.route("/stop", methods=["POST"])
def stop_route():
    if not worker.is_running():
        return jsonify({"ok": False, "msg": "Worker is not running."})
    worker.stop()
    return jsonify({"ok": True, "msg": "Stopped."})

@app.route("/events")
def sse_events():
    def gen():
        q = pubsub.subscribe()
        try:
            while True:
                msg = q.get()
                yield f"data: {json.dumps(msg)}\n\n"
        except GeneratorExit:
            pass
        finally:
            pubsub.unsubscribe(q)
    return Response(gen(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"  # for proxies that buffer
    })

if __name__ == "__main__":
    # Flask dev server
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
