/* Amadeus avatar v1.
 *
 * Two pieces behind a stable interface so a Live2D backend can slot in later:
 *
 *   AvatarDriver      — layered animation: breathing, blink scheduler, gaze,
 *                       audio-energy lip sync, expression from emotional
 *                       state, posture from conversation state
 *
 * A future Live2DAvatar only needs render(params) mapping the same params
 * onto model parameters (ParamMouthOpenY, ParamEyeLOpen, ...).
 *
 * The figure is an original design; no third-party character assets.
 */
"use strict";

class AvatarDriver {
  constructor(renderer, getVoiceLevel) {
    this.renderer = renderer;
    this.getVoiceLevel = getVoiceLevel;   // () => 0..1 from playback analyser
    this.state = "idle";                  // idle | thinking | speaking | listening
    this.emotion = { mood: 0.55, energy: 0.6, curiosity: 0.65,
                     confidence: 0.6, stress: 0.3, trust: 0.15 };
    this._now = performance.now();
    this._blinkAt = this._now + 1500;
    this._blinkPhase = -1;               // <0 idle, 0..1 during blink
    this._gaze = { x: 0, y: 0, tx: 0, ty: 0, until: 0 };
    this._mouth = 0;
    this._breathePhase = 0;
    this._wavePhase = 0;
    this._waveAmp = 3;
    this._tiltTarget = 0;
    this._tilt = 0;
    this._jolt = 0;
    // Deliberately ignore the OS reduced-motion setting: this app IS an
    // animated character; honoring it once left her frozen and cost an
    // hour of debugging. We note the override so it's visible, not silent.
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      console.info("[avatar] OS reduced-motion is on; Amadeus animates anyway");
    }
    this._warned = false;
    requestAnimationFrame((t) => this._frame(t));
  }

  setState(state) {
    this.state = state;
    this._tiltTarget = state === "thinking" ? 0.07 : state === "listening" ? -0.03 : 0;
  }

  setEmotion(traits) { Object.assign(this.emotion, traits || {}); }

  setExpression(name, holdMs = 9000) {
    if (!this.renderer.setExpression) return;   // parametric has no files
    this.renderer.setExpression(name);
    clearTimeout(this._exprTimer);
    if (name) {
      this._exprTimer = setTimeout(() => this.renderer.setExpression(null), holdMs);
    }
  }

  reactInterrupted() {
    this._jolt = 1;
    this._blinkPhase = 0;                 // immediate blink
    this.setState("listening");
  }

  _frame(now) {
    // schedule the next frame FIRST so a bad frame can never kill the loop
    requestAnimationFrame((t) => this._frame(t));
    try {
      this._tick(now);
    } catch (error) {
      if (!this._warned) {
        this._warned = true;
        console.warn("[avatar] frame error (loop continues):", error);
      }
    }
  }

  _tick(now) {
    const dt = Math.min((now - this._now) / 1000, 0.05);
    this._now = now;
    const e = this.emotion;

    // breathing (slightly quicker when energetic or speaking)
    const rate = 0.9 + e.energy * 0.5 + (this.state === "speaking" ? 0.25 : 0);
    this._breathePhase += dt * rate;
    const breathe = Math.sin(this._breathePhase);

    // blink scheduler
    if (this._blinkPhase < 0 && now >= this._blinkAt) this._blinkPhase = 0;
    let lid = 1;
    if (this._blinkPhase >= 0) {
      this._blinkPhase += dt / 0.13;      // 130 ms blink
      lid = Math.abs(1 - 2 * Math.min(this._blinkPhase, 1));
      if (this._blinkPhase >= 1) {
        this._blinkPhase = -1;
        const gap = 2000 + Math.random() * 4000 * (0.6 + e.energy);
        this._blinkAt = now + (Math.random() < 0.14 ? 260 : gap);
      }
    }

    // gaze micro-movement
    if (now >= this._gaze.until) {
      const bias = this.state === "thinking" ? { x: -0.5, y: -0.6 } : { x: 0, y: 0.05 };
      this._gaze.tx = bias.x * 0.6 + (Math.random() - 0.5) * 0.7;
      this._gaze.ty = bias.y * 0.6 + (Math.random() - 0.5) * 0.4;
      this._gaze.until = now + 1400 + Math.random() * 2600;
    }
    this._gaze.x += (this._gaze.tx - this._gaze.x) * Math.min(dt * 6, 1);
    this._gaze.y += (this._gaze.ty - this._gaze.y) * Math.min(dt * 6, 1);

    // mouth from live audio energy
    const level = this.getVoiceLevel ? this.getVoiceLevel() : 0;
    const target = this.state === "speaking" || level > 0.02 ? Math.min(1, level * 3.2) : 0;
    this._mouth += (target - this._mouth) * Math.min(dt * 14, 1);

    // presence-line wave follows speech
    const waveTarget = target > 0.03 ? 10 : this.state === "thinking" ? 5 : 3;
    this._waveAmp += (waveTarget - this._waveAmp) * Math.min(dt * 4, 1);
    this._wavePhase += dt * (waveTarget > 6 ? 5 : 1.2);

    // posture
    this._tilt += (this._tiltTarget - this._tilt) * Math.min(dt * 5, 1);
    this._jolt = Math.max(0, this._jolt - dt * 3);

    const eyeBase = 0.62 + e.energy * 0.3
      + (this.state === "listening" ? 0.08 : 0)
      + e.curiosity * 0.12 + this._jolt * 0.25;

    this.renderer.render({
      breathe,
      eyeOpen: Math.min(1.15, eyeBase) * lid,
      pupilX: this._gaze.x,
      pupilY: this._gaze.y,
      browRaise: e.curiosity * 0.5 + this._jolt * 0.9 - e.stress * 0.2,
      browTilt: e.stress * 0.5 - e.confidence * 0.2,
      mouthOpen: this._mouth,
      mouthCurve: (e.mood - 0.5) * 1.4 + (e.trust - 0.3) * 0.6 - this._mouth * 0.3,
      blush: Math.max(0, e.trust - 0.45) * 1.6,
      headTilt: this._tilt + Math.sin(this._breathePhase * 0.5) * 0.008 + this._jolt * 0.05,
      headX: Math.sin(this._breathePhase * 0.35) * 0.8,
      headY: this._jolt * 2,
      wavePhase: this._wavePhase,
      waveAmp: this._waveAmp,
    });
  }
}

window.AmadeusAvatar = { AvatarDriver };

/* ---------------------------------------------------------------------------
 * Live2D backend. Renders one frame from the driver's flat params object,
 * driver's signals onto standard Cubism parameter ids. Loaded only when
 * /api/avatar reports an installed model + runtime; any failure falls back
 * to the parametric renderer.
 * ------------------------------------------------------------------------- */

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const tag = document.createElement("script");
    tag.src = src;
    tag.onload = resolve;
    tag.onerror = () => reject(new Error("failed to load " + src));
    document.head.appendChild(tag);
  });
}

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(label + " timed out after " + ms / 1000 + "s")), ms)
    ),
  ]);
}

window.addEventListener("unhandledrejection", (event) => {
  console.warn("[avatar] unhandled rejection:", event.reason);
});

class Live2DAvatar {
  static async create(canvas, config) {
    console.info("[avatar] loading runtime scripts...");
    await loadScript("/vendor/live2dcubismcore.min.js");
    await loadScript("/vendor/pixi.min.js");
    await loadScript("/vendor/cubism4.min.js");
    console.info("[avatar] runtime ready; PIXI.live2d =", typeof PIXI.live2d);

    const app = new PIXI.Application({
      view: canvas,
      autoStart: false,
      backgroundAlpha: 0,
      antialias: true,
      resolution: window.devicePixelRatio || 1,
      autoDensity: true,
      width: canvas.clientWidth,
      height: canvas.clientHeight,
    });
    console.info("[avatar] loading model:", config.model_url);
    const model = await withTimeout(
      PIXI.live2d.Live2DModel.from(config.model_url, {
        autoUpdate: false,
        autoInteract: false,
        idleMotionGroup: "amadeus_none",   // we drive her; no canned motions
      }),
      45000,
      "model load"
    );
    console.info("[avatar] model loaded");
    app.stage.addChild(model);

    const avatar = new Live2DAvatar(app, model, canvas, config);

    // CRITICAL: the framework's update cycle begins with loadParameters(),
    // which restores a saved snapshot and ERASES any values set beforehand.
    // Injecting our parameters via the motionManager hook places them inside
    // the load/save window, so they persist — and physics then reacts to
    // them (hair/ears/tail sway from our head motion).
    const internal = model.internalModel;
    internal.eyeBlink = undefined;   // we drive blinking; disable the built-in
    internal.motionManager.update = () => {
      avatar._applyParams();
      return true;
    };
    console.info("[avatar] parameter hook installed");

    // preload expression definitions (name -> parameter list)
    for (const [name, url] of Object.entries(config.expressions || {})) {
      try {
        const data = await (await fetch(url)).json();
        avatar._expressions[name] = data.Parameters || [];
      } catch (error) {
        console.warn("[avatar] failed to load expression", name, error);
      }
    }
    console.info("[avatar] expressions loaded:", Object.keys(avatar._expressions).join(", ") || "(none)");

    avatar._fit();
    // verification render: if the model came out zero-sized or NaN-placed,
    // fail loudly so the caller keeps the parametric face instead of a blank
    avatar.render(Live2DAvatar.NEUTRAL);
    const bounds = model.getBounds();
    console.info("[avatar] live2d bounds after fit:", bounds.width, "x", bounds.height);
    if (!isFinite(bounds.width) || bounds.width < 1 || !isFinite(bounds.height)) {
      throw new Error("model rendered with invalid size " + bounds.width + "x" + bounds.height);
    }
    window.addEventListener("resize", () => {
      app.renderer.resize(canvas.clientWidth, canvas.clientHeight);
      avatar._fit();
    });
    return avatar;
  }

  constructor(app, model, canvas, config) {
    this.app = app;
    this.model = model;
    this.canvas = canvas;
    this.config = config;
    this._map = config.param_map || {};   // per-model id overrides
    this._last = performance.now();
    this._badParams = new Set();
    this._expressions = {};               // name -> [{Id, Value, Blend}]
    this._active = null;                  // {params, weight, target}
  }

  setExpression(name) {
    if (name && this._expressions[name]) {
      this._active = { params: this._expressions[name], weight: this._active?.weight || 0, target: 1 };
    } else if (this._active) {
      this._active.target = 0;            // fade out
    }
  }

  _fit() {
    const h = this.canvas.clientHeight, w = this.canvas.clientWidth;
    // natural model size: property names vary across models/library
    // versions, so try several sources before giving up
    this.model.scale.set(1);
    const im = this.model.internalModel;
    let nh = im.originalHeight || im.height || this.model.height;
    let nw = im.originalWidth || im.width || this.model.width;
    if (!isFinite(nh) || nh <= 0) {
      const bounds = this.model.getBounds();
      nh = bounds.height; nw = bounds.width;
    }
    console.info("[avatar] model natural size:", nw, "x", nh, "| canvas:", w, "x", h);
    let scale = (h / nh) * 1.55 * (this.config.scale || 1);
    if (!isFinite(scale) || scale <= 0) scale = 0.1;   // visible beats blank
    this.model.scale.set(scale);
    this.model.anchor.set(0.5, 0.03 + (this.config.offset_y || 0));
    this.model.position.set(w / 2, 0);
    console.info("[avatar] applied scale:", scale);
  }

  _set(id, value) {
    if (this._badParams.has(id)) return;
    try {
      this.model.internalModel.coreModel.setParameterValueById(id, value);
    } catch (error) {
      this._badParams.add(id);   // model lacks this parameter; skip forever
    }
  }

  render(p) {
    this._params = p;
    const now = performance.now();
    this._dt = Math.min((now - this._last) / 1000, 0.05);
    const dt = now - this._last;
    this._last = now;
    this.model.update(dt);   // physics + our hooked params
    this.app.render();
  }

  _applyExpression() {
    const active = this._active;
    if (!active) return;
    active.weight += (active.target - active.weight) * Math.min((this._dt || 0.016) * 6, 1);
    if (active.target === 0 && active.weight < 0.01) { this._active = null; return; }
    const core = this.model.internalModel.coreModel;
    for (const entry of active.params) {
      try {
        const blend = entry.Blend || "Add";
        if (blend === "Add") core.addParameterValueById(entry.Id, entry.Value, active.weight);
        else if (blend === "Multiply") core.multiplyParameterValueById(entry.Id, entry.Value, active.weight);
        else core.setParameterValueById(entry.Id, entry.Value, active.weight);
      } catch (error) { /* model lacks this parameter; skip */ }
    }
  }

  _applyParams() {
    const p = this._params;
    if (!p) return;
    const eye = Math.max(0, Math.min(1, p.eyeOpen));
    this._set("ParamEyeLOpen", eye);
    this._set("ParamEyeROpen", eye);
    this._set("ParamEyeBallX", p.pupilX);
    this._set("ParamEyeBallY", -p.pupilY);
    this._set("ParamBrowLY", p.browRaise * 0.8);
    this._set("ParamBrowRY", p.browRaise * 0.8);
    this._set("ParamMouthOpenY", Math.max(0, Math.min(1, p.mouthOpen * 1.15)));
    this._set("ParamMouthForm", Math.max(-1, Math.min(1, p.mouthCurve)));
    const smile = Math.max(0, Math.min(1, p.mouthCurve * 0.9));
    this._set("ParamEyeLSmile", smile);
    this._set("ParamEyeRSmile", smile);
    this._set(this._map.blush || "ParamCheek", Math.max(0, Math.min(1, p.blush)));
    this._set("ParamAngleX", p.pupilX * 9 + p.headX * 2);
    this._set("ParamAngleY", -p.pupilY * 7 - p.headY * 2);
    this._set("ParamAngleZ", p.headTilt * 45);
    this._set(this._map.bodyTilt || "ParamBodyAngleZ", p.headTilt * 12);
    this._set("ParamBreath", (p.breathe + 1) / 2);
    this._applyExpression();   // last, so it layers over the base pose
  }
}

async function createAvatarRenderer(container, onNotice) {
  try {
    const response = await fetch("/api/avatar");
    const config = await response.json();
    if (config.renderer !== "live2d") {
      if (config.note) {
        console.warn("[avatar]", config.note);
        if (onNotice) onNotice(config.note);
      }
      return null;   // empty stage until a model is installed
    }
    const canvas = document.createElement("canvas");
    canvas.id = "avatar-live2d";
    canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;";
    container.appendChild(canvas);
    return await Live2DAvatar.create(canvas, config);
  } catch (error) {
    console.warn("[avatar] Live2D failed:", error);
    const stray = document.getElementById("avatar-live2d");
    if (stray) stray.remove();
    if (onNotice) onNotice("Avatar failed to load: " + (error && error.message ? error.message : error));
    return null;
  }
}

Live2DAvatar.NEUTRAL = {
  breathe: 0, eyeOpen: 1, pupilX: 0, pupilY: 0, browRaise: 0, browTilt: 0,
  mouthOpen: 0, mouthCurve: 0, blush: 0, headTilt: 0, headX: 0, headY: 0,
  wavePhase: 0, waveAmp: 0,
};

window.AmadeusAvatar.Live2DAvatar = Live2DAvatar;
window.AmadeusAvatar.createAvatarRenderer = createAvatarRenderer;
