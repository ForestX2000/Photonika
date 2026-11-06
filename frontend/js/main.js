// Photonika — логика сайта. Без сборки: ES-модуль, работает с API из backend/ (тот же origin).
const API = "";                       // при отдельном фронтенд-сервере: "http://127.0.0.1:8000"
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const fmt = (v, d = 3) => Number(v).toLocaleString("ru-RU", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtMs = (ms) => ms >= 1000 ? `${fmt(ms / 1000, ms >= 10000 ? 1 : 2)} с` : `${fmt(ms, ms >= 100 ? 0 : 1)} мс`;

const DEFAULT_DISTANCES = [24.1, 25.3, 26.5, 27.7];
// PCC по плоскостям для картинок первого экрана (photos_classic, GS-200 + Adam-400; tools/make_web_assets.py)
const HERO_PCC = [0.59, 0.57, 0.55, 0.54];
const SET_NOTES = {
  photos_classic: "Классика обработки изображений: самый трудный случай",
  mephi_letters: "Крупные буквы на чёрном фоне — видно спекл",
  text_binary: "Текст, силуэт и фигуры: резкие края",
  digits: "Цифры «2026» — простые бинарные формы",
};
const METHOD_TAGS = {
  "net+gs": ["быстро", "fast"], "net": ["мгновенно", "fast"], "net+adam": ["точнее", ""],
  "gs+adam": ["эталон · долго", "slow"], "gs": ["классика", ""], "adam": ["долго", "slow"],
};
// Какие параметры показывать для метода
const METHOD_PARAMS = {
  "net+gs": ["refine_steps"], "net": [], "net+adam": ["refine_steps"],
  "gs+adam": ["gs_iters", "adam_steps"], "gs": ["gs_iters"], "adam": ["adam_steps"],
};

const state = {
  info: null,
  distances: DEFAULT_DISTANCES,
  slots: [null, null, null, null],   // {url, file?, testset?, index?}
  testset: null,
  method: null,
  result: null,
  view: "recon",
  busy: false,
};

// ---------------------------------------------------------------- утилиты
async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    let msg = `Ошибка ${res.status}`;
    try { const j = await res.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg);
  }
  return res;
}

let toastTimer;
function toast(text, ok = false) {
  const el = $("#toast");
  el.textContent = text;
  el.classList.toggle("ok", ok);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 5200);
}

const zLabel = (z) => `${fmt(z, 1)} мм`;

// ---------------------------------------------------------------- навигация
function initNav() {
  const nav = $(".nav");
  const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 20);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  const burger = $("#navBurger"), links = $("#navLinks");
  burger.addEventListener("click", () => {
    const open = burger.getAttribute("aria-expanded") !== "true";
    burger.setAttribute("aria-expanded", String(open));
    links.classList.toggle("open", open);
  });
  links.addEventListener("click", (e) => {
    if (e.target.closest("a")) { burger.setAttribute("aria-expanded", "false"); links.classList.remove("open"); }
  });
}

// ---------------------------------------------------------------- первый экран
function initHero() {
  const plates = $$(".plate[data-plane]");
  const items = $$("#planeIndex li");
  const progress = $("#factProgress");
  let current = 0, timer;
  const show = (i) => {
    current = i;
    plates.forEach((p, k) => p.classList.toggle("active", k === i));
    items.forEach((li, k) => li.classList.toggle("active", k === i));
    progress.style.transform = `translateX(${i * 100}%)`;
    $("#focusTitle").textContent = `Плоскость 0${i + 1}`;
    $("#focusMeta").textContent = `z = ${zLabel(DEFAULT_DISTANCES[i])} · PCC ${fmt(HERO_PCC[i], 2)}`;
    const bar = $("#focusBar");
    bar.classList.remove("run"); void bar.offsetWidth; bar.classList.add("run");
  };
  const loop = () => { clearInterval(timer); timer = setInterval(() => show((current + 1) % 4), 2600); };
  items.forEach((li, i) => li.querySelector("button").addEventListener("click", () => { show(i); loop(); }));
  show(0);
  if (!matchMedia("(prefers-reduced-motion: reduce)").matches) loop();
  initSpeckle($("#heroCanvas"));
  const track = $("#marqueeTrack");
  track.innerHTML += track.innerHTML;          // две копии — бесшовная прокрутка на −50 %
}

// Фон: дрейфующий «спекл» — мерцающие точки лазерного цвета
function initSpeckle(canvas) {
  const ctx = canvas.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let w, h, dots = [], raf;
  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = canvas.clientWidth; h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round((w * h) / 9000);
    dots = Array.from({ length: n }, () => ({
      x: Math.random() * w, y: Math.random() * h, r: Math.random() * 1.4 + .2,
      v: Math.random() * .25 + .05, p: Math.random() * Math.PI * 2, g: Math.random() < .55,
    }));
  };
  const draw = (t) => {
    ctx.clearRect(0, 0, w, h);
    for (const d of dots) {
      const a = .25 + .75 * Math.abs(Math.sin(d.p + t / 1400));
      ctx.fillStyle = d.g ? `rgba(92,255,143,${a * .55})` : `rgba(234,243,237,${a * .35})`;
      ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2); ctx.fill();
      if (!reduce) { d.x += d.v; if (d.x > w + 4) d.x = -4; }
    }
    if (!reduce) raf = requestAnimationFrame(draw);
  };
  resize();
  window.addEventListener("resize", () => { cancelAnimationFrame(raf); resize(); draw(0); });
  draw(0);
}

// ---------------------------------------------------------------- данные с сервера
async function loadInfo() {
  const chip = $("#deviceChip"), text = $("#deviceText");
  try {
    const info = await (await api("/api/info")).json();
    state.info = info;
    state.distances = info.distances_mm;
    chip.classList.add("ok");
    text.textContent = (info.gpu ? `GPU · ${info.gpu.replace("NVIDIA GeForce ", "")}` : "CPU") + (info.weights ? " · сеть готова" : "");
    $("#critDist").textContent = `${fmt(info.critical_distance_mm, 2)} мм`;
    renderMethods(info.methods, info.default_method);
  } catch (e) {
    chip.classList.add("err");
    text.textContent = "сервер недоступен";
    renderMethods(null);
    toast("Не удалось подключиться к серверу. Запустите: python -m backend");
  }
}

async function loadTestsets() {
  const box = $("#testsets");
  try {
    const sets = await (await api("/api/testsets")).json();
    box.innerHTML = "";
    for (const s of sets) {
      const b = document.createElement("button");
      b.type = "button"; b.className = "testset"; b.setAttribute("role", "listitem"); b.setAttribute("aria-pressed", "false");
      b.dataset.name = s.name;
      b.innerHTML = `<div class="thumbs">${s.images.map((u) => `<img src="${API}${u}" alt="" loading="lazy">`).join("")}</div><span></span>`;
      b.querySelector("span").textContent = s.title;
      b.addEventListener("click", () => useTestset(s));
      box.appendChild(b);
    }
    if (!sets.length) box.innerHTML = `<p class="muted small">Тестовых наборов нет: python -m ml.cli make-testsets</p>`;
    renderGallery(sets);
  } catch { box.innerHTML = `<p class="muted small">Тестовые наборы недоступны.</p>`; }
}

// Большие карточки «Готовые наборы»: клик — набор в студию
function renderGallery(sets) {
  const box = $("#galleryCards");
  const order = ["photos_classic", "mephi_letters", "text_binary", "digits"];
  const featured = sets.filter((s) => !s.name.startsWith("random"))
    .sort((x, y) => (order.indexOf(x.name) + 1 || 99) - (order.indexOf(y.name) + 1 || 99)).slice(0, 4);
  box.innerHTML = "";
  featured.forEach((s, k) => {
    const card = document.createElement("button");
    card.type = "button"; card.className = "set-card reveal";
    card.innerHTML = `<div class="mosaic">${s.images.map((u) => `<img src="${API}${u}" alt="" loading="lazy">`).join("")}</div>
      <span class="set-num glass">НАБОР №${k + 1}</span>
      <div class="set-meta"><h3></h3><p></p><span class="set-go">Синтезировать
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg></span></div>`;
    card.querySelector("h3").textContent = s.title.split(":")[0];
    card.querySelector("p").textContent = SET_NOTES[s.name] || s.title;
    card.setAttribute("aria-label", `${s.title}: открыть в студии`);
    card.addEventListener("click", () => {
      useTestset(s);
      $("#studio").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    box.appendChild(card);
    requestAnimationFrame(() => card.classList.add("in"));
  });
}

// ---------------------------------------------------------------- слоты изображений
function renderSlots() {
  const box = $("#slots");
  box.innerHTML = "";
  state.slots.forEach((slot, i) => {
    const el = document.createElement("div");
    el.className = "slot" + (slot ? " filled" : "");
    el.tabIndex = 0;
    el.setAttribute("role", "button");
    el.setAttribute("aria-label", `Изображение ${i + 1}, z = ${zLabel(state.distances[i])}` + (slot ? ", загружено" : ", пусто"));
    if (slot) {
      el.innerHTML = `<img alt=""><span class="slot-tag">${i + 1} · ${zLabel(state.distances[i])}</span>
        <button class="slot-remove" type="button" aria-label="Убрать изображение ${i + 1}">×</button>`;
      el.querySelector("img").src = slot.url;
      el.querySelector(".slot-remove").addEventListener("click", (e) => { e.stopPropagation(); setSlot(i, null); });
    } else {
      el.innerHTML = `<span class="slot-plus">+</span><span class="slot-z">${i + 1} · ${zLabel(state.distances[i])}</span>
        <span class="slot-hint">перетащите или нажмите</span>`;
    }
    el.addEventListener("click", () => pickFile(i));
    el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pickFile(i); } });
    el.addEventListener("dragover", (e) => { e.preventDefault(); el.classList.add("drag"); });
    el.addEventListener("dragleave", () => el.classList.remove("drag"));
    el.addEventListener("drop", (e) => {
      e.preventDefault(); el.classList.remove("drag");
      const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith("image/"));
      if (files.length > 1) distribute(files, i); else if (files[0]) setSlot(i, fileSlot(files[0]));
    });
    box.appendChild(el);
  });
  updateRunState();
}

const fileSlot = (file) => ({ file, url: URL.createObjectURL(file) });

function setSlot(i, value) {
  const old = state.slots[i];
  if (old?.file) URL.revokeObjectURL(old.url);
  state.slots[i] = value;
  if (!value || value.file) markTestset(null);
  renderSlots();
}

function distribute(files, start = 0) {
  files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  files.slice(0, 4).forEach((f, k) => {
    const i = files.length >= 4 ? k : (start + k) % 4;
    const old = state.slots[i];
    if (old?.file) URL.revokeObjectURL(old.url);
    state.slots[i] = fileSlot(f);
  });
  markTestset(null);
  renderSlots();
  if (files.length > 4) toast("Взяты первые 4 файла по имени.");
}

let pickIndex = 0;
function pickFile(i) { pickIndex = i; $("#fileInput").value = ""; $("#fileInput").click(); }

function useTestset(s) {
  state.slots.forEach((old) => old?.file && URL.revokeObjectURL(old.url));
  state.slots = s.images.map((u, k) => ({ url: API + u, testset: s.name, index: k + 1 }));
  markTestset(s.name);
  renderSlots();
}

function markTestset(name) {
  state.testset = name;
  $$(".testset").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.name === name)));
}

function initSlots() {
  $("#fileInput").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) setSlot(pickIndex, fileSlot(f)); });
  $("#multiInput").addEventListener("change", (e) => { const fs = [...e.target.files]; if (fs.length) distribute(fs, 0); });
  $("#pickAll").addEventListener("click", () => { $("#multiInput").value = ""; $("#multiInput").click(); });
  $("#clearSlots").addEventListener("click", () => {
    state.slots.forEach((old) => old?.file && URL.revokeObjectURL(old.url));
    state.slots = [null, null, null, null]; markTestset(null); renderSlots();
  });
  renderSlots();
}

// ---------------------------------------------------------------- методы
function renderMethods(methods, defaultMethod) {
  const box = $("#methods");
  box.innerHTML = "";
  if (!methods) { box.innerHTML = `<p class="muted small">Методы появятся после подключения к серверу.</p>`; return; }
  const sorted = [...methods].sort((a, b) => Number(b.available) - Number(a.available));
  for (const m of sorted) {
    const el = document.createElement("div");
    el.className = "method"; el.setAttribute("role", "radio"); el.tabIndex = -1;
    el.dataset.id = m.id;
    el.setAttribute("aria-checked", "false");
    if (!m.available) el.setAttribute("aria-disabled", "true");
    const [tag, cls] = m.available ? (METHOD_TAGS[m.id] || ["", ""]) : ["нужны веса", ""];
    el.innerHTML = `<span class="radio" aria-hidden="true"></span><span><b></b><small></small></span>${tag ? `<span class="tag ${cls}">${tag}</span>` : ""}`;
    el.querySelector("b").textContent = m.title;
    el.querySelector("small").textContent = m.description;
    el.addEventListener("click", () => m.available && selectMethod(m.id));
    el.addEventListener("keydown", (e) => {
      const avail = $$(".method:not([aria-disabled='true'])", box);
      const k = avail.indexOf(el);
      if (["ArrowDown", "ArrowRight"].includes(e.key)) { e.preventDefault(); const n = avail[(k + 1) % avail.length]; selectMethod(n.dataset.id); n.focus(); }
      if (["ArrowUp", "ArrowLeft"].includes(e.key)) { e.preventDefault(); const n = avail[(k - 1 + avail.length) % avail.length]; selectMethod(n.dataset.id); n.focus(); }
      if (e.key === " " || e.key === "Enter") { e.preventDefault(); m.available && selectMethod(m.id); }
    });
    box.appendChild(el);
  }
  selectMethod(defaultMethod);
}

function selectMethod(id) {
  state.method = id;
  $$(".method").forEach((el) => {
    const on = el.dataset.id === id;
    el.setAttribute("aria-checked", String(on));
    el.tabIndex = on ? 0 : -1;
  });
  const params = METHOD_PARAMS[id] || [];
  $$(".range-row").forEach((row) => row.classList.toggle("off", !params.includes(row.dataset.param)));
  $("#advanced").hidden = params.length === 0;
  updateRunState();
}

function initRanges() {
  for (const [input, out] of [["#gsIters", "#gsItersOut"], ["#adamSteps", "#adamStepsOut"], ["#refineSteps", "#refineStepsOut"]]) {
    const el = $(input);
    const sync = () => ($(out).textContent = el.value);
    el.addEventListener("input", sync); sync();
  }
}

// ---------------------------------------------------------------- синтез
function updateRunState() {
  const ready = state.slots.every(Boolean) && state.method && state.info;
  const btn = $("#runBtn"), hint = $("#runHint");
  btn.disabled = !ready || state.busy;
  hint.classList.remove("err");
  if (!state.info) hint.textContent = "Ожидание сервера…";
  else if (!state.slots.every(Boolean)) {
    const n = state.slots.filter(Boolean).length;
    hint.textContent = n ? `Добавлено ${n} из 4 изображений.` : "Добавьте 4 изображения или выберите тестовый набор.";
  } else hint.textContent = state.method?.includes("adam") && !state.method.startsWith("net")
    ? "Итерационный эталон: на CPU это несколько секунд." : "Готово к синтезу.";
}

async function slotBlob(slot) {
  if (slot.file) return slot.file;
  return (await api(slot.url.replace(API, ""))).blob();
}

async function runSynthesis() {
  if (state.busy) return;
  const btn = $("#runBtn"), label = $(".btn-label", btn), hint = $("#runHint");
  state.busy = true; btn.classList.add("loading"); btn.disabled = true;
  const started = performance.now();
  let error = null;
  const tick = setInterval(() => (label.textContent = `Считаю… ${fmt((performance.now() - started) / 1000, 1)} с`), 100);
  try {
    const fd = new FormData();
    fd.append("method", state.method);
    fd.append("gs_iters", $("#gsIters").value);
    fd.append("adam_steps", $("#adamSteps").value);
    fd.append("refine_steps", $("#refineSteps").value);
    if (state.testset && state.slots.every((s) => s.testset === state.testset)) fd.append("testset", state.testset);
    else {
      const blobs = await Promise.all(state.slots.map(slotBlob));
      blobs.forEach((b, i) => fd.append(`image_${i + 1}`, b, b.name || `image_${i + 1}.png`));
    }
    const result = await (await api("/api/synthesize", { method: "POST", body: fd })).json();
    state.result = result;
    renderResult(result);
    $("#results").hidden = false;
    $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    error = e.message;
    toast(e.message);
  } finally {
    clearInterval(tick);
    label.textContent = "Синтезировать ДОЭ";
    state.busy = false; btn.classList.remove("loading");
    updateRunState();
    if (error) { hint.textContent = error; hint.classList.add("err"); }
  }
}

// ---------------------------------------------------------------- результат
function renderResult(r) {
  $("#resultEyebrow").textContent = `${r.method_title} · ${r.device.toUpperCase()}`;
  $("#stats").innerHTML = [
    [fmt(r.mean.pcc), "PCC, среднее по 4 плоскостям"],
    [fmt(r.mean.ssim), "SSIM (skimage), среднее"],
    [fmt(r.mean.ssim_global), "SSIM по формуле слайда"],
    [fmtMs(r.time_ms), `время синтеза · <span class="sub">${r.method_title}</span>`],
  ].map(([v, s]) => `<div class="stat"><b>${v}</b><span>${s}</span></div>`).join("");

  setDoeView("color");
  $("#downloadBtn").href = `${API}/api/results/${r.id}/download.zip`;
  $("#downloadBtn").setAttribute("download", `photonika_${r.id}.zip`);
  $("#compare").hidden = true;
  $("#compareBtn").disabled = false;
  $("#compareBtn").textContent = "Сравнить с GS-50";

  renderPlanes();
  renderTable(r);
  initViewerFor(r);
}

function setDoeView(kind) {
  const r = state.result;
  $$("[data-doe]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.doe === kind)));
  $("#doeImg").src = kind === "color" ? r.doe_color : r.doe;
  $("#doeCaption").textContent = kind === "color"
    ? "Фаза от −π до π в циклической палитре (−π и +π — один цвет)."
    : "Так выглядит doe.png: 8 бит, 0 = −π, 255 = +π.";
}

function renderPlanes() {
  const r = state.result;
  const box = $("#planes");
  box.innerHTML = "";
  for (const p of r.planes) {
    const main = state.view === "source" ? p.source : state.view === "laser" ? p.recon_laser : p.recon;
    const alt = state.view === "source" ? p.recon : p.source;
    const card = document.createElement("article");
    card.className = "plane-card";
    card.innerHTML = `<img alt=""><img class="alt" alt="">
      <div class="plane-meta"><h4>Плоскость №${p.index}</h4><p>z = ${zLabel(p.distance_mm)}</p>
        <div class="plane-chips"><span class="chip">PCC <b>${fmt(p.pcc)}</b></span><span class="chip">SSIM <b>${fmt(p.ssim)}</b></span></div></div>`;
    const [a, b] = card.querySelectorAll("img");
    a.src = main; a.alt = `Плоскость ${p.index}: ${state.view === "source" ? "исходное" : "восстановление"}`;
    b.src = alt;
    box.appendChild(card);
  }
}

function renderTable(r) {
  const rows = r.planes.map((p) => `<tr><td>${p.index}</td><td>${fmt(p.distance_mm, 1)}</td><td>${fmt(p.pcc, 4)}</td><td>${fmt(p.ssim, 4)}</td><td>${fmt(p.ssim_global, 4)}</td></tr>`);
  rows.push(`<tr><td>среднее</td><td></td><td>${fmt(r.mean.pcc, 4)}</td><td>${fmt(r.mean.ssim, 4)}</td><td>${fmt(r.mean.ssim_global, 4)}</td></tr>`);
  $("#metricsTable").innerHTML = `<thead><tr><th>Плоскость</th><th>z, мм</th><th>PCC</th><th>SSIM</th><th>SSIM (слайд)</th></tr></thead><tbody>${rows.join("")}</tbody>`;
}

async function runCompare() {
  const r = state.result, btn = $("#compareBtn");
  btn.disabled = true; btn.textContent = "Считаю GS-50…";
  try {
    const c = await (await api(`/api/results/${r.id}/compare?gs_iters=50`, { method: "POST" })).json();
    const tMax = Math.max(r.time_ms, c.time_ms);
    const faster = c.time_ms / Math.max(r.time_ms, 1e-6);
    const dPcc = r.mean.pcc - c.mean.pcc;
    const box = $("#compare");
    box.innerHTML = `<h4>${r.method_title} против GS-50</h4>
      <div class="bar-row"><span>время, наш</span><div class="bar"><span style="width:${(r.time_ms / tMax) * 100}%"></span></div><b>${fmtMs(r.time_ms)}</b></div>
      <div class="bar-row"><span>время, GS-50</span><div class="bar ref"><span style="width:${(c.time_ms / tMax) * 100}%"></span></div><b>${fmtMs(c.time_ms)}</b></div>
      <div class="bar-row"><span>PCC, наш</span><div class="bar"><span style="width:${Math.max(0, r.mean.pcc) * 100}%"></span></div><b>${fmt(r.mean.pcc)}</b></div>
      <div class="bar-row"><span>PCC, GS-50</span><div class="bar ref"><span style="width:${Math.max(0, c.mean.pcc) * 100}%"></span></div><b>${fmt(c.mean.pcc)}</b></div>
      <p class="verdict muted">${faster >= 1 ? `В ${fmt(faster, 1)}× быстрее` : `В ${fmt(1 / faster, 1)}× медленнее`} GS-50,
        PCC ${dPcc >= 0 ? "выше" : "ниже"} на ${fmt(Math.abs(dPcc))}.</p>`;
    box.hidden = false;
    btn.textContent = "Пересчитать GS-50";
  } catch (e) { toast(e.message); btn.textContent = "Сравнить с GS-50"; }
  finally { btn.disabled = false; }
}

// ---------------------------------------------------------------- 3D-просмотр
const viewer = { id: null, z: 24.1, pending: false, queued: null, playing: false, dir: 1, raf: 0 };

function initViewerFor(r) {
  viewer.id = r.id;
  const range = $("#zRange");
  const min = +range.min, max = +range.max;
  $("#zMarks").innerHTML = state.distances.map((z, i) =>
    `<span style="left:${((z - min) / (max - min)) * 100}%">${i + 1}</span>`).join("");
  viewer.z = state.distances[0];
  range.value = viewer.z;
  stopPlay();
  updateSlice(viewer.z);
}

function updateSlice(z) {
  $("#zBadge").textContent = zLabel(z);
  const near = state.distances.reduce((best, d, i) => (Math.abs(d - z) < Math.abs(state.distances[best] - z) ? i : best), 0);
  const inFocus = Math.abs(state.distances[near] - z) < 0.2;
  const fb = $("#focusBadge");
  fb.textContent = inFocus ? `в фокусе: ${near + 1}` : `ближайшая: ${near + 1}`;
  fb.classList.toggle("on", inFocus);
  const crit = state.info?.critical_distance_mm ?? 24.06;
  $("#zNote").textContent = z < crit ? `z < ${fmt(crit, 2)} мм — квадратичная фаза дискретизирована с алиасингом, картинка менее достоверна.` : "";
  requestSlice(z);
}

// Не больше одного запроса одновременно: новый z ставится в очередь, промежуточные пропускаются
function requestSlice(z) {
  if (viewer.pending) { viewer.queued = z; return; }
  viewer.pending = true;
  const img = new Image();
  const done = () => {
    viewer.pending = false;
    if (viewer.queued !== null) { const q = viewer.queued; viewer.queued = null; requestSlice(q); }
    else if (viewer.playing) stepPlay();
  };
  img.onload = () => { $("#sliceImg").src = img.src; done(); };
  img.onerror = () => { done(); };
  img.src = `${API}/api/results/${viewer.id}/slice.png?z_mm=${z.toFixed(2)}`;
}

function stepPlay() {
  const range = $("#zRange");
  const min = +range.min, max = +range.max;
  let z = Math.round((viewer.z + 0.1 * viewer.dir) * 100) / 100;
  if (z >= max) { z = max; viewer.dir = -1; }
  if (z <= min) { z = min; viewer.dir = 1; }
  viewer.z = z;
  range.value = z;
  viewer.raf = requestAnimationFrame(() => viewer.playing && updateSlice(z));
}
function stopPlay() {
  viewer.playing = false; cancelAnimationFrame(viewer.raf);
  $("#playLabel").textContent = "Пролететь";
}
function togglePlay() {
  if (viewer.playing) return stopPlay();
  viewer.playing = true; $("#playLabel").textContent = "Остановить";
  stepPlay();
}

// ---------------------------------------------------------------- прочее
function initResultsUi() {
  $("#runBtn").addEventListener("click", runSynthesis);
  $("#compareBtn").addEventListener("click", runCompare);
  $$("[data-doe]").forEach((b) => b.addEventListener("click", () => setDoeView(b.dataset.doe)));
  $$("[data-view]").forEach((b) => b.addEventListener("click", () => {
    state.view = b.dataset.view;
    $$("[data-view]").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    renderPlanes();
  }));
  $("#zRange").addEventListener("input", (e) => { stopPlay(); viewer.z = +e.target.value; updateSlice(viewer.z); });
  $("#playBtn").addEventListener("click", togglePlay);
}

// Подсветка под курсором: координаты в CSS-переменные --mx/--my
function initSpotlight() {
  const sel = ".studio, .how-card, .formula, .pipeline, .viewer, .stats, .doe-card, .table-card";
  $$(sel).forEach((el) => el.classList.add("spot"));
  document.addEventListener("pointermove", (e) => {
    const el = e.target.closest?.(".spot");
    if (!el) return;
    const r = el.getBoundingClientRect();
    el.style.setProperty("--mx", `${e.clientX - r.left}px`);
    el.style.setProperty("--my", `${e.clientY - r.top}px`);
  }, { passive: true });
}

function initReveal() {
  const els = $$(".section-head, .studio, .how-card, .formula, .number, .numbers-title, .pipeline");
  els.forEach((el) => el.classList.add("reveal"));
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  }, { threshold: 0.12 });
  els.forEach((el) => io.observe(el));
}

initNav();
initHero();
initSlots();
initRanges();
initResultsUi();
initReveal();
initSpotlight();
loadInfo().then(renderSlots);
loadTestsets();
