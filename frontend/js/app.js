// ── ResumeX App.js ── Shared utilities
// ── Auth ─────────────────────────────────────────────────────────────────────
const AUTH = {
  get token()    { return localStorage.getItem("rx_tk"); },
  get username() { return localStorage.getItem("rx_un"); },
  get email()    { return localStorage.getItem("rx_em"); },
  set(t, u, e)   { localStorage.setItem("rx_tk", t); localStorage.setItem("rx_un", u); localStorage.setItem("rx_em", e || ""); },
  clear()        { ["rx_tk","rx_un","rx_em"].forEach(k => localStorage.removeItem(k)); },
  get ok()       { return !!this.token; }
};

function authHeaders() {
  return AUTH.token ? { "Authorization": `Bearer ${AUTH.token}` } : {};
}

// ── API Fetch ─────────────────────────────────────────────────────────────────
async function apiFetch(url, opts = {}) {
  const resp = await fetch(url, {
    ...opts,
    headers: { ...authHeaders(), ...(opts.headers || {}) }
  });
  const data = await resp.json().catch(() => ({ detail: "Unknown error" }));
  if (!resp.ok) throw new Error(data.detail || "Request failed");
  return data;
}

// ── Nav Setup ─────────────────────────────────────────────────────────────────
function setupNav() {
  const li = AUTH.ok;
  document.querySelectorAll(".auth-show").forEach(el => el.classList.toggle("hidden", !li));
  document.querySelectorAll(".guest-show").forEach(el => el.classList.toggle("hidden", li));

  const av = document.getElementById("nav-av");
  const nl = document.getElementById("nav-name-lbl");
  const ne = document.getElementById("nav-email-lbl");

  if (av) av.textContent = li && AUTH.username ? AUTH.username[0].toUpperCase() : "?";
  if (nl) nl.textContent = AUTH.username || "";
  if (ne) ne.textContent = AUTH.email || "";

  document.querySelectorAll("[data-logout]").forEach(el =>
    el.addEventListener("click", e => {
      e.preventDefault();
      AUTH.clear();
      window.location.href = "/";
    })
  );
}

function toggleDD() {
  document.getElementById("nav-dd")?.classList.toggle("open");
}

document.addEventListener("click", e => {
  if (!e.target.closest(".nav-avatar-wrap")) {
    document.getElementById("nav-dd")?.classList.remove("open");
  }
});

// ── Score Helpers ─────────────────────────────────────────────────────────────
function sCls(s)   { return s >= 55 ? "high" : s >= 25 ? "medium" : "low"; }
function sColor(s) { return { high: "var(--teal)", medium: "var(--warn)", low: "var(--danger)" }[sCls(s)]; }

function renderRing(score, el) {
  if (!el) return;
  const r = 56, c = 2 * Math.PI * r;
  const off = c - (Math.min(score, 100) / 100) * c;
  const cls = sCls(score);
  const id = "arc" + Math.random().toString(36).slice(2);
  el.innerHTML = `<div class="score-ring-wrap">
    <svg width="130" height="130" viewBox="0 0 130 130">
      <circle class="ring-bg" cx="65" cy="65" r="${r}"/>
      <circle class="ring-fill ${cls}" cx="65" cy="65" r="${r}"
        stroke-dasharray="${c}" stroke-dashoffset="${c}" id="${id}"/>
    </svg>
    <div class="score-center">
      <div class="score-big ${cls}">${Math.round(score)}%</div>
      <div class="score-lbl">match</div>
    </div>
  </div>`;
  setTimeout(() => {
    const arc = document.getElementById(id);
    if (arc) arc.style.strokeDashoffset = off;
  }, 80);
}

function renderBar(score, el) {
  if (!el) return;
  const cls = sCls(score);
  const labels = { high: "Strong match", medium: "Moderate match", low: "Weak match" };
  const bid = "bar" + Math.random().toString(36).slice(2);
  el.innerHTML = `<div class="score-bar-wrap">
    <div class="score-bar-label">
      <span>${labels[cls]}</span>
      <span style="color:${sColor(score)};font-weight:700">${Math.round(score)}%</span>
    </div>
    <div class="score-bar-track">
      <div class="score-bar-fill ${cls}" style="width:0%" id="${bid}"></div>
    </div>
  </div>`;
  setTimeout(() => {
    const bar = document.getElementById(bid);
    if (bar) bar.style.width = Math.min(score, 100) + "%";
  }, 80);
}

function chips(skills, cls) {
  if (!skills?.length) return '<span style="font-size:12px;color:var(--text3)">None found</span>';
  return `<div class="chips">${skills.map(s => `<span class="chip ${cls}">${s}</span>`).join("")}</div>`;
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(msg, type = "ok") {
  const t = document.createElement("div");
  t.className = `toast toast-${type}`;
  t.innerHTML = msg;
  document.body.appendChild(t);
  requestAnimationFrame(() => requestAnimationFrame(() => t.classList.add("in")));
  setTimeout(() => { t.classList.remove("in"); setTimeout(() => t.remove(), 400); }, 3500);
}

// ── Shared Nav HTML ───────────────────────────────────────────────────────────
function navHTML(activePage) {
  const pages = [
    { href: "/", label: "Home", id: "home" },
    { href: "/analyzer", label: "Analyzer", id: "analyzer" },
    { href: "/jobs", label: "Jobs", id: "jobs" },
    { href: "/dashboard", label: "Dashboard", id: "dashboard" },
  ];
  return pages.map(p =>
    `<a href="${p.href}" class="nav-link${activePage === p.id ? " active" : ""}">${p.label}</a>`
  ).join("");
}

document.addEventListener("DOMContentLoaded", setupNav);