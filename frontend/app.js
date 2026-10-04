// Vanilla RAG — frontend logic.
// Talks to the FastAPI backend at a configurable base URL (persisted in
// localStorage). Because GitHub Pages hosts only this static UI, the backend
// must be reachable (default: a locally running `uvicorn app.main:app`).

const DEFAULT_API = "http://localhost:8000";
const SAMPLE_DOC = `Acme Corp Refund and Support Policy

Refunds are available within 30 days of purchase for all standard plans. To request a refund, contact support with your order number. Refunds are processed to the original payment method within 5 to 7 business days.

Enterprise contracts have custom refund terms defined in the master services agreement. Annual enterprise plans may be prorated at Acme's discretion.

Technical support is available 24/7 for enterprise customers via phone and email. Standard plan customers receive email support with a response target of one business day. Priority support can be added to any standard plan for an additional monthly fee.

Warranty covers manufacturing defects for a period of twelve months from the date of delivery. Damage from misuse, unauthorized repair, or normal wear is not covered under warranty.`;

const $ = (id) => document.getElementById(id);
let apiBase = localStorage.getItem("ragApiBase") || DEFAULT_API;

function setBanner(msg, kind = "warn") {
  const b = $("banner");
  if (!msg) { b.classList.add("hidden"); return; }
  b.textContent = msg;
  b.className = `banner ${kind}`;
}

function setStatus(state, text) {
  $("statusDot").className = `dot ${state}`;
  $("statusText").textContent = text;
}

function renderPipeline(info) {
  if (!info) { $("pipelineInfo").innerHTML = ""; return; }
  $("pipelineInfo").innerHTML =
    `embedder <code>${info.embedding_provider}</code> · ` +
    `model <code>${info.embedding_model}</code> (${info.embedding_dim}-d) · ` +
    `index <code>${info.vector_store}</code> · ` +
    `generator <code>${info.llm_provider}</code> · ` +
    `${info.documents} docs / ${info.chunks} chunks`;
}

async function api(path, options = {}) {
  const res = await fetch(apiBase.replace(/\/$/, "") + path, options);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); if (j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

async function checkHealth() {
  setStatus("", "connecting…");
  try {
    const h = await api("/health");
    setStatus("ok", `online · ${h.documents} docs / ${h.chunks} chunks`);
    renderPipeline(h);
    setBanner("");
    await refreshDocs();
    return true;
  } catch (err) {
    setStatus("bad", "offline");
    renderPipeline(null);
    setBanner(
      `Cannot reach the backend at ${apiBase}. This GitHub Pages site is only the UI — ` +
      `run the API locally with "uvicorn app.main:app --port 8000" (see the README), then click Connect. ` +
      `(${err.message})`
    );
    return false;
  }
}

async function refreshDocs() {
  try {
    const docs = await api("/documents");
    $("docCount").textContent = docs.length;
    const ul = $("docList");
    ul.innerHTML = "";
    if (!docs.length) {
      ul.innerHTML = `<li><span class="chunks">No documents yet — ingest some text to begin.</span></li>`;
      return;
    }
    for (const d of docs) {
      const li = document.createElement("li");
      li.innerHTML = `<span>${escapeHtml(d.source_id)}</span><span class="chunks">${d.num_chunks} chunks</span>`;
      ul.appendChild(li);
    }
  } catch (_) {}
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function linkifyCitations(text) {
  return escapeHtml(text).replace(/\[(\d+)\]/g, '<span class="cite" data-n="$1">[$1]</span>');
}

async function ingestText() {
  const text = $("docText").value.trim();
  if (!text) { setBanner("Paste some document text first."); return; }
  const sourceId = $("sourceId").value.trim() || undefined;
  const btn = $("ingestTextBtn");
  btn.disabled = true; btn.textContent = "Ingesting…";
  try {
    const r = await api("/ingest/text", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, source_id: sourceId, metadata: {} }),
    });
    setBanner(`Ingested "${r.source_id}" → ${r.num_chunks} chunks (embedder: ${r.embedding_provider}, index: ${r.vector_store}).`, "warn");
    $("docText").value = ""; $("sourceId").value = "";
    await checkHealth();
  } catch (err) { setBanner("Ingest failed: " + err.message, "warn"); }
  finally { btn.disabled = false; btn.textContent = "Ingest Text"; }
}

async function ingestFile(file) {
  const form = new FormData();
  form.append("file", file);
  try {
    const r = await api("/ingest/file", { method: "POST", body: form });
    setBanner(`Ingested file "${r.source_id}" → ${r.num_chunks} chunks.`, "warn");
    await checkHealth();
  } catch (err) { setBanner("File ingest failed: " + err.message, "warn"); }
}

async function ask() {
  const question = $("question").value.trim();
  if (!question) { setBanner("Type a question first."); return; }
  const topK = parseInt($("topK").value || "4", 10);
  const btn = $("askBtn");
  btn.disabled = true; const old = btn.textContent; btn.innerHTML = '<span class="spinner"></span>Thinking';
  try {
    const r = await api("/query", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: topK }),
    });
    $("answerWrap").classList.remove("hidden");
    $("answerText").innerHTML = linkifyCitations(r.answer);
    $("answerMeta").textContent =
      `generator: ${r.llm_provider} · retrieved: ${r.retrieved} · ` +
      `${r.timings_ms.retrieval}ms retrieval + ${r.timings_ms.generation}ms generation`;

    const wrap = $("sourcesWrap"); const ol = $("sourcesList");
    ol.innerHTML = "";
    if (r.citations.length) {
      wrap.classList.remove("hidden");
      r.citations.forEach((c, i) => {
        const li = document.createElement("li");
        li.id = `src-${i + 1}`;
        li.innerHTML =
          `<div class="src-head"><span>${escapeHtml(c.source_id)} · chunk #${c.chunk_index}</span>` +
          `<span class="score">cosine ${c.score.toFixed(3)}</span></div>` +
          `<div class="passage">${escapeHtml(c.text)}</div>`;
        ol.appendChild(li);
      });
      ol.querySelectorAll(".cite").forEach((el) => {
        el.addEventListener("click", () => {
          const n = el.getAttribute("data-n");
          const target = document.getElementById(`src-${n}`);
          if (target) { target.scrollIntoView({ behavior: "smooth", block: "center" }); target.classList.add("flash"); setTimeout(() => target.classList.remove("flash"), 1200); }
        });
      });
      $("answerText").querySelectorAll(".cite").forEach((el) => {
        el.addEventListener("click", () => {
          const n = el.getAttribute("data-n");
          const target = document.getElementById(`src-${n}`);
          if (target) { target.scrollIntoView({ behavior: "smooth", block: "center" }); target.classList.add("flash"); setTimeout(() => target.classList.remove("flash"), 1200); }
        });
      });
    } else { wrap.classList.add("hidden"); }
    setBanner("");
  } catch (err) { setBanner("Query failed: " + err.message, "warn"); }
  finally { btn.disabled = false; btn.textContent = old; }
}

function wire() {
  $("apiBase").value = apiBase;
  $("connectBtn").addEventListener("click", () => {
    apiBase = $("apiBase").value.trim() || DEFAULT_API;
    localStorage.setItem("ragApiBase", apiBase);
    checkHealth();
  });
  $("ingestTextBtn").addEventListener("click", ingestText);
  $("fileInput").addEventListener("change", (e) => { if (e.target.files[0]) ingestFile(e.target.files[0]); });
  $("sampleBtn").addEventListener("click", () => {
    $("docText").value = SAMPLE_DOC;
    $("sourceId").value = "acme-policy";
  });
  $("askBtn").addEventListener("click", ask);
  $("question").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask(); });

  // Footer links: GitHub repo + live API docs (relative to the backend).
  $("docsLink").href = apiBase.replace(/\/$/, "") + "/docs";
  $("docsLink").addEventListener("click", () => { window.open(apiBase.replace(/\/$/, "") + "/docs", "_blank"); });
}

wire();
checkHealth();
