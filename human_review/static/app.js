const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const storageKey = "actground.review.workspace.v1";
const videoExposureKey = "actground.review.videoExposure.v1";
const timelineExposureKey = "actground.review.timelineExposure.v1";
const initialQuery = new URLSearchParams(window.location.search);
const requestedField = initialQuery.get("field") || "";
const requestedNext = /^next_[123]$/.test(requestedField) ? requestedField : "";
let initialLinkPending = true;
let stored = {};
try { stored = JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch {}
const state = {
  bootstrap: null, examples: [], filtered: [], detail: null, id: initialQuery.get("example") || stored.id || "",
  reviewer: stored.reviewer || "", model: initialQuery.get("model") || stored.model || "", field: requestedNext ? "next" : ["why","goal","current","next"].includes(requestedField) ? requestedField : stored.field || "why",
  blind: !!stored.blind, auto: false, compare: false, future: false, futureSeen: false, frames: false,
  progress: {}, drafts: new Map(), saved: new Map(), timers: new Map(), queues: new Map(),
  focusedField: requestedNext || "next_1", nav: 0, failedSaves: new Map(), playing: false,
};
const flagNames = ["unsupported_addition", "contradiction", "too_broad", "too_narrow", "reference_issue", "mapper_issue"];
const flagLabels = {unsupported_addition:"Unsupported addition", contradiction:"Contradiction", too_broad:"Too broad", too_narrow:"Too narrow", reference_issue:"Reference issue", mapper_issue:"Mapping issue"};
const semanticTitles = {why:"Stated purpose", goal:"Active activity", current:"Current action", next:"Next three actions"};

function persist() {
  try { localStorage.setItem(storageKey, JSON.stringify({id:state.id, reviewer:state.reviewer, model:state.model, field:state.field, blind:state.blind})); } catch {}
}
function priorFullVideoExposure(videoId, queryTime) {
  try {
    const viewed = JSON.parse(localStorage.getItem(videoExposureKey) || "{}");
    const timelines = JSON.parse(localStorage.getItem(timelineExposureKey) || "{}");
    return Number(viewed[videoId] || 0) > queryTime || !!timelines[videoId];
  } catch { return false; }
}
function setStatus(text, kind = "") { $("save-status").textContent = text; $("save-status").className = `save-status ${kind}`; }
function error(message) { $("global-error").textContent = message; $("global-error").hidden = !message; }
async function api(url, options = {}) {
  const response = await fetch(url, {...options, headers:{...(options.body ? {"Content-Type":"application/json", "X-Review-Token":state.bootstrap?.csrf_token || ""} : {}), ...options.headers}});
  let data;
  try { data = await response.json(); } catch { throw new Error(`Server returned ${response.status}; check the review server.`); }
  if (!response.ok) throw new Error(data.error || data.message || `Request failed (${response.status})`);
  return data;
}
function safeURL(value) {
  if (typeof value !== "string" || !value.trim()) return "";
  try { const url = new URL(value, window.location.origin); return url.origin === window.location.origin && ["http:", "https:"].includes(url.protocol) ? url.href : ""; } catch { return ""; }
}
function timestamp(value) { const number = Number(value); if (!Number.isFinite(number)) return "—"; const seconds = Math.max(0, number); return `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(2).padStart(5,"0")}`; }
function textOf(value) { if (typeof value === "string") return value; return value?.text ?? value?.action ?? ""; }
function modelName(id) { const index = (state.bootstrap?.models || []).findIndex((m) => m.id === id); return state.blind ? `Model ${String.fromCharCode(65 + Math.max(index,0))}` : (state.bootstrap?.models || []).find((m) => m.id === id)?.name || id; }
function keyFor(id, model, field, reviewer = state.reviewer) { return JSON.stringify([reviewer, id, model, field]); }
function activePrediction() { return state.detail?.predictions.find((p) => p.model === state.model); }
function fieldEligible(field) {
  if (!state.detail) return false;
  if (field.startsWith("next_")) return !!state.detail.reference.next[Number(field.split("_")[1]) - 1];
  if (field === "next") return !!state.detail.reference.next.length;
  return !!state.detail.reference[field]?.eligible;
}
function actionReference(field) { return field.startsWith("next_") ? state.detail?.reference.next[Number(field.split("_")[1]) - 1] : state.detail?.reference.current; }
function completeActionReference(field) { const pair = actionReference(field)?.pair; return Array.isArray(pair) && pair.length === 2 && pair.every((id) => id !== null && id !== undefined); }
function draftFor(field) {
  const key = keyFor(state.id, state.model, field);
  if (!state.drafts.has(key)) {
    const saved = state.saved.get(key);
    state.drafts.set(key, {label:saved?.label || null, notes:saved?.notes || "", flags:{...Object.fromEntries(flagNames.map((f) => [f,false])), ...saved?.flags}, verb_correct:saved?.verb_correct ?? null, noun_correct:saved?.noun_correct ?? null, dirty:false});
  }
  return state.drafts.get(key);
}
function labelChoices(field, draft, enabled = true) {
  const semantic = field === "why" || field === "goal";
  const options = semantic ? [["correct","Correct","C"],["partial","Partial","P"],["incorrect","Incorrect","I"],["uncertain","Uncertain","U"]] : [["correct","Correct","C"],["incorrect","Incorrect","I"],["uncertain","Uncertain","U"]];
  const partialGold = !semantic && !completeActionReference(field);
  return `<div class="label-heading">Your assessment${partialGold ? " · joint reference incomplete" : ""}</div><div class="label-choices" role="group" aria-label="${esc(field)} assessment">${options.map(([label,title,shortcut]) => `<button class="label-button ${draft.label === label ? "selected" : ""}" data-review-field="${esc(field)}" data-label="${label}" aria-pressed="${draft.label === label}" ${enabled && (!partialGold || label === "uncertain") ? "" : "disabled"}>${title}<span class="shortcut">${shortcut}</span></button>`).join("")}</div>`;
}
function componentChoices(field, draft, enabled = true) {
  const pair = actionReference(field)?.pair || [];
  return `<div class="component-choices">${["verb","noun"].map((component,index) => `<div class="component-group"><span>${component[0].toUpperCase()+component.slice(1)}${pair[index] == null ? " (unlabeled)" : ""}</span>${[[true,"✓"],[false,"✕"],[null,"?"]].map(([value,title]) => `<button class="component-button ${draft[`${component}_correct`] === value ? "selected" : ""}" data-review-field="${esc(field)}" data-component="${component}" data-value="${value}" aria-label="${component} ${value === true ? "correct" : value === false ? "incorrect" : "uncertain"}" aria-pressed="${draft[`${component}_correct`] === value}" ${enabled && pair[index] != null ? "" : "disabled"}>${title}</button>`).join("")}</div>`).join("")}</div>`;
}
function noteControls(field, draft, enabled = true) {
  const flags = field === "why" || field === "goal" ? flagNames.filter((f) => f !== "mapper_issue") : ["reference_issue", "mapper_issue"];
  const saved = state.saved.get(keyFor(state.id,state.model,field));
  return `<div class="flags-grid">${flags.map((flag) => `<label class="flag-label"><input type="checkbox" data-review-field="${esc(field)}" data-flag="${flag}" ${draft.flags[flag] ? "checked" : ""} ${enabled ? "" : "disabled"}>${flagLabels[flag]}</label>`).join("")}</div><textarea class="annotation-notes" data-review-field="${esc(field)}" aria-label="${esc(field)} review notes" placeholder="Notes or evidence for your decision…" ${enabled ? "" : "disabled"}>${esc(draft.notes)}</textarea><div class="saved-label" data-saved-field="${esc(field)}">${saved ? `Saved${saved.updated_at ? ` · ${esc(new Date(saved.updated_at).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"}))}` : ""}` : "No human label yet"}</div>`;
}
function autoBlock(assessment) {
  if (!state.auto) return "";
  if (!assessment) return '<div class="auto-assessment"><strong>AUTOMATED ASSESSMENT</strong>Pending or unavailable. Your review is independent.</div>';
  if (assessment.label) return `<div class="auto-assessment"><strong>AUTOMATED ASSESSMENT · ${esc(assessment.label)}</strong>${esc(assessment.justification || assessment.evidence_note || "Provisional semantic assessment.")}</div>`;
  return `<div class="auto-assessment"><strong>AUTOMATED CLASS COMPARISON</strong>Verb: ${assessment.verb === null ? "unavailable" : assessment.verb ? "match" : "mismatch"} · Noun: ${assessment.noun === null ? "unavailable" : assessment.noun ? "match" : "mismatch"} · Pair: ${assessment.pair === null ? "unavailable" : assessment.pair ? "match" : "mismatch"}</div>`;
}
function classChips(reference) {
  if (!reference) return "";
  const pair = reference.pair || [];
  return `<div class="class-chips"><span class="class-chip">Verb: ${esc(reference.verb ?? pair[0] ?? "unlabeled")}</span><span class="class-chip">Noun: ${esc(reference.noun ?? pair[1] ?? "unlabeled")}</span>${reference.start_sec != null ? `<span class="class-chip">${timestamp(reference.start_sec)}–${timestamp(reference.end_sec)}</span>` : ""}</div>`;
}
function renderReference() {
  const detail = state.detail, field = state.field, reference = detail.reference[field];
  $("reference-title").textContent = semanticTitles[field];
  let html = "";
  if (field === "next") {
    html = '<p class="source-note">Three sequential fine-grained actions. Order matters.</p>' + detail.reference.next.map((action,i) => `<div class="action-reference"><strong>${i + 1}. ${esc(textOf(action))}</strong>${classChips(action)}</div>`).join("");
  } else if (!reference?.eligible) {
    html = `<p class="empty-answer">${field === "why" && detail.task === "track" ? "WHY is not requested for this Track example." : "This reference is unavailable or excluded. Human scoring is disabled for this field."}</p><p class="source-note">${esc(reference?.source_status || "Not applicable")}</p>`;
  } else {
    html = `<span class="source-tag">${esc(reference.source_status || "Annotated")}</span><div class="reference-text">${esc(textOf(reference))}</div>`;
    if (field === "current") html += `<div style="margin-top:10px">${classChips(reference)}</div>`;
    if (field === "why" && reference.original_narration) html += `<details><summary>Original narration</summary><p class="narration">${esc(reference.original_narration)}</p></details>`;
    if (field === "goal" && reference.units?.length > 1) html += `<p class="source-note">Required activity units</p><ul class="unit-list">${reference.units.map((unit) => `<li>${esc(unit)}</li>`).join("")}</ul>`;
    if (field === "goal") html += '<p class="source-note">Use the activity annotation’s own scope; it may be a step or a broader activity.</p>';
  }
  $("reference-content").innerHTML = `<div class="reference-content">${html}</div>`;
}
function predictionField(prediction, field) {
  if (field === "next") return (prediction.fields.next || []).map((action,i) => `${i + 1}. ${textOf(action) || "[missing]"}`).join("\n");
  return textOf(prediction.fields[field] ?? (field === "current" ? prediction.fields.action : null));
}
function renderPrediction() {
  const prediction = activePrediction();
  if (!prediction) { $("prediction-content").innerHTML = '<p class="empty-state">No saved output for this model.</p>'; return; }
  const field = state.field;
  let html = "";
  if (field === "next") {
    html = state.detail.reference.next.map((reference,i) => {
      const name = `next_${i + 1}`, draft = draftFor(name), text = textOf(prediction.fields.next?.[i]), enabled = fieldEligible(name);
      return `<div class="next-row" data-next-position="${i+1}"><div class="next-row-heading"><span class="position-badge">${i+1}</span><p class="prediction-text">${esc(text || "No extractable action")}</p></div>${autoBlock(prediction.auto?.next?.[i])}${componentChoices(name,draft,enabled)}${labelChoices(name,draft,enabled)}${noteControls(name,draft,enabled)}</div>`;
    }).join("");
  } else {
    const draft = draftFor(field), text = predictionField(prediction,field), enabled = fieldEligible(field);
    html = `<div class="${text ? "prediction-text" : "empty-answer"}">${esc(text || "No extractable answer for this field")}</div>${autoBlock(prediction.auto?.[field])}${field === "current" ? componentChoices(field,draft,enabled) : ""}${labelChoices(field,draft,enabled)}${noteControls(field,draft,enabled)}`;
  }
  const rawUnavailable = prediction.raw_available === false || prediction.raw === null;
  html += `<details class="raw-output"><summary>${rawUnavailable ? "Full output unavailable" : "Full saved output"}</summary><pre>${esc(rawUnavailable ? "The raw saved output is unavailable. Extracted fields remain available above." : prediction.raw || "The model returned an empty output.")}</pre></details>`;
  $("prediction-content").innerHTML = `<div class="answer-body">${html}</div>`;
  renderComparison();
}
function renderComparison() {
  $("comparison-content").hidden = !state.compare;
  if (!state.compare) return;
  const others = state.detail.predictions.filter((p) => p.model !== state.model);
  $("comparison-content").innerHTML = `<div class="comparison-section"><p class="label-heading">Other saved outputs · view only; labels above apply to ${esc(modelName(state.model))}</p><div class="comparison-grid">${others.length ? others.map((p) => `<div class="comparison-card"><h4>${esc(modelName(p.model))}</h4><p>Switch the active model to review this answer.</p><div class="comparison-answer">${esc(predictionField(p,state.field) || "No extractable answer")}</div><button class="button subtle" style="margin-top:9px" data-activate-model="${esc(p.model)}">Review this model</button></div>`).join("") : '<p class="quiet">No additional models for this example.</p>'}</div></div>`;
}
function renderTabs() {
  document.querySelectorAll(".field-tab").forEach((button) => {
    const field = button.dataset.field;
    button.classList.toggle("selected", field === state.field);
    button.setAttribute("aria-selected", String(field === state.field));
    button.disabled = field === "why" && state.detail.task !== "explain" || field === "current" && state.detail.task !== "track";
  });
}
function renderModelSelect() {
  $("model-select").innerHTML = state.detail.predictions.map((p) => `<option value="${esc(p.model)}">${esc(modelName(p.model))}</option>`).join("");
  $("model-select").value = state.model;
}
function renderWorkspace() {
  const detail = state.detail;
  $("timeline-workspace-link").href = `/timeline?video=${encodeURIComponent(detail.video_id)}`;
  $("question-heading").textContent = detail.question || "Review this example";
  $("example-id").textContent = detail.id;
  $("task-badge").textContent = detail.task.toUpperCase();
  $("video-label").textContent = detail.video_id;
  $("query-label").textContent = `QUERY ${timestamp(detail.query_time_sec)}`;
  $("warnings").hidden = !state.bootstrap.warnings?.length;
  $("warnings").textContent = (state.bootstrap.warnings || []).join(" · ");
  renderModelSelect(); renderTabs(); renderReference(); renderPrediction(); renderEvidence();
  $("review-workspace").hidden = false;
  $("loading-state").hidden = true;
  $("workspace").setAttribute("aria-busy","false");
  $("review-progress-text").textContent = `${state.filtered.findIndex((e) => e.id === state.id) + 1} of ${state.filtered.length} filtered examples`;
  $("previous-button").disabled = state.filtered.findIndex((e) => e.id === state.id) <= 0;
  $("next-button").disabled = state.filtered.findIndex((e) => e.id === state.id) >= state.filtered.length - 1;
}
function reviewStatus(example) {
  const status = example.review_state || {};
  const model = status.by_model?.[state.model];
  if (model) return model.uncertain ? "uncertain" : model.complete ? "reviewed" : "unreviewed";
  if (status.reviewed_models?.includes(state.model)) return "reviewed";
  if (status.uncertain) return "uncertain";
  if (status.complete) return "reviewed";
  return "unreviewed";
}
function renderExamples() {
  const query = $("search").value.trim().toLowerCase(), task = $("task-filter").value, video = $("video-filter").value, status = $("status-filter").value;
  state.filtered = state.examples.filter((e) => (!task || e.task === task) && (!video || e.video_id === video) && (!query || `${e.id} ${e.video_id} ${e.question}`.toLowerCase().includes(query)) && (!status || status === "mappingerror" ? (!status || !!e.mapping_error) : reviewStatus(e) === status));
  $("example-count").textContent = state.filtered.length.toLocaleString();
  $("example-list").innerHTML = state.filtered.length ? state.filtered.map((example) => `<button class="example-item ${example.id === state.id ? "selected" : ""}" data-example-id="${esc(example.id)}" role="listitem" aria-current="${example.id === state.id ? "true" : "false"}"><span class="row-meta">${esc(example.task.toUpperCase())}<span>· ${timestamp(example.query_time_sec)}</span><span class="status-dot ${example.mapping_error ? "mappingerror" : reviewStatus(example)}" title="${esc(reviewStatus(example))}"></span></span><span class="row-title">${esc(example.question || "Action review")}</span><span class="row-video">${esc(example.video_id)}</span></button>`).join("") : '<p class="empty-state">No examples match these filters.</p>';
  updateProgress();
}
function updateProgress() {
  const reviewed = Number(state.progress.reviewed || 0), total = Number(state.progress.total_predictions || 0), percent = total ? reviewed / total * 100 : 0;
  $("progress-label").textContent = `${reviewed.toLocaleString()} / ${total.toLocaleString()} outputs reviewed`;
  $("progress-percent").textContent = `${Math.round(percent)}%`;
  $("progress-fill").style.width = `${Math.min(100, percent)}%`;
}
let refreshTimer;
async function refreshOverview() {
  const reviewer = state.reviewer;
  const [examples,progress] = await Promise.all([api(`/api/examples?reviewer=${encodeURIComponent(reviewer)}`),api(`/api/progress?reviewer=${encodeURIComponent(reviewer)}`)]);
  if (reviewer !== state.reviewer) return;
  state.examples = examples.examples; state.progress = progress; renderExamples();
}
function scheduleOverview() { clearTimeout(refreshTimer); refreshTimer = setTimeout(() => refreshOverview().catch((e) => error(e.message)),500); }

async function queueSave(field) {
  if (!state.reviewer.trim()) { error("Enter a reviewer name before saving a human label."); $("reviewer").focus(); return; }
  const prediction = activePrediction(), draft = draftFor(field);
  if (!prediction || !draft.label || !fieldEligible(field)) return;
  const key = keyFor(state.id,state.model,field), captured = {...draft, flags:{...draft.flags}};
  const payload = {reviewer:state.reviewer, id:state.id, model:state.model, answer_sha256:prediction.answer_sha256, field, label:captured.label, notes:captured.notes, flags:captured.flags, evidence:state.futureSeen ? "future" : "observed"};
  if (field === "current" || field.startsWith("next_")) { payload.verb_correct = captured.verb_correct; payload.noun_correct = captured.noun_correct; }
  draft.dirty = false;
  setStatus("Saving…","pending");
  const previous = state.queues.get(key) || Promise.resolve();
  const promise = previous.catch(() => {}).then(async () => {
    payload.expected_revision = state.saved.get(key)?.revision || 0;
    const response = await api("/api/labels", {method:"POST", body:JSON.stringify(payload)});
    const saved = typeof response.saved === "object" ? response.saved : {...payload, revision:response.revision, updated_at:response.updated_at};
    state.saved.set(key,saved); state.failedSaves.delete(key);
    const node = document.querySelector(`[data-saved-field="${CSS.escape(field)}"]`);
    if (state.id === payload.id && state.model === payload.model && state.reviewer === payload.reviewer && node) node.textContent = "Saved just now";
    scheduleOverview();
  }).catch((e) => { state.failedSaves.set(key, {payload,error:e.message}); setStatus("Save failed","error"); error(`Label was not saved: ${e.message}`); throw e; }).finally(() => {
    if (state.queues.get(key) === promise) state.queues.delete(key);
    if (!state.queues.size && !state.failedSaves.size) { setStatus("All saved"); error(""); }
  });
  state.queues.set(key,promise);
  promise.catch(() => {});
  return promise;
}
function scheduleSave(field) {
  const key = keyFor(state.id,state.model,field);
  clearTimeout(state.timers.get(key));
  state.timers.set(key,setTimeout(() => { state.timers.delete(key); queueSave(field)?.catch(() => {}); },600));
}
async function flushSaves() {
  for (const [key,timer] of [...state.timers]) {
    clearTimeout(timer); state.timers.delete(key);
    const [reviewer,id,model,field] = JSON.parse(key);
    if (reviewer === state.reviewer && id === state.id && model === state.model) await queueSave(field);
  }
  await Promise.all([...state.queues.values()]);
  if (state.failedSaves.size) throw new Error("Resolve the save error before navigating. Your draft remains in this page.");
}
async function loadExample(id) {
  const nav = ++state.nav;
  try {
    await flushSaves();
    if (nav !== state.nav) return;
    pauseVideo(); error(""); $("workspace").setAttribute("aria-busy","true");
    const reviewer = state.reviewer;
    const detail = await api(`/api/example?id=${encodeURIComponent(id)}&reviewer=${encodeURIComponent(reviewer)}`);
    if (nav !== state.nav) return;
    if (reviewer !== state.reviewer) { await loadExample(id); return; }
    const previousTask = state.detail?.task;
    state.detail = detail; state.id = id; state.future = false; state.futureSeen = (detail.labels || []).some((label) => label.evidence === "future") || priorFullVideoExposure(detail.video_id, detail.query_time_sec) || initialLinkPending && initialQuery.get("example") === id && initialQuery.get("evidence") === "future";
    if (!detail.predictions.some((p) => p.model === state.model)) state.model = detail.predictions[0]?.model || "";
    if (previousTask && previousTask !== detail.task || state.field === "why" && detail.task !== "explain" || state.field === "current" && detail.task !== "track") state.field = detail.task === "explain" ? "why" : "goal";
    // Clean drafts reload from the server. Keep genuinely unsaved notes, but
    // never display an older local label with a newer server revision attached.
    for (const [key,draft] of state.drafts) {
      const [reviewer,example] = JSON.parse(key);
      if (reviewer === state.reviewer && example === id && !draft.dirty) state.drafts.delete(key);
    }
    for (const [key] of state.saved) {
      const [reviewer,example] = JSON.parse(key);
      if (reviewer === state.reviewer && example === id) state.saved.delete(key);
    }
    for (const label of detail.labels || []) state.saved.set(keyFor(id,label.model,label.field,label.reviewer || state.reviewer),label);
    persist(); renderExamples(); renderWorkspace();
    if (initialLinkPending && requestedNext && state.field === "next") {
      const target = document.querySelector(`[data-next-position="${requestedNext.split("_")[1]}"]`);
      target?.querySelector("button:not(:disabled)")?.focus({preventScroll:true});
      target?.scrollIntoView({block:"center"});
    }
    initialLinkPending = false;
    const selected = document.querySelector(".example-item.selected"); selected?.scrollIntoView({block:"nearest",inline:"nearest"});
  } catch (e) { error(e.message); $("workspace").setAttribute("aria-busy","false"); }
}
async function navigate(direction) {
  const index = state.filtered.findIndex((e) => e.id === state.id), next = state.filtered[index + direction];
  if (next) await loadExample(next.id);
}
async function switchModel(model) { try { await flushSaves(); state.model = model; persist(); renderModelSelect(); renderPrediction(); renderExamples(); } catch(e) { error(e.message); renderModelSelect(); } }
async function switchField(field) { try { await flushSaves(); state.field = field; persist(); renderTabs(); renderReference(); renderPrediction(); } catch(e) { error(e.message); } }

function evidenceBounds() {
  const media = state.detail?.media;
  const start = Number(media?.observation_start_sec || 0), query = Number(media?.query_time_sec ?? state.detail?.query_time_sec ?? 0);
  return {start,end:state.future ? Math.max(query,Number(media?.future_end_sec || query)) : query,query};
}
function pauseVideo() { $("evidence-video").pause(); state.playing = false; $("play-button").textContent = state.future ? "▶ Play evidence" : "▶ Play observed"; }
function updatePlayback() {
  const video = $("evidence-video"), bounds = evidenceBounds();
  $("playback-time").textContent = timestamp(video.currentTime);
  $("video-scrubber").value = String(Math.min(bounds.end,Math.max(bounds.start,video.currentTime)));
  if (video.currentTime >= bounds.end && !video.paused) { pauseVideo(); video.currentTime = bounds.end; }
  if (video.currentTime > bounds.end + .03 || video.currentTime < bounds.start - .03) { pauseVideo(); video.currentTime = Math.min(bounds.end,Math.max(bounds.start,video.currentTime)); }
}
function seekTo(time) {
  const bounds = evidenceBounds(), video = $("evidence-video"); pauseVideo();
  video.currentTime = Math.min(bounds.end,Math.max(bounds.start,Number(time))); updatePlayback();
}
function renderEvidence() {
  const media = state.detail.media, video = $("evidence-video"), bounds = evidenceBounds();
  const source = safeURL(media.video?.url);
  if (video.dataset.source !== source) { video.dataset.source = source; if (source) { video.src = source; video.load(); } else { video.removeAttribute("src"); video.load(); } }
  $("video-unavailable").hidden = !!(media.video?.available && source);
  $("play-button").disabled = !(media.video?.available && source);
  $("video-scrubber").min = String(bounds.start); $("video-scrubber").max = String(bounds.end);
  $("window-label").textContent = `${timestamp(bounds.start)}–${timestamp(bounds.end)}`;
  $("future-toggle").classList.toggle("active",state.future);
  $("future-toggle").setAttribute("aria-pressed",String(state.future));
  $("future-toggle").textContent = state.future ? "Future enabled" : "Show future";
  $("evidence-badge").textContent = state.future ? "FUTURE ENABLED" : "OBSERVED";
  $("evidence-badge").classList.toggle("future",state.future);
  $("evidence-description").textContent = state.future ? "Future evidence is enabled. Labels saved now record this context." : state.futureSeen ? "The player stops at the query. Earlier full-video context is recorded with your labels." : "The model’s observed window ends at the query.";
  $("play-button").textContent = state.future ? "▶ Play evidence" : "▶ Play observed";
  const events = [{label:"Observed window",time:bounds.start,kind:"past"},{label:"Query",time:bounds.query,kind:"query"}];
  if (state.future) state.detail.reference.next.forEach((action,i) => events.push({label:`${i+1}. ${textOf(action)}`,time:action.start_sec,kind:"future"}));
  $("timeline-events").innerHTML = events.map((event) => `<button class="event-button ${event.kind}" data-seek="${esc(event.time)}">${esc(event.label)}<span>${timestamp(event.time)}</span></button>`).join("");
  $("history-count").textContent = `(${state.detail.history.length})`;
  $("history-list").innerHTML = state.detail.history.map((action) => `<li>${esc(textOf(action))} <span>${timestamp(action.end_sec)}</span></li>`).join("") || "<li>No completed actions supplied.</li>";
  $("frames-panel").hidden = !state.frames;
  $("frames-toggle").setAttribute("aria-expanded",String(state.frames));
  $("frames-toggle").textContent = `Input frames (${media.frames.length})`;
  $("filmstrip").innerHTML = media.frames.map((frame,i) => `<button class="frame-button" data-frame="${i}" aria-label="Enlarge input frame ${i+1} at ${timestamp(frame.time_sec)}" ${frame.available && safeURL(frame.url) ? "" : "disabled"}>${frame.available && safeURL(frame.url) ? `<img src="${esc(safeURL(frame.url))}" alt="Model input frame ${i+1}" loading="lazy">` : '<div class="missing-frame">Unavailable</div>'}<span>${timestamp(frame.time_sec)}</span></button>`).join("");
  seekTo(bounds.start);
}
let animation;
function trackPlayback() { updatePlayback(); if (!$("evidence-video").paused) animation = requestAnimationFrame(trackPlayback); }
async function togglePlay() {
  const video = $("evidence-video"), bounds = evidenceBounds();
  if (video.paused) {
    if (video.currentTime >= bounds.end - .03 || video.currentTime < bounds.start) video.currentTime = bounds.start;
    try { await video.play(); state.playing = true; $("play-button").textContent = "Ⅱ Pause"; cancelAnimationFrame(animation); trackPlayback(); } catch(e) { error(`Could not play evidence: ${e.message}`); }
  } else pauseVideo();
}
function openModal(title, html) { $("modal-title").textContent = title; $("modal-content").innerHTML = html; if (!$("modal").open) $("modal").showModal(); }
function showRubric() {
  const rubric = state.bootstrap?.rubrics[state.field] || "";
  openModal(`${semanticTitles[state.field]} · review rubric`, `<div class="rubric-text">${esc(typeof rubric === "string" ? rubric : JSON.stringify(rubric,null,2))}</div><div class="shortcuts-grid"><span><span class="keycap">J</span> / <span class="keycap">K</span></span><span>Previous / next example</span><span><span class="keycap">C</span> <span class="keycap">P</span> <span class="keycap">I</span> <span class="keycap">U</span></span><span>Correct / partial / incorrect / uncertain</span><span><span class="keycap">Space</span></span><span>Play / pause the allowed evidence window</span></div><p class="source-note" style="margin-top:18px">Shortcuts are disabled while typing. For NEXT, shortcuts apply to the last focused action row.</p>`);
}

$("example-list").addEventListener("click",(event) => { const item = event.target.closest("[data-example-id]"); if (item) loadExample(item.dataset.exampleId); });
for (const id of ["search","task-filter","video-filter","status-filter"]) $(id).addEventListener(id === "search" ? "input" : "change",renderExamples);
$("field-tabs").addEventListener("click",(event) => { const button = event.target.closest("[data-field]"); if (button && !button.disabled) switchField(button.dataset.field); });
$("model-select").addEventListener("change",() => switchModel($("model-select").value));
$("blind-toggle").addEventListener("change",() => { state.blind = $("blind-toggle").checked; if (state.blind) { state.auto = false; $("auto-toggle").checked = false; } persist(); renderModelSelect(); renderPrediction(); });
$("auto-toggle").addEventListener("change",() => { state.auto = $("auto-toggle").checked; renderPrediction(); });
$("compare-toggle").addEventListener("change",() => { state.compare = $("compare-toggle").checked; renderComparison(); });
$("comparison-content").addEventListener("click",(event) => { const button = event.target.closest("[data-activate-model]"); if (button) switchModel(button.dataset.activateModel); });
$("prediction-content").addEventListener("click",(event) => {
  const button = event.target.closest("button[data-review-field]");
  if (!button || button.disabled) return;
  const field = button.dataset.reviewField, draft = draftFor(field); state.focusedField = field;
  if (button.dataset.label) {
    draft.label = button.dataset.label;
    if (field === "current" || field.startsWith("next_")) {
      if (draft.label === "correct" && completeActionReference(field)) {
        draft.verb_correct = true; draft.noun_correct = true;
      } else if (draft.label === "incorrect" && draft.verb_correct === true && draft.noun_correct === true
                 || draft.label === "uncertain" && (draft.verb_correct === false || draft.noun_correct === false
                     || draft.verb_correct === true && draft.noun_correct === true)) {
        draft.verb_correct = null; draft.noun_correct = null;
      }
    }
    if (draft.label !== "incorrect") draft.flags.contradiction = false;
  }
  if (button.dataset.component) {
    draft[`${button.dataset.component}_correct`] = button.dataset.value === "true" ? true : button.dataset.value === "false" ? false : null;
    draft.label = !completeActionReference(field) ? "uncertain" : draft.verb_correct === false || draft.noun_correct === false ? "incorrect" : draft.verb_correct === true && draft.noun_correct === true ? "correct" : "uncertain";
  }
  draft.dirty = true; renderPrediction(); queueSave(field).catch(() => {});
});
$("prediction-content").addEventListener("input",(event) => {
  const element = event.target, field = element.dataset.reviewField;
  if (!field) return; state.focusedField = field;
  const draft = draftFor(field);
  if (element.matches("textarea")) draft.notes = element.value;
  if (element.dataset.flag) {
    draft.flags[element.dataset.flag] = element.checked;
    if (element.dataset.flag === "contradiction" && element.checked && draft.label) {
      draft.label = "incorrect";
      document.querySelectorAll(`button[data-review-field="${CSS.escape(field)}"][data-label]`).forEach((button) => {
        const selected = button.dataset.label === "incorrect"; button.classList.toggle("selected",selected); button.setAttribute("aria-pressed",String(selected));
      });
    }
  }
  draft.dirty = true; if (draft.label) scheduleSave(field);
});
$("prediction-content").addEventListener("focusin",(event) => { if (event.target.dataset.reviewField) state.focusedField = event.target.dataset.reviewField; });
$("reviewer").value = state.reviewer;
$("reviewer").addEventListener("change",async () => {
  const nav = state.nav, identity = state.id;
  try {
    await flushSaves();
    state.reviewer = $("reviewer").value.trim(); persist(); state.drafts.clear(); state.saved.clear();
    if (state.detail) renderPrediction();
    await refreshOverview();
    // A user's newer navigation wins over this background reviewer refresh.
    if (identity && state.nav === nav && state.id === identity) await loadExample(identity);
  }
  catch(e) { $("reviewer").value = state.reviewer; error(e.message); }
});
$("previous-button").addEventListener("click",() => navigate(-1));
$("next-button").addEventListener("click",() => navigate(1));
$("play-button").addEventListener("click",togglePlay);
$("evidence-video").addEventListener("timeupdate",updatePlayback);
$("evidence-video").addEventListener("seeking",updatePlayback);
$("evidence-video").addEventListener("loadedmetadata",() => seekTo(evidenceBounds().start));
$("evidence-video").addEventListener("error",() => { $("video-unavailable").hidden = false; $("play-button").disabled = true; });
$("evidence-video").addEventListener("ended",pauseVideo);
$("video-scrubber").addEventListener("input",() => seekTo($("video-scrubber").value));
$("future-toggle").addEventListener("click",() => { pauseVideo(); state.future = !state.future; if (state.future) state.futureSeen = true; renderEvidence(); });
$("frames-toggle").addEventListener("click",() => { state.frames = !state.frames; $("frames-panel").hidden = !state.frames; $("frames-toggle").setAttribute("aria-expanded",String(state.frames)); });
$("timeline-events").addEventListener("click",(event) => { const button = event.target.closest("[data-seek]"); if (button) seekTo(button.dataset.seek); });
$("filmstrip").addEventListener("click",(event) => { const button = event.target.closest("[data-frame]"); if (!button || button.disabled) return; const frame = state.detail.media.frames[Number(button.dataset.frame)]; openModal(`Input frame ${Number(button.dataset.frame)+1} · ${timestamp(frame.time_sec)}`, `<img class="modal-frame" src="${esc(safeURL(frame.url))}" alt="Enlarged actual model input frame">`); });
$("help-button").addEventListener("click",showRubric); $("field-help").addEventListener("click",showRubric);
$("modal-close").addEventListener("click",() => $("modal").close());
$("modal").addEventListener("click",(event) => { if (event.target === $("modal")) $("modal").close(); });
$("export-button").addEventListener("click",async () => {
  try { await flushSaves(); const reviewer = encodeURIComponent(state.reviewer); openModal("Export your review", `<div class="export-options"><a class="export-link" href="/api/export?format=jsonl&reviewer=${reviewer}" download>Human labels · JSONL<span>All current field labels, notes, evidence and revision metadata.</span></a><a class="export-link" href="/api/export?format=csv&reviewer=${reviewer}" download>Human labels · CSV<span>Readable table for analysis and annotation review.</span></a><a class="export-link" href="/api/export?format=judgments&reviewer=${reviewer}" download>Semantic judgments · JSONL<span>Scored WHY / GOAL judgments bound to evaluation packets.</span></a></div>`); } catch(e) { error(e.message); }
});
document.addEventListener("keydown",(event) => {
  if (event.target.closest("input,textarea,select,[contenteditable=true]") || $("modal").open || event.metaKey || event.ctrlKey || event.altKey || !state.detail) return;
  const key = event.key.toLowerCase();
  if (key === "j") { event.preventDefault(); navigate(-1); }
  else if (key === "k") { event.preventDefault(); navigate(1); }
  else if (key === " ") { event.preventDefault(); if (!$("play-button").disabled) togglePlay(); }
  else if ({c:"correct",p:"partial",i:"incorrect",u:"uncertain"}[key]) {
    const field = state.field === "next" ? state.focusedField.startsWith("next_") ? state.focusedField : "next_1" : state.field;
    if (key === "p" && !["why","goal"].includes(field) || !fieldEligible(field)) return;
    const button = document.querySelector(`button[data-review-field="${CSS.escape(field)}"][data-label="${{c:"correct",p:"partial",i:"incorrect",u:"uncertain"}[key]}"]`);
    if (button) { event.preventDefault(); button.click(); }
  }
});
window.addEventListener("beforeunload",(event) => {
  const dirty = [...state.drafts.values()].some((draft) => draft.dirty);
  if (dirty || state.queues.size || state.failedSaves.size) { event.preventDefault(); event.returnValue = ""; }
});

async function bootstrap() {
  try {
    state.bootstrap = await api("/api/bootstrap");
    document.title = state.bootstrap.title || "ActGround · Human review";
    $("blind-toggle").checked = state.blind;
    $("task-filter").innerHTML += (state.bootstrap.tasks || []).map((task) => `<option value="${esc(task)}">${esc(task[0].toUpperCase()+task.slice(1))}</option>`).join("");
    $("video-filter").innerHTML += (state.bootstrap.videos || []).map((video) => `<option value="${esc(video)}">${esc(video)}</option>`).join("");
    await refreshOverview();
    const initial = state.examples.find((e) => e.id === state.id) || state.examples[0];
    if (initial) await loadExample(initial.id);
    else { $("loading-state").innerHTML = '<h2>No examples available</h2><p>Check the server’s dataset and saved prediction paths.</p>'; $("workspace").setAttribute("aria-busy","false"); }
  } catch(e) { error(e.message); $("loading-state").innerHTML = '<h2>Workspace could not load</h2><p>Check the review server, then reload this page.</p>'; $("workspace").setAttribute("aria-busy","false"); }
}
bootstrap();
