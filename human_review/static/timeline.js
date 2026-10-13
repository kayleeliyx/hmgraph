const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const workspaceKey = "actground.review.workspace.v1";
let preferences = {};
try { preferences = JSON.parse(localStorage.getItem(workspaceKey) || "{}"); } catch {}
const params = new URLSearchParams(window.location.search);
const state = {bootstrap:null, videos:[], detail:null, id:params.get("video") || "", reviewer:preferences.reviewer || "", model:params.get("model") || preferences.model || "", blind:!!preferences.blind, field:params.get("field") || "why", start:0, end:1, playhead:0, selected:null, category:null, whyStatus:"accepted", request:0, svgWidth:1200};
const colors = {goal:"#4b9291", action:"#667e9c", why:"#b99452", query:"#536687", correct:"#0b7c77", partial:"#c49a49", incorrect:"#bd5860", uncertain:"#9782b5", unreviewed:"#b6c2cf"};
const laneY = {goal:44, action:82, why:120, query:158, human:196};
const laneTitle = {goal:"GOAL", action:"Actions", why:"WHY candidates", query:"Queries", human:"Human review"};
function error(message) { $("timeline-error").textContent = message; $("timeline-error").hidden = !message; }
async function api(url) { const response = await fetch(url); const data = await response.json(); if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`); return data; }
function safeURL(value) { if (typeof value !== "string" || !value.trim()) return ""; try { const url = new URL(value,location.origin); return url.origin === location.origin && ["http:","https:"].includes(url.protocol) ? url.href : ""; } catch { return ""; } }
function number(value, fallback = 0) { const result = Number(value); return Number.isFinite(result) ? result : fallback; }
function time(value, precise = true) { const seconds = Math.max(0,number(value)); if (!precise) return `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2,"0")}`; return `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(2).padStart(5,"0")}`; }
function sourceTime(value) { if (typeof value !== "number" || !Number.isFinite(value)) return "unavailable"; return value < 0 ? `${value.toFixed(2)} s` : time(value); }
function shortDuration(value) { return value >= 60 ? `${(value/60).toFixed(1)} min` : `${value.toFixed(1)} s`; }
function humanModelName(id) { const models=state.bootstrap?.models || [], index=models.findIndex((model) => model.id===id); return state.blind ? `Model ${String.fromCharCode(65+Math.max(0,index))}` : models.find((model) => model.id===id)?.name || id; }
function accepted(item) { return typeof item.status === "string" && item.status.startsWith("accepted") || item.accepted === true; }
function duration() { return Math.max(1,number(state.detail?.duration_sec, Math.max(...(state.detail?.actions || []).map((a) => number(a.end_sec)),1))); }
function persistPreferences() { try { const existing = JSON.parse(localStorage.getItem(workspaceKey) || "{}"); localStorage.setItem(workspaceKey,JSON.stringify({...existing, reviewer:state.reviewer, model:state.model || existing.model})); } catch {} }
function markFullExposure() { if (!state.id) return; try { const exposures = JSON.parse(localStorage.getItem("actground.review.timelineExposure.v1") || "{}"); exposures[state.id] = true; localStorage.setItem("actground.review.timelineExposure.v1",JSON.stringify(exposures)); } catch {} }
function markVideoSeen(seconds) { if (!state.id || !Number.isFinite(seconds)) return; try { const exposures = JSON.parse(localStorage.getItem("actground.review.videoExposure.v1") || "{}"); exposures[state.id] = Math.max(number(exposures[state.id]),seconds); localStorage.setItem("actground.review.videoExposure.v1",JSON.stringify(exposures)); } catch {} }
function renderVideoList() {
  const search = $("timeline-video-filter").value.trim().toLowerCase(), partition = $("timeline-partition-filter").value, coverage = $("timeline-task-filter").value;
  const videos = state.videos.filter((video) => (!search || `${video.id} ${video.participant}`.toLowerCase().includes(search)) && (!partition || video.partition === partition) && (!coverage || coverage === "evaluated" && video.query_count > 0 || coverage === "unevaluated" && !video.query_count || coverage === "explain" && (video.query_task_counts?.explain > 0 || video.query_tasks?.includes("explain")) || coverage === "track" && (video.query_task_counts?.track > 0 || video.query_tasks?.includes("track"))));
  $("timeline-video-count").textContent = videos.length.toLocaleString();
  $("timeline-library-note").textContent = `${state.videos.length.toLocaleString()} source videos · ${state.videos.reduce((sum,v) => sum+number(v.action_count),0).toLocaleString()} action annotations`;
  $("timeline-video-list").innerHTML = videos.length ? videos.map((video) => `<button class="timeline-video-item ${video.id === state.id ? "selected" : ""}" data-video-id="${esc(video.id)}" role="listitem" aria-current="${video.id === state.id}"><span class="video-title">${esc(video.id)}</span><span class="video-meta"><span>${esc(video.participant)} · ${esc(video.partition || "unspecified")}</span><span>${time(video.duration_sec,false)}</span></span><span class="video-counts">${number(video.action_count).toLocaleString()} actions · ${number(video.goal_count).toLocaleString()} activities · ${number(video.query_count).toLocaleString()} queries</span></button>`).join("") : '<p class="empty-state">No videos match these filters.</p>';
}
async function loadVideo(id, preserve = false) {
  const request = ++state.request, reviewer = state.reviewer, model = state.model;
  try {
    error(""); $("timeline-main").setAttribute("aria-busy","true");
    const detail = await api(`/api/timeline/video?id=${encodeURIComponent(id)}&reviewer=${encodeURIComponent(reviewer)}&model=${encodeURIComponent(model)}`);
    if (request !== state.request) return;
    const same = id === state.id && state.detail;
    state.id = id; state.detail = detail;
    if (!preserve || !same) { state.start = 0; state.end = duration(); state.playhead = 0; state.selected = null; state.category = null; }
    else { state.start = Math.min(state.start,duration()); state.end = Math.min(state.end,duration()); }
    markFullExposure();
    const url = new URL(location.href); url.searchParams.set("video",id); if (state.model) url.searchParams.set("model",state.model); else url.searchParams.delete("model"); history.replaceState(null,"",url);
    $("timeline-loading").hidden = true; $("timeline-workspace").hidden = false;
    renderVideoList(); renderHeader(); renderPlayer(); renderTimeline(); renderStatistics(); renderDetail();
    $("timeline-main").setAttribute("aria-busy","false");
    document.querySelector(".timeline-video-item.selected")?.scrollIntoView({block:"nearest",inline:"nearest"});
  } catch (e) { error(e.message); $("timeline-main").setAttribute("aria-busy","false"); }
}
function filteredHuman() { return (state.detail?.human_review?.points || []).filter((point) => (!state.model || point.model === state.model) && (!state.field || point.field === state.field)); }
function renderHeader() {
  const detail = state.detail;
  $("timeline-video-title").textContent = detail.id;
  $("timeline-participant").textContent = detail.participant;
  $("timeline-partition").textContent = (detail.partition || "Unspecified partition").replaceAll("_"," ").toUpperCase();
  $("timeline-duration").textContent = time(duration(),false);
  $("timeline-source-note").textContent = `${detail.source_complete ? "All released source annotations loaded." : "Only a subset of source annotations is loaded; this is not the complete annotation set."} ${detail.source_note || ""}`;
  $("timeline-source-note").classList.toggle("timeline-incomplete",!detail.source_complete);
  $("timeline-warnings").hidden = !detail.warnings?.length;
  $("timeline-warnings").textContent = (detail.warnings || []).join(" · ");
  const acceptedWhy = (detail.why || []).filter(accepted).length;
  const actualLabels = (detail.human_review?.points || []).filter((point) => point.label !== "unreviewed").length;
  const cards = [[detail.actions.length,"Source actions"],[detail.goals.length,"Source activity intervals"],[acceptedWhy,"Accepted WHY candidates"],[detail.queries.length,"Sampled evaluation queries"],[actualLabels,"Actual human labels"]];
  $("timeline-counts").innerHTML = cards.map(([count,label]) => `<div class="timeline-count-card"><strong>${count.toLocaleString()}</strong><span>${esc(label)}</span></div>`).join("");
}
function renderPlayer() {
  const video = $("timeline-video"), data = state.detail.video || {}, source = safeURL(data.url);
  if (video.dataset.source !== source) { video.pause(); video.dataset.source = source; if (source) { video.src = source; video.load(); } else { video.removeAttribute("src"); video.load(); } }
  $("timeline-no-video").hidden = !!(data.available && source);
  $("timeline-current-time").textContent = time(state.playhead);
}
function seek(seconds) {
  state.playhead = Math.min(duration(),Math.max(0,number(seconds)));
  const video = $("timeline-video");
  if (state.detail.video?.available && video.dataset.source) { try { video.currentTime = state.playhead; } catch {} }
  markVideoSeen(state.playhead); updatePlayhead();
}
function updatePlayhead() {
  $("timeline-current-time").textContent = time(state.playhead);
  const left = 116, right = state.svgWidth - 20, span = state.end-state.start;
  const x = left + (state.playhead-state.start)/span*(right-left);
  const line = $("timeline-playhead");
  if (line) { line.setAttribute("x1",String(x)); line.setAttribute("x2",String(x)); line.style.display = state.playhead < state.start || state.playhead > state.end ? "none" : ""; }
  const overviewLine = $("timeline-overview-playhead");
  if (overviewLine) { const ox = state.playhead/duration()*1000; overviewLine.setAttribute("x1",String(ox)); overviewLine.setAttribute("x2",String(ox)); }
}
function setView(start,end) {
  const full = duration(), minSpan = Math.min(1,full), span = Math.max(minSpan,Math.min(full,end-start));
  state.start = Math.max(0,Math.min(full-span,start)); state.end = state.start+span;
  renderTimeline();
}
function zoom(factor) {
  const span = Math.min(duration(),Math.max(1,(state.end-state.start)*factor));
  const center = state.playhead >= state.start && state.playhead <= state.end ? state.playhead : (state.start+state.end)/2;
  setView(center-span/2,center+span/2);
}
function categoryMatches(action) {
  if (!state.category) return true;
  if (state.category.type === "verb") return String(action.verb_id) === state.category.id;
  if (state.category.type === "noun") return String(action.noun_id) === state.category.id;
  return state.detail.goals.some((goal) => goalKey(goal) === state.category.id && number(goal.start_sec) < number(action.end_sec) && number(goal.end_sec) > number(action.start_sec));
}
function goalKey(goal) { return typeof goal.text === "string" && goal.text.trim() ? goal.text : "__EMPTY_ACTIVITY__"; }
function sourceWhy() {
  return state.detail.why.filter((item) => (state.whyStatus === "all" || accepted(item)) && (!state.category || categoryMatches(state.detail.actions.find((action) => action.id === item.action_id) || {start_sec:item.start_sec,end_sec:item.end_sec})));
}
function eventTitle(kind,item) {
  if (kind === "human") return `${item.field.toUpperCase()} · ${item.label}\n${humanModelName(item.model)}\nQuery ${time(item.time_sec)}`;
  if (kind === "query") return `${item.task.toUpperCase()} query · ${time(item.time_sec)}\n${item.question || item.id}`;
  return `${laneTitle[kind]} · ${sourceTime(item.start_sec)}–${sourceTime(item.end_sec)}\n${item.text || "[empty source label]"}${kind === "why" ? `\nSource status: ${item.status || "unspecified"}` : ""}${item.valid_interval === false ? "\nInvalid source interval; a marker identifies its original start. Original timing is preserved." : ""}`;
}
function renderTimeline() {
  if (!state.detail) return;
  const element = $("timeline-svg"), width = Math.max(600,element.parentElement.clientWidth-32), left = 116, right = width-20, span = Math.max(.001,state.end-state.start);
  state.svgWidth = width; element.setAttribute("viewBox",`0 0 ${width} 260`);
  const x = (seconds) => left + (seconds-state.start)/span*(right-left);
  const drawingBounds = (item) => { const start=Math.min(duration(),Math.max(0,number(item.start_sec))); return {start,end:Math.min(duration(),Math.max(start,number(item.end_sec,start)))}; };
  const visible = (item) => { const bounds=drawingBounds(item); return bounds.end>=state.start && bounds.start<=state.end; };
  let svg = `<rect width="${width}" height="260" fill="white"/><defs><clipPath id="timeline-lanes-clip"><rect x="${left}" y="24" width="${right-left}" height="207"/></clipPath></defs>`;
  for (let tick=0;tick<=6;tick++) {
    const seconds = state.start+span*tick/6, px=x(seconds);
    svg += `<line x1="${px}" y1="25" x2="${px}" y2="227" stroke="#e8edf3" stroke-width="1"/><text x="${px}" y="15" text-anchor="${tick===0 ? "start" : tick===6 ? "end" : "middle"}" fill="#758597" font-size="9">${time(seconds,span<90)}</text>`;
  }
  for (const [kind,y] of Object.entries(laneY)) svg += `<text x="4" y="${y+13}" fill="#53697f" font-size="10" font-weight="550">${laneTitle[kind]}</text><rect x="${left}" y="${y-1}" width="${right-left}" height="25" rx="3" fill="#f6f8fa"/>`;
  svg += '<g clip-path="url(#timeline-lanes-clip)">';
  const sources = {goal:state.detail.goals.filter((goal) => !state.category || state.category.type !== "goal" || goalKey(goal) === state.category.id),action:state.detail.actions.filter(categoryMatches),why:sourceWhy()};
  for (const [kind,items] of Object.entries(sources)) {
    const ends=[], packed=[];
    for (const item of items.filter(visible).sort((a,b) => number(a.start_sec)-number(b.start_sec))) {
      const bounds=drawingBounds(item); let band=ends.findIndex((end) => end<=bounds.start);
      if (band<0) band=ends.length; ends[band]=bounds.end; packed.push({item,bounds,band});
    }
    const bandHeight=20/Math.max(1,ends.length);
    for (const {item,bounds,band} of packed) {
      const start = Math.max(state.start,bounds.start), end = Math.min(state.end,bounds.end), px=x(start), barWidth=Math.max(item.valid_interval === false ? 4 : 1,x(end)-px), selected=state.selected?.kind===kind && state.selected.item.id===item.id, y=laneY[kind]+1+band*bandHeight, height=Math.max(2,bandHeight-1);
      svg += `<rect class="timeline-bar" data-kind="${kind}" data-id="${esc(item.id)}" data-source-valid="${item.valid_interval !== false}" x="${px}" y="${y}" width="${barWidth}" height="${height}" rx="${barWidth>3 ? 2 : 0}" fill="${kind === "why" && !accepted(item) ? "#c6bca9" : colors[kind]}" fill-opacity="${kind === "action" ? .8 : .86}" ${selected ? 'stroke="#153349" stroke-width="2"' : item.valid_interval === false ? 'stroke="#bd5860" stroke-width="1.8" stroke-dasharray="3 2"' : ""} role="button" tabindex="0" aria-label="${esc(eventTitle(kind,item))}"><title>${esc(eventTitle(kind,item))}</title></rect>`;
      if (barWidth > 75 && height>13) svg += `<text x="${px+5}" y="${y+13}" fill="white" font-size="9" pointer-events="none">${esc((item.text || "").slice(0,Math.floor((barWidth-10)/5.5)))}</text>`;
    }
  }
  const queries = state.detail.queries.filter((query) => number(query.time_sec)>=state.start && number(query.time_sec)<=state.end);
  for (const query of queries) svg += `<path class="timeline-bar" data-kind="query" data-id="${esc(query.id)}" d="M ${x(query.time_sec)} ${laneY.query+4} l 4 7 l -4 7 l -4 -7 Z" fill="${query.task === "explain" ? "#536687" : "#96a5b9"}" role="button" tabindex="0" aria-label="${esc(eventTitle("query",query))}"><title>${esc(eventTitle("query",query))}</title></path>`;
  const points = filteredHuman();
  const groups = new Map();
  for (const point of points.filter((point) => number(point.time_sec)>=state.start && number(point.time_sec)<=state.end)) {
    const key = `${point.time_sec}:${point.label}`;
    if (!groups.has(key)) groups.set(key,[]); groups.get(key).push(point);
  }
  for (const grouped of groups.values()) {
    const point = grouped[0], index = state.detail.human_review.points.indexOf(point);
    svg += `<circle class="timeline-bar" data-kind="human" data-id="${esc(point.id)}" data-point-index="${index}" cx="${x(point.time_sec)}" cy="${laneY.human+10}" r="${point.label === "unreviewed" ? 3 : 4}" fill="${point.label === "unreviewed" ? "white" : colors[point.label] || colors.uncertain}" stroke="${colors[point.label] || colors.uncertain}" stroke-width="1.4" role="button" tabindex="0" aria-label="${esc(eventTitle("human",point))}"><title>${esc(eventTitle("human",point))}${grouped.length>1 ? `\n${grouped.length} decisions at this timestamp` : ""}</title></circle>`;
  }
  svg += '</g><line id="timeline-playhead" y1="24" y2="227" stroke="#d46064" stroke-width="1.5" pointer-events="none"/>';
  if (!state.detail.queries.length) svg += `<text x="${left+9}" y="${laneY.query+15}" fill="#8b99a6" font-size="10">No evaluated queries in this video</text>`;
  if (!points.length) svg += `<text x="${left+9}" y="${laneY.human+15}" fill="#8b99a6" font-size="10">${state.detail.queries.length ? "No eligible decisions for this model / field" : "No evaluated predictions"}</text>`;
  svg += `<text x="${left}" y="246" fill="#8290a0" font-size="9">${sources.action.length.toLocaleString()} source actions${state.category ? " after category filter" : ""} · ${state.detail.queries.length.toLocaleString()} sampled queries</text>`;
  element.innerHTML=svg;
  $("timeline-range-start").textContent=time(state.start); $("timeline-range-end").textContent=time(state.end); $("timeline-range-span").textContent=`Visible range · ${shortDuration(span)}`;
  $("timeline-zoom-in").disabled=span<=1; $("timeline-zoom-out").disabled=span>=duration();
  $("timeline-active-filter").hidden=!state.category; $("timeline-filter-label").textContent=state.category ? `Source interval filter: ${state.category.type === "goal" ? "activity" : state.category.type} · ${state.category.name}` : "";
  const labeled=points.filter((point) => point.label!=="unreviewed").length;
  $("timeline-human-note").textContent=points.length ? `${labeled} actual human labels / ${points.length} eligible query–model–field decisions` : "No human accuracy is reported without evaluated decisions.";
  const counts=Object.fromEntries(["correct","partial","incorrect","uncertain","unreviewed"].map((label) => [label,points.filter((point) => point.label===label).length]));
  $("timeline-human-summary").innerHTML=points.length ? `<span class="timeline-human-summary-title">Human review · selected model / field</span>${Object.entries(counts).map(([label,count]) => `<span class="timeline-human-count"><i style="background:${colors[label]}"></i>${label[0].toUpperCase()+label.slice(1)} <strong>${count.toLocaleString()}</strong></span>`).join("")}` : '<span class="timeline-human-summary-title">No evaluated predictions for this model / field. No accuracy is reported.</span>';
  const listedActions=sources.action.filter(visible);
  $("timeline-action-list-count").textContent=`(${listedActions.length.toLocaleString()})`;
  $("timeline-action-list").innerHTML=listedActions.length ? listedActions.map((action) => `<button class="timeline-action-row" data-timeline-action="${esc(action.id)}"><span>${sourceTime(action.start_sec)}–${sourceTime(action.end_sec)}</span><span>${esc(action.text || "[missing action text]")}${action.valid_interval === false ? " · invalid source interval" : ""}</span></button>`).join("") : '<p class="empty-state">No source actions match this range and category.</p>';
  renderOverview(); updatePlayhead();
}
function renderOverview() {
  const full=duration(), start=state.start/full*1000, width=(state.end-state.start)/full*1000;
  let svg='<rect x="0" y="3" width="1000" height="27" rx="4" fill="#edf2f6"/>';
  for (const action of state.detail.actions) svg+=`<rect x="${number(action.start_sec)/full*1000}" y="13" width="${Math.max(.5,(number(action.end_sec)-number(action.start_sec))/full*1000)}" height="7" fill="#bdcbd7"/>`;
  svg+=`<rect x="${start}" y="3" width="${Math.max(1,width)}" height="27" rx="3" fill="#0b7c771c" stroke="#58a39a" stroke-width="1.3"/><line id="timeline-overview-playhead" y1="2" y2="32" stroke="#d46064" stroke-width="1.5"/>`;
  $("timeline-overview-svg").setAttribute("viewBox","0 0 1000 36"); $("timeline-overview-svg").innerHTML=svg;
  const span=state.end-state.start, pan=$("timeline-pan"); pan.min=String(span/2); pan.max=String(full-span/2); pan.value=String((state.start+state.end)/2); pan.disabled=span>=full;
}
function selectEvent(kind,id,pointIndex) {
  let item;
  if (kind==="human") item=state.detail.human_review.points[number(pointIndex,-1)];
  else item=({goal:state.detail.goals,action:state.detail.actions,why:state.detail.why,query:state.detail.queries}[kind] || []).find((entry) => entry.id===id);
  if (!item) return;
  state.selected={kind,item}; seek(kind==="query" || kind==="human" ? item.time_sec : item.start_sec); renderDetail(); renderTimeline();
}
function relatedQueries(action,interval) {
  return state.detail.queries.filter((query) => action && (query.action_id===action.id || query.id.endsWith(action.id)) || interval && (query.time_sec>=interval.start_sec && query.time_sec<interval.end_sec));
}
function reviewLink(query,field,model) {
  const available=query.models || [], selected=available.includes(model) ? model : available[0];
  if (!selected) return "";
  const search=new URLSearchParams({example:query.id, field:field || (query.task==="explain" ? "why" : "goal"), model:selected,evidence:"future"});
  return `<a class="review-link" href="/?${esc(search.toString())}">Open in human review · ${esc(query.task.toUpperCase())} ${time(query.time_sec)} · ${esc(humanModelName(selected))}</a>`;
}
function renderDetail() {
  const selected=state.selected;
  if (!selected) { $("timeline-detail").innerHTML='<div class="card-heading"><div><div class="eyebrow">SELECTED EVENT</div><h3>Inspect an annotation</h3></div></div><p class="empty-state">Select an event in the timeline to see its source, timing and related evaluation queries.</p>'; return; }
  const {kind,item}=selected;
  const query=kind==="query" ? item : kind==="human" ? state.detail.queries.find((q) => q.id===item.id) : null;
  let action=kind==="action" ? item : kind==="why" ? state.detail.actions.find((a) => a.id===item.action_id) : query ? state.detail.actions.find((a) => query.action_id===a.id || query.id.endsWith(a.id)) || state.detail.actions.find((a) => a.start_sec<=query.time_sec && a.end_sec>=query.time_sec) : null;
  const interval=action || item, at=kind==="human" || kind==="query" ? item.time_sec : item.start_sec;
  const goals=kind==="goal" ? [item] : state.detail.goals.filter((goal) => goal.start_sec<=at && at<goal.end_sec);
  const why=kind==="why" ? [item] : action ? state.detail.why.filter((entry) => entry.action_id===action.id) : [];
  let html=`<div class="card-heading"><div><div class="eyebrow">${esc(kind==="human" ? "HUMAN LABEL" : laneTitle[kind].toUpperCase())}</div><h3>${kind==="human" ? `${esc(item.field.toUpperCase())} · ${esc(item.label)}` : kind==="query" ? `${esc(item.task.toUpperCase())} query` : "Source annotation"}</h3></div></div><div class="timeline-detail-body"><div class="timeline-detail-text">${esc(item.text || item.question || action?.text || item.id)}</div><div class="timeline-detail-meta"><span>${kind==="query" || kind==="human" ? time(item.time_sec) : `${sourceTime(item.start_sec)}–${sourceTime(item.end_sec)}`}</span>${action ? `<span>Verb: ${esc(action.verb || (action.verb_id ?? "unlabeled"))}</span><span>Noun: ${esc(action.noun || (action.noun_id ?? "unlabeled"))}</span>` : ""}${kind==="why" ? `<span>${esc(item.status || "unspecified")}</span>` : ""}</div>`;
  if (kind==="human") html+=`<div class="timeline-detail-section"><h4>HUMAN ASSESSMENT</h4><p>${esc(humanModelName(item.model))} · ${esc(item.label === "unreviewed" ? "No human label saved" : item.label)}${item.flags?.reference_issue ? " · reference issue flagged" : ""}</p></div>`;
  if (item.valid_interval === false || kind === "goal" && !item.text) html+=`<div class="timeline-detail-section"><h4>SOURCE ISSUE</h4><p>${item.valid_interval === false ? "Invalid source timing. Original endpoints are preserved, shown with a marker at the start, and excluded from time coverage. The source row remains in annotation counts." : "The released activity label is empty. The source interval remains in annotation counts."}</p></div>`;
  if (goals.length) html+=`<div class="timeline-detail-section"><h4>ACTIVE ACTIVITY AT THIS TIME</h4><ul>${goals.map((goal) => `<li>${esc(goal.text)}</li>`).join("")}</ul></div>`;
  if (why.length) html+=`<div class="timeline-detail-section"><h4>ASSOCIATED WHY CANDIDATE</h4><ul>${why.map((entry) => `<li>${esc(entry.text)} <span class="quiet">(${esc(entry.status)})</span></li>`).join("")}</ul></div>`;
  const narration=item.narration || action?.narration;
  if (narration) html+=`<details class="timeline-detail-section"><summary>Original narration</summary><p style="margin-top:7px">${esc(narration)}</p></details>`;
  const queries=query ? [query] : relatedQueries(action,interval);
  const links=queries.slice(0,12).map((q) => reviewLink(q,kind==="human" ? item.field : kind==="goal" ? "goal" : kind==="why" ? "why" : q.task==="explain" ? "why" : "current",kind==="human" ? item.model : state.model)).filter(Boolean);
  html+=`<div class="timeline-detail-section"><h4>RELATED EVALUATION QUERIES</h4>${links.length ? `<div class="timeline-review-links">${links.join("")}</div>${queries.length>12 ? `<p class="quiet" style="margin-top:7px">Showing 12 of ${queries.length} related queries.</p>` : ""}` : '<p>No saved evaluation query is linked to this interval.</p>'}</div></div>`;
  $("timeline-detail").innerHTML=html;
}
function distribution() {
  const kind=$("timeline-category-select").value, measure=$("timeline-measure-select").value, grouped=new Map();
  const items=kind==="goal" ? state.detail.goals : state.detail.actions;
  for (const item of items) {
    const id=kind==="goal" ? goalKey(item) : item[`${kind}_id`];
    if (id===null || id===undefined || id==="") continue;
    const key=String(id), name=kind==="goal" ? item.text || "[empty activity label]" : item[kind] || `Class ${id}`;
    if (!grouped.has(key)) grouped.set(key,{id:key,name,count:0,duration:0,intervals:[]});
    const value=grouped.get(key); value.count++;
    const start=number(item.start_sec,-1), end=number(item.end_sec,-1);
    if (item.valid_interval !== false && start>=0 && end>=start && end<=duration()) value.intervals.push([start,end]);
  }
  for (const value of grouped.values()) {
    let left=null,right=null;
    for (const [start,end] of value.intervals.sort((a,b) => a[0]-b[0] || a[1]-b[1])) {
      if (left===null) { left=start; right=end; }
      else if (start<=right) right=Math.max(right,end);
      else { value.duration+=right-left; left=start; right=end; }
    }
    if (left!==null) value.duration+=right-left;
  }
  return {kind,measure,values:[...grouped.values()].sort((a,b) => b[measure=== "frequency" ? "count" : "duration"]-a[measure=== "frequency" ? "count" : "duration"] || a.name.localeCompare(b.name)).slice(0,12)};
}
function renderStatistics() {
  if (!state.detail) return;
  const {kind,measure,values}=distribution(), metric=measure==="frequency" ? "count" : "duration", maximum=Math.max(...values.map((value) => value[metric]),1);
  $("timeline-statistics-description").textContent=`Top ${Math.min(12,values.length)} ${kind==="goal" ? "activities" : `${kind} classes`} by ${measure==="frequency" ? "source annotation frequency" : "covered time (seconds), merging overlapping intervals"} · ${state.detail.source_complete ? "all released annotations for this video" : "loaded annotation subset"}`;
  $("timeline-statistics").innerHTML=values.length ? values.map((value) => `<button class="timeline-distribution-row ${state.category?.type===kind && state.category.id===value.id ? "selected" : ""}" data-category-type="${kind}" data-category-id="${esc(value.id)}" data-category-name="${esc(value.name)}" data-value="${value[metric]}" aria-pressed="${state.category?.type===kind && state.category.id===value.id}"><div class="timeline-distribution-heading"><span>${esc(value.name)}</span><span>${measure==="frequency" ? `${value.count.toLocaleString()} annotations` : `${value.duration.toFixed(1)} s covered`}</span></div><div class="timeline-distribution-track"><div class="timeline-distribution-fill" style="width:${value[metric]/maximum*100}%"></div></div></button>`).join("") : '<p class="empty-state">No class annotations are available for this distribution.</p>';
}

$("timeline-video-list").addEventListener("click",(event) => { const button=event.target.closest("[data-video-id]"); if (button) loadVideo(button.dataset.videoId); });
for (const id of ["timeline-video-filter","timeline-partition-filter","timeline-task-filter"]) $(id).addEventListener(id.endsWith("video-filter") ? "input" : "change",renderVideoList);
$("timeline-reviewer").value=state.reviewer;
$("timeline-reviewer").addEventListener("change",() => { state.reviewer=$("timeline-reviewer").value.trim(); persistPreferences(); if (state.id) loadVideo(state.id,true); });
$("timeline-model-select").addEventListener("change",() => { state.model=$("timeline-model-select").value; persistPreferences(); if (state.id) loadVideo(state.id,true); });
$("timeline-field-select").value=state.field;
$("timeline-field-select").addEventListener("change",() => { state.field=$("timeline-field-select").value; renderTimeline(); });
$("timeline-why-status-select").addEventListener("change",() => { state.whyStatus=$("timeline-why-status-select").value; renderTimeline(); });
$("timeline-zoom-in").addEventListener("click",() => zoom(.5)); $("timeline-zoom-out").addEventListener("click",() => zoom(2)); $("timeline-fit").addEventListener("click",() => setView(0,duration()));
$("timeline-pan").addEventListener("input",() => { const center=number($("timeline-pan").value), span=state.end-state.start; setView(center-span/2,center+span/2); });
$("timeline-svg").addEventListener("click",(event) => { const bar=event.target.closest(".timeline-bar"); if (bar) selectEvent(bar.dataset.kind,bar.dataset.id,bar.dataset.pointIndex); else { const bounds=$("timeline-svg").getBoundingClientRect(), px=(event.clientX-bounds.left)/bounds.width*state.svgWidth; seek(state.start+(px-116)/(state.svgWidth-136)*(state.end-state.start)); } });
$("timeline-svg").addEventListener("keydown",(event) => { const bar=event.target.closest(".timeline-bar"); if (bar && ["Enter"," "].includes(event.key)) { event.preventDefault(); selectEvent(bar.dataset.kind,bar.dataset.id,bar.dataset.pointIndex); } });
$("timeline-svg").addEventListener("pointermove",(event) => {
  const bar=event.target.closest(".timeline-bar"), tooltip=$("timeline-tooltip"); if (!bar) { tooltip.hidden=true; return; }
  tooltip.innerHTML=`<strong>${esc(laneTitle[bar.dataset.kind])}</strong>${esc(bar.getAttribute("aria-label"))}`;
  tooltip.hidden=false; tooltip.style.left=`${Math.min(innerWidth-tooltip.offsetWidth-12,event.clientX+12)}px`; tooltip.style.top=`${Math.max(12,Math.min(innerHeight-tooltip.offsetHeight-12,event.clientY+15))}px`;
});
$("timeline-svg").addEventListener("pointerleave",() => { $("timeline-tooltip").hidden=true; });
$("timeline-overview-svg").addEventListener("click",(event) => { const bounds=event.currentTarget.getBoundingClientRect(), center=(event.clientX-bounds.left)/bounds.width*duration(), span=state.end-state.start; setView(center-span/2,center+span/2); });
$("timeline-video").addEventListener("timeupdate",() => { state.playhead=$("timeline-video").currentTime; markVideoSeen(state.playhead); updatePlayhead(); });
$("timeline-video").addEventListener("seeking",() => { state.playhead=$("timeline-video").currentTime; markVideoSeen(state.playhead); updatePlayhead(); });
$("timeline-video").addEventListener("error",() => { $("timeline-no-video").hidden=false; });
$("timeline-statistics").addEventListener("click",(event) => { const button=event.target.closest("[data-category-id]"); if (!button) return; const category={type:button.dataset.categoryType,id:button.dataset.categoryId,name:button.dataset.categoryName}; state.category=state.category?.type===category.type && state.category.id===category.id ? null : category; renderTimeline(); renderStatistics(); });
for (const id of ["timeline-category-select","timeline-measure-select"]) $(id).addEventListener("change",renderStatistics);
$("timeline-clear-filter").addEventListener("click",() => { state.category=null; renderTimeline(); renderStatistics(); });
$("timeline-action-list").addEventListener("click",(event) => { const button=event.target.closest("[data-timeline-action]"); if (button) selectEvent("action",button.dataset.timelineAction); });
$("timeline-svg-export").addEventListener("click",() => { const clone=$("timeline-svg").cloneNode(true); clone.setAttribute("xmlns","http://www.w3.org/2000/svg"); clone.setAttribute("width",String(state.svgWidth)); clone.setAttribute("height","260"); const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)],{type:"image/svg+xml"})), link=document.createElement("a"); link.href=url; link.download=`${state.id}-timeline.svg`; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000); });
let resizeTimer; window.addEventListener("resize",() => { clearTimeout(resizeTimer); resizeTimer=setTimeout(() => { if (state.detail) renderTimeline(); },120); });
async function bootstrap() {
  try {
    const [boot,library]=await Promise.all([api("/api/bootstrap"),api("/api/timeline/videos")]); state.bootstrap=boot; state.videos=library.videos;
    $("timeline-partition-filter").innerHTML += [...new Set(state.videos.map((video) => video.partition).filter(Boolean))].sort().map((partition) => `<option value="${esc(partition)}">${esc(partition.replaceAll("_"," "))}</option>`).join("");
    $("timeline-model-select").innerHTML='<option value="">All models</option>'+boot.models.map((model) => `<option value="${esc(model.id)}">${esc(humanModelName(model.id))}</option>`).join("");
    if (!boot.models.some((model) => model.id===state.model)) state.model=boot.models[0]?.id || "";
    $("timeline-model-select").value=state.model; renderVideoList();
    const initial=state.videos.find((video) => video.id===state.id) || state.videos.find((video) => video.query_count>0) || state.videos[0];
    if (initial) await loadVideo(initial.id); else { $("timeline-loading").innerHTML='<h2>No source videos available</h2><p>Check the timeline annotation asset.</p>'; $("timeline-main").setAttribute("aria-busy","false"); }
  } catch(e) { error(e.message); $("timeline-loading").innerHTML='<h2>Video timeline could not load</h2><p>Check the local review server, then reload.</p>'; $("timeline-main").setAttribute("aria-busy","false"); }
}
bootstrap();
