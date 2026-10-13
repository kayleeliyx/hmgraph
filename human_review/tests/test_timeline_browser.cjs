#!/usr/bin/env node
/* Full-source timeline smoke: external Playwright, real saved annotations/video,
 * and a temporary label database. No evaluator, annotation or training writes.
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { spawn } = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const root = path.resolve(__dirname, '..');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'actground-timeline-browser-'));
const artifacts = process.env.TIMELINE_TEST_ARTIFACT_DIR || '/tmp/actground-timeline-browser-smoke';
fs.mkdirSync(artifacts, { recursive: true });
const config = JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf8'));
for (const key of ['evaluation_dir', 'evaluation_config', 'media_records', 'video_inventory', 'timeline_annotations', 'timeline_inventory', 'timeline_video_inventory']) {
  if (config[key]) config[key] = path.resolve(root, config[key]);
}
config.state_dir = path.join(temporary, 'state');
const configPath = path.join(temporary, 'config.json');
fs.writeFileSync(configPath, JSON.stringify(config));

let browser;
let server;
let page;
let output = '';
const errors = [];
const checks = [];
function pass(name) { checks.push(name); console.log('PASS', name); }

function unionSeconds(rows) {
  const intervals = rows.filter(row => row.valid_interval !== false && Number.isFinite(row.start_sec) && Number.isFinite(row.end_sec)).map(row => [row.start_sec, row.end_sec]).sort((a, b) => a[0] - b[0]);
  let total = 0, left = null, right = null;
  for (const [start, end] of intervals) {
    if (left === null) { left = start; right = end; }
    else if (start <= right) right = Math.max(right, end);
    else { total += right - left; left = start; right = end; }
  }
  return total + (left === null ? 0 : right - left);
}

function start() {
  server = spawn(process.env.REVIEW_TEST_PYTHON || 'python3', [path.join(root, 'review.py'), '--config', configPath, '--port', process.env.TIMELINE_TEST_PORT || '0'], { stdio: ['ignore', 'pipe', 'pipe'] });
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Timeline server startup timeout: ' + output)), 60000);
    function read(chunk) {
      output += chunk.toString();
      const match = output.match(/ActGround Review: (http:\/\/127\.0\.0\.1:\d+)/);
      if (match) { clearTimeout(timeout); resolve(match[1]); }
    }
    server.stdout.on('data', read);
    server.stderr.on('data', read);
    server.once('error', error => { clearTimeout(timeout); reject(error); });
    server.once('exit', code => { clearTimeout(timeout); reject(new Error(`Timeline server exited ${code}: ${output}`)); });
  });
}

async function main() {
  const url = await start();
  browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
  page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  const videosResponse = await context.request.get(url + '/api/timeline/videos');
  assert.equal(videosResponse.status(), 200, await videosResponse.text());
  const videos = (await videosResponse.json()).videos;
  assert.equal(videos.length, 156);
  assert.equal(videos.reduce((sum, video) => sum + video.action_count, 0), 59454);
  assert.equal(videos.reduce((sum, video) => sum + video.goal_count, 0), 1490);
  assert.equal(videos.reduce((sum, video) => sum + video.accepted_why_count, 0), 3124);
  assert.equal(videos.reduce((sum, video) => sum + video.query_count, 0), 2209);
  assert.ok(videos.every(video => video.source_complete));
  pass('All 156 videos and 59,454 source actions remain separate from 2,209 queries');
  const reviewedVideo = videos.find(video => video.query_count > 0);
  const unreviewedVideo = videos.find(video => video.query_count === 0 && video.action_count > 0);
  const selectedResponse = await context.request.get(url + '/api/timeline/video?id=' + encodeURIComponent(reviewedVideo.id) + '&reviewer=Timeline%20smoke%20reviewer');
  assert.equal(selectedResponse.status(), 200, await selectedResponse.text());
  const selected = await selectedResponse.json();
  assert.equal(selected.actions.length, reviewedVideo.action_count);
  assert.equal(selected.queries.length, reviewedVideo.query_count);
  const sourceById = new Map(selected.actions.map(action => [action.id, action]));
  for (const why of selected.why) {
    const action = sourceById.get(why.action_id);
    assert.ok(action, 'WHY references an unknown source action');
    assert.equal(why.start_sec, action.start_sec);
    assert.equal(why.end_sec, action.end_sec);
    assert.equal(why.interval_meaning, 'target_action_interval; not purpose duration');
  }
  assert.ok(selected.goals.every(goal => goal.scope === 'as_annotated'));
  assert.ok(selected.queries.every(query => 'time_sec' in query && !('start_sec' in query) && !('end_sec' in query)));
  assert.ok(selected.human_review.points.every(point => 'time_sec' in point && !('start_sec' in point) && !('end_sec' in point)));
  assert.ok(Math.abs(selected.statistics.action_coverage.union_duration_sec - unionSeconds(selected.actions)) < 1e-6);
  assert.ok(Math.abs(selected.statistics.goal_coverage.union_duration_sec - unionSeconds(selected.goals)) < 1e-6);
  assert.ok(selected.statistics.action_coverage.union_duration_sec <= selected.duration_sec + 1e-6);
  const knownVerbs = selected.actions.filter(action => action.verb_id !== null);
  assert.equal(selected.statistics.verb_frequency.reduce((sum, category) => sum + category.count, 0), knownVerbs.length);
  pass('WHY alignment, heterogeneous GOAL scope, query points and union coverage are preserved');
  const emptyResponse = await context.request.get(url + '/api/timeline/video?id=' + encodeURIComponent(unreviewedVideo.id) + '&reviewer=Timeline%20smoke%20reviewer');
  assert.equal(emptyResponse.status(), 200, await emptyResponse.text());
  const empty = await emptyResponse.json();
  assert.equal(empty.queries.length, 0);
  assert.equal(empty.human_review.eligible_count, 0);
  assert.equal(empty.human_review.points.length, 0);
  assert.ok(!('accuracy' in empty.human_review));
  pass('Videos without predictions have no review points or artificial zero accuracy');
  await page.goto(url + '/timeline', { waitUntil: 'domcontentloaded' });
  await page.locator('#timeline-video-list button[data-video-id]').first().waitFor();
  await page.waitForFunction(id => document.querySelector('#timeline-video-title').textContent.trim() === id && document.querySelector('#timeline-main').getAttribute('aria-busy') === 'false', reviewedVideo.id);
  assert.equal(await page.locator('#timeline-video-list button[data-video-id]').count(), 156);
  const displayedCounts = await page.locator('#timeline-counts strong').allTextContents();
  assert.deepEqual(displayedCounts.slice(0, 4).map(value => Number(value.replaceAll(',', ''))), [selected.actions.length, selected.goals.length, selected.why.filter(item => item.accepted).length, selected.queries.length]);
  const partitions = await page.locator('#timeline-partition-filter option').evaluateAll(options => options.map(option => option.value));
  assert.equal(new Set(partitions).size, partitions.length, 'Duplicate partition options');
  pass('Timeline workspace renders source videos');

  await page.locator('#timeline-task-filter').selectOption('evaluated');
  assert.equal(await page.locator('#timeline-video-list button').count(), videos.filter(video => video.query_count > 0).length);
  await page.locator('#timeline-task-filter').selectOption('unevaluated');
  assert.equal(await page.locator('#timeline-video-list button').count(), videos.filter(video => video.query_count === 0).length);
  for (const task of ['explain', 'track']) {
    await page.locator('#timeline-task-filter').selectOption(task);
    assert.equal(await page.locator('#timeline-video-list button').count(), videos.filter(video => video.query_task_counts[task] > 0).length);
  }
  await page.locator('#timeline-task-filter').selectOption('');
  await page.locator('#timeline-video-filter').fill(reviewedVideo.id);
  assert.equal(await page.locator('#timeline-video-list button').count(), 1);
  await page.locator('#timeline-video-filter').fill('');
  pass('Source-video search and evaluated-query filters preserve the correct cohorts');

  assert.equal(selected.video.available, true, 'Reference video is unavailable');
  await page.waitForFunction(() => { const video = document.querySelector('#timeline-video'); return video.readyState >= 2 && video.videoWidth > 0; }, null, { timeout: 30000 });
  await page.locator('#timeline-video').evaluate(video => video.play());
  await page.waitForFunction(() => { const video = document.querySelector('#timeline-video'); return video.currentTime > .2 && video.getVideoPlaybackQuality().totalVideoFrames > 1; });
  const decoded = await page.locator('#timeline-video').evaluate(video => { video.pause(); return { frames: video.getVideoPlaybackQuality().totalVideoFrames, width: video.videoWidth, duration: video.duration }; });
  assert.ok(decoded.frames > 1 && decoded.width > 0);
  assert.ok(Math.abs(decoded.duration - selected.duration_sec) < 1);
  pass('Full-video playback decodes actual local video frames');

  const fullRange = await page.locator('#timeline-range-span').innerText();
  assert.equal(await page.locator('#timeline-pan').isDisabled(), true);
  await page.locator('#timeline-zoom-in').click();
  assert.notEqual(await page.locator('#timeline-range-span').innerText(), fullRange);
  assert.equal(await page.locator('#timeline-pan').isDisabled(), false);
  await page.locator('#timeline-pan').evaluate(input => { input.value = String(Number(input.max)); input.dispatchEvent(new Event('input', { bubbles: true })); });
  assert.notEqual(await page.locator('#timeline-range-start').innerText(), '0:00.00');
  await page.locator('#timeline-fit').click();
  assert.equal(await page.locator('#timeline-range-span').innerText(), fullRange);
  assert.equal(await page.locator('#timeline-pan').isDisabled(), true);
  pass('Zoom, pan and full-video fit change the visible time window');

  assert.equal(await page.locator('#timeline-svg .timeline-bar[data-kind="action"]').count(), selected.actions.length);
  assert.equal(await page.locator('#timeline-svg .timeline-bar[data-kind="goal"]').count(), selected.goals.length);
  assert.equal(await page.locator('#timeline-svg .timeline-bar[data-kind="why"]').count(), selected.why.filter(item => item.accepted).length);
  await page.locator('#timeline-why-status-select').selectOption('all');
  assert.equal(await page.locator('#timeline-svg .timeline-bar[data-kind="why"]').count(), selected.why.length);
  await page.locator('#timeline-why-status-select').selectOption('accepted');
  const sourceAction = selected.actions.find(action => action.end_sec - action.start_sec > 8 && action.start_sec > 0) || selected.actions[0];
  const sourceBar = page.locator(`#timeline-svg .timeline-bar[data-kind="action"][data-id="${sourceAction.id}"]`);
  // Source intervals can overlap; keyboard focus must still expose every event.
  await sourceBar.focus();
  await sourceBar.press('Enter');
  await page.waitForFunction(start => Math.abs(document.querySelector('#timeline-video').currentTime - start) < .1, sourceAction.start_sec);
  assert.ok((await page.locator('#timeline-detail').innerText()).includes(sourceAction.text));
  const clickableGoal = [...selected.goals].sort((a, b) => (b.end_sec - b.start_sec) - (a.end_sec - a.start_sec))[0];
  await page.locator(`#timeline-svg .timeline-bar[data-kind="goal"][data-id="${clickableGoal.id}"]`).click();
  await page.waitForFunction(start => Math.abs(document.querySelector('#timeline-video').currentTime - start) < .1, clickableGoal.start_sec);
  assert.ok((await page.locator('#timeline-detail').innerText()).includes(clickableGoal.text));
  for (const kind of ['action', 'goal']) {
    const rows = kind === 'action' ? selected.actions : selected.goals;
    const pair = rows.flatMap((first, index) => rows.slice(index + 1).filter(second => first.start_sec < second.end_sec && second.start_sec < first.end_sec).map(second => [first, second]))[0];
    if (pair) {
      const y = await Promise.all(pair.map(item => page.locator(`#timeline-svg .timeline-bar[data-kind="${kind}"][data-id="${item.id}"]`).getAttribute('y')));
      assert.notEqual(y[0], y[1], 'Overlapping source intervals occlude each other');
    }
  }
  pass('Complete source intervals and WHY statuses remain inspectable and seek the video');

  const categoryButton = page.locator('#timeline-statistics button[data-category-id]').first();
  const categoryId = await categoryButton.getAttribute('data-category-id');
  const matchingActions = selected.actions.filter(action => String(action.verb_id) === categoryId);
  assert.equal(Number(await categoryButton.getAttribute('data-value')), matchingActions.length);
  await categoryButton.click();
  assert.equal(await page.locator('#timeline-svg .timeline-bar[data-kind="action"]').count(), matchingActions.length);
  assert.equal(await page.locator('#timeline-active-filter').isVisible(), true);
  await page.locator('#timeline-clear-filter').click();
  assert.equal(await page.locator('#timeline-svg .timeline-bar[data-kind="action"]').count(), selected.actions.length);
  await page.locator('#timeline-measure-select').selectOption('duration');
  const durationButtons = await page.locator('#timeline-statistics button[data-category-id]').evaluateAll(buttons => buttons.map(button => ({ id: button.dataset.categoryId, value: Number(button.dataset.value) })));
  for (const button of durationButtons) assert.ok(Math.abs(button.value - unionSeconds(selected.actions.filter(action => String(action.verb_id) === button.id))) < 1e-6, 'Class duration double-counts overlapping intervals');
  assert.match(await page.locator('#timeline-statistics-description').innerText(), /merging overlapping intervals/);
  await page.locator('#timeline-measure-select').selectOption('frequency');
  pass('Category filters use annotation counts and time charts merge overlaps');

  const bootstrap = await (await context.request.get(url + '/api/bootstrap')).json();
  const model = bootstrap.models.find(value => value.id === '8b_base')?.id || bootstrap.models[0].id;
  await page.locator('#timeline-reviewer').fill('Timeline smoke reviewer');
  await Promise.all([page.waitForResponse(response => response.url().includes('/api/timeline/video?') && response.url().includes('reviewer=Timeline%20smoke%20reviewer')), page.locator('#timeline-reviewer').press('Tab')]);
  await page.waitForFunction(() => document.querySelector('#timeline-main').getAttribute('aria-busy') === 'false');
  await page.locator('#timeline-model-select').selectOption(model);
  await page.waitForFunction(() => document.querySelector('#timeline-main').getAttribute('aria-busy') === 'false');
  const query = selected.queries.find(value => value.task === 'explain' && value.models.includes(model));
  assert.ok(query, 'No evaluated Explain case for the selected model');
  const queryBar = page.locator(`#timeline-svg .timeline-bar[data-kind="query"][data-id="${query.id}"]`);
  await queryBar.focus();
  await queryBar.press('Enter');
  const reviewHref = await page.locator('#timeline-detail .review-link').first().getAttribute('href');
  const reviewURL = new URL(reviewHref, url);
  assert.equal(reviewURL.searchParams.get('example'), query.id);
  assert.equal(reviewURL.searchParams.get('model'), model);
  assert.equal(reviewURL.searchParams.get('field'), 'why');
  assert.equal(reviewURL.searchParams.get('evidence'), 'future');
  assert.ok(await page.evaluate(id => JSON.parse(localStorage.getItem('actground.review.timelineExposure.v1'))[id], selected.id));
  await page.locator('#timeline-main').evaluate(element => { element.scrollTop = 0; });
  await page.setViewportSize({ width: 1600, height: 1600 });
  await page.screenshot({ path: path.join(artifacts, 'timeline-desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 1500, height: 900 });
  await page.screenshot({ path: path.join(artifacts, 'timeline-laptop.png'), fullPage: true });
  await page.locator('.timeline-chart-card').scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(artifacts, 'timeline-laptop-chart.png'), fullPage: true });
  await page.setViewportSize({ width: 1600, height: 1000 });
  pass('Query points link to the correct review case, field and model with future evidence');

  await page.goto(reviewURL.href, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(id => document.querySelector('#example-id')?.textContent.trim() === id && document.querySelector('#workspace').getAttribute('aria-busy') === 'false', query.id);
  assert.equal(await page.locator('#model-select').inputValue(), model);
  assert.match(await page.locator('.field-tab[data-field="why"]').getAttribute('class'), /selected/);
  assert.equal(await page.locator('#reviewer').inputValue(), 'Timeline smoke reviewer');
  const savedResponse = await Promise.all([page.waitForResponse(response => response.url().endsWith('/api/labels') && response.request().method() === 'POST'), page.locator('button[data-review-field="why"][data-label="correct"]').click()]);
  assert.equal(savedResponse[0].status(), 200, await savedResponse[0].text());
  const example = await (await context.request.get(url + '/api/example?id=' + encodeURIComponent(query.id) + '&reviewer=Timeline%20smoke%20reviewer')).json();
  const decision = example.labels.find(label => label.field === 'why' && label.model === model);
  assert.equal(decision.label, 'correct');
  assert.equal(decision.evidence, 'future');
  assert.match(decision.answer_sha256, /^[a-f0-9]{64}$/);
  pass('Review deep links preserve selection and bind saved human labels to future evidence');

  const goalURL = new URL(reviewURL.href);
  goalURL.searchParams.set('field', 'goal');
  goalURL.searchParams.delete('evidence');
  await page.goto(goalURL.href, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(id => document.querySelector('#example-id')?.textContent.trim() === id && document.querySelector('#workspace').getAttribute('aria-busy') === 'false', query.id);
  assert.match(await page.locator('.field-tab[data-field="goal"]').getAttribute('class'), /selected/);
  assert.equal(await page.locator('#model-select').inputValue(), model);
  const goalResponse = await Promise.all([page.waitForResponse(response => response.url().endsWith('/api/labels') && response.request().method() === 'POST'), page.locator('button[data-review-field="goal"][data-label="partial"]').click()]);
  assert.equal(goalResponse[0].status(), 200, await goalResponse[0].text());
  assert.equal((await goalResponse[0].json()).saved.evidence, 'future');
  pass('GOAL deep links retain the field and remember timeline evidence without URL hints');

  await page.evaluate(() => {
    localStorage.removeItem('actground.review.timelineExposure.v1');
    localStorage.removeItem('actground.review.videoExposure.v1');
  });
  const nextURL = new URL(reviewURL.href);
  nextURL.searchParams.set('field', 'next_2');
  await page.goto(nextURL.href, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(id => document.querySelector('#example-id')?.textContent.trim() === id && document.querySelector('#workspace').getAttribute('aria-busy') === 'false', query.id);
  assert.match(await page.locator('.field-tab[data-field="next"]').getAttribute('class'), /selected/);
  await page.waitForFunction(() => document.activeElement?.dataset.reviewField === 'next_2');
  const nextResponse = await Promise.all([page.waitForResponse(response => response.url().endsWith('/api/labels') && response.request().method() === 'POST'), page.locator('button[data-review-field="next_2"][data-label="correct"]').click()]);
  assert.equal(nextResponse[0].status(), 200, await nextResponse[0].text());
  const nextDecision = (await nextResponse[0].json()).saved;
  assert.equal(nextDecision.field, 'next_2');
  assert.equal(nextDecision.evidence, 'future');
  assert.equal(nextDecision.verb_correct, true);
  assert.equal(nextDecision.noun_correct, true);
  pass('NEXT position deep links focus the requested row and retain explicit future evidence');

  await page.goto(url + '/timeline?video=' + encodeURIComponent(selected.id) + '&model=' + encodeURIComponent(model) + '&field=why', { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(id => document.querySelector('#timeline-video-title').textContent.trim() === id && document.querySelector('#timeline-main').getAttribute('aria-busy') === 'false', selected.id);
  const refreshed = await (await context.request.get(url + '/api/timeline/video?id=' + encodeURIComponent(selected.id) + '&reviewer=Timeline%20smoke%20reviewer&model=' + encodeURIComponent(model))).json();
  const humanPoint = refreshed.human_review.points.find(point => point.id === query.id && point.model === model && point.field === 'why');
  assert.equal(humanPoint.label, 'correct');
  assert.equal(humanPoint.time_sec, query.time_sec);
  const humanBar = page.locator(`#timeline-svg .timeline-bar[data-kind="human"][data-id="${query.id}"]`);
  assert.match(await humanBar.getAttribute('aria-label'), /WHY · correct/);
  assert.match(await page.locator('#timeline-human-note').innerText(), /^1 actual human labels/);
  pass('Saved labels reload as timestamped human review points');

  const downloadEvent = page.waitForEvent('download');
  await page.locator('#timeline-svg-export').click();
  const download = await downloadEvent;
  assert.equal(download.suggestedFilename(), selected.id + '-timeline.svg');
  await download.saveAs(path.join(artifacts, 'timeline-export.svg'));
  assert.match(fs.readFileSync(path.join(artifacts, 'timeline-export.svg'), 'utf8'), /<svg.*xmlns="http:\/\/www.w3.org\/2000\/svg"/);
  pass('Current timeline exports a standalone SVG');

  await page.locator(`#timeline-video-list button[data-video-id="${unreviewedVideo.id}"]`).click();
  await page.waitForFunction(id => document.querySelector('#timeline-video-title').textContent.trim() === id && document.querySelector('#timeline-main').getAttribute('aria-busy') === 'false', unreviewedVideo.id);
  assert.equal(await page.locator('#timeline-svg .timeline-bar[data-kind="action"]').count(), empty.actions.length);
  assert.equal(await page.locator('#timeline-svg .timeline-bar[data-kind="query"]').count(), 0);
  assert.equal(await page.locator('#timeline-svg .timeline-bar[data-kind="human"]').count(), 0);
  assert.match(await page.locator('#timeline-svg').textContent(), /No evaluated predictions/);
  assert.match(await page.locator('#timeline-human-note').innerText(), /No human accuracy is reported/);
  await page.locator('#timeline-svg .timeline-bar[data-kind="action"]').first().focus();
  await page.locator('#timeline-svg .timeline-bar[data-kind="action"]').first().press('Enter');
  assert.equal(await page.locator('#timeline-detail .review-link').count(), 0);
  await page.waitForFunction(() => document.querySelector('#timeline-video').readyState >= 2 && document.querySelector('#timeline-video').videoWidth > 0, null, { timeout: 30000 });
  await page.locator('#timeline-main').evaluate(element => { element.scrollTop = 0; });
  await page.screenshot({ path: path.join(artifacts, 'timeline-source-only.png'), fullPage: true });
  pass('Source-only videos remain playable and explicitly show unmeasured review results');

  const invalidVideo = 'P04-20240414-065311';
  const invalidDetail = await (await context.request.get(url + '/api/timeline/video?id=' + invalidVideo)).json();
  const invalidGoal = invalidDetail.goals.find(goal => goal.valid_interval === false);
  assert.ok(invalidGoal.start_sec > invalidGoal.end_sec, 'Reversed source timing was silently repaired');
  assert.ok(invalidDetail.statistics.goal_coverage.excluded_invalid_intervals > 0);
  await page.locator(`#timeline-video-list button[data-video-id="${invalidVideo}"]`).click();
  await page.waitForFunction(id => document.querySelector('#timeline-video-title').textContent.trim() === id && document.querySelector('#timeline-main').getAttribute('aria-busy') === 'false', invalidVideo);
  const invalidBar = page.locator(`#timeline-svg .timeline-bar[data-kind="goal"][data-id="${invalidGoal.id}"]`);
  assert.equal(await invalidBar.getAttribute('data-source-valid'), 'false');
  assert.equal(Number(await invalidBar.getAttribute('width')), 4, 'Invalid source timing was drawn as a duration span');
  await invalidBar.focus();
  await invalidBar.press('Enter');
  assert.match(await page.locator('#timeline-detail').innerText(), /Invalid source timing.*original endpoints.*excluded from.*coverage/is);
  await page.locator('#timeline-main').evaluate(element => { element.scrollTop = 0; });
  await page.screenshot({ path: path.join(artifacts, 'timeline-invalid-source.png'), fullPage: true });
  pass('Reversed GOAL timings retain original endpoints and render only as flagged markers');
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(artifacts, 'timeline-browser.json'), JSON.stringify({ passed: true, checks, browser: browser.version(), isolated_state: true, page_errors: errors }, null, 2));
}

main().catch(async error => {
  if (page && !page.isClosed()) {
    try { await page.screenshot({ path: path.join(artifacts, 'timeline-failure.png'), fullPage: true }); } catch {}
  }
  fs.writeFileSync(path.join(artifacts, 'timeline-browser.json'), JSON.stringify({ passed: false, checks, error: error.stack, page_errors: errors, server_output: output }, null, 2));
  console.error(error.stack);
  process.exitCode = 1;
}).finally(async () => {
  if (browser) await browser.close();
  if (server) {
    const exited = new Promise(resolve => server.once('exit', resolve));
    server.kill('SIGTERM');
    await Promise.race([exited, new Promise(resolve => setTimeout(resolve, 3000))]);
  }
  fs.rmSync(temporary, { recursive: true, force: true });
});
