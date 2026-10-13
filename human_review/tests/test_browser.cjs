#!/usr/bin/env node
/* Real-browser smoke using actual saved evaluation data and an isolated label DB.
 * Requires Playwright externally; it is not an application dependency.
 * Example: PLAYWRIGHT_MODULE=/tmp/hdepic-human-review-browser/node_modules/playwright
 *          PLAYWRIGHT_BROWSERS_PATH=/tmp/hdepic-human-review-browser/browsers
 *          node human_review/tests/test_browser.cjs
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { spawn } = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const reviewRoot = path.resolve(__dirname, '..');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'actground-review-browser-'));
const artifactRoot = process.env.REVIEW_TEST_ARTIFACT_DIR || path.join(os.tmpdir(), 'actground-review-browser-smoke');
fs.mkdirSync(artifactRoot, { recursive: true });
const originalConfig = path.join(reviewRoot, 'config.json');
const config = JSON.parse(fs.readFileSync(originalConfig, 'utf8'));
for (const key of ['evaluation_dir', 'evaluation_config', 'media_records', 'video_inventory']) {
  if (config[key]) config[key] = path.resolve(reviewRoot, config[key]);
}
config.state_dir = path.join(temporary, 'state');
const temporaryConfig = path.join(temporary, 'config.json');
fs.writeFileSync(temporaryConfig, JSON.stringify(config));

let browser;
let server;
let serverOutput = '';
const pageErrors = [];
const checks = [];
function checked(name) { checks.push(name); console.log('PASS', name); }

async function startServer() {
  server = spawn(process.env.REVIEW_TEST_PYTHON || 'python3', [
    path.join(reviewRoot, 'review.py'), '--config', temporaryConfig,
    '--port', process.env.REVIEW_TEST_PORT || '0',
  ], { stdio: ['ignore', 'pipe', 'pipe'] });
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Review server startup timed out: ' + serverOutput)), 45000);
    function capture(chunk) {
      serverOutput += chunk.toString();
      const match = serverOutput.match(/ActGround Review: (http:\/\/127\.0\.0\.1:\d+)/);
      if (match) { clearTimeout(timer); resolve(match[1]); }
    }
    server.stdout.on('data', capture);
    server.stderr.on('data', capture);
    server.once('error', error => { clearTimeout(timer); reject(error); });
    server.once('exit', code => {
      clearTimeout(timer);
      reject(new Error(`Review server exited with ${code}: ${serverOutput}`));
    });
  });
}

async function main() {
  const url = await startServer();
  browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  const context = await browser.newContext({ viewport: { width: 1500, height: 900 }, acceptDownloads: true });
  const page = await context.newPage();
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.locator('#review-workspace').waitFor({ state: 'visible' });
  assert.equal(await page.locator('#global-error').isVisible(), false);
  assert.ok((await page.locator('#example-list button').count()) > 0);
  checked('Interface renders saved examples');

  await page.locator('#reviewer').fill('Browser smoke reviewer');
  await Promise.all([
    page.waitForResponse(response => response.url().includes('/api/example?') && response.url().includes('reviewer=Browser%20smoke%20reviewer')),
    page.locator('#reviewer').press('Tab'),
  ]);
  await page.waitForFunction(() => document.querySelector('#workspace').getAttribute('aria-busy') === 'false');
  await page.locator('#task-filter').selectOption('explain');
  const selectedIdentity = await page.locator('#example-list button').first().getAttribute('data-example-id');
  await page.locator('#example-list button').first().click();
  await page.waitForFunction(identity => document.querySelector('#example-id').textContent.trim() === identity, selectedIdentity);
  await page.locator('.field-tab[data-field="why"]').click();
  const identity = (await page.locator('#example-id').innerText()).trim();
  assert.ok(identity.includes(':explain:'));
  checked('Task filtering and example navigation');

  const detailResponse = await context.request.get(url + '/api/example?id=' + encodeURIComponent(identity) + '&reviewer=Browser%20smoke%20reviewer');
  assert.ok(detailResponse.ok());
  const detail = await detailResponse.json();
  assert.equal(detail.media.frames.length, 16);
  await page.waitForFunction(() => document.querySelector('#evidence-video').readyState >= 2);
  await page.screenshot({ path: path.join(artifactRoot, 'review-laptop.png'), fullPage: false });
  const labelBounds = await page.locator('button[data-review-field="why"][data-label="correct"]').boundingBox();
  assert.ok(labelBounds && labelBounds.y + labelBounds.height <= 900, 'First semantic label control is below the laptop viewport');
  checked('Compact laptop layout keeps the first decision visible');
  if (await page.locator('#frames-toggle').getAttribute('aria-expanded') === 'false') await page.locator('#frames-toggle').click();
  await page.waitForFunction(() => {
    const images = Array.from(document.querySelectorAll('#filmstrip img'));
    return images.length === 16 && images.every(image => image.complete && image.naturalWidth > 0);
  });
  assert.ok(detail.media.frames.every(frame => frame.time_sec <= detail.query_time_sec + 1e-6));
  checked('All 16 causal input frames load');

  assert.equal(detail.media.video.available, true, 'Selected example must have real reference video');
  await page.waitForFunction(() => document.querySelector('#evidence-video').readyState >= 2);
  const initialVideo = await page.locator('#evidence-video').evaluate(video => ({ width: video.videoWidth, height: video.videoHeight, ready: video.readyState }));
  assert.ok(initialVideo.width > 0 && initialVideo.height > 0, 'Video metadata loaded without decoded imagery');
  await page.locator('#play-button').click();
  await page.waitForFunction(() => {
    const video = document.querySelector('#evidence-video');
    return video.getVideoPlaybackQuality().totalVideoFrames > 0;
  });
  const played = await page.locator('#evidence-video').evaluate(video => ({ time: video.currentTime, frames: video.getVideoPlaybackQuality().totalVideoFrames, paused: video.paused }));
  assert.ok(played.frames > 0);
  assert.ok(played.time <= detail.query_time_sec + .03);
  await page.locator('#play-button').click();
  await page.locator('#video-scrubber').evaluate((input, end) => {
    input.value = String(end - .15);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }, detail.query_time_sec);
  await page.waitForFunction(end => Math.abs(document.querySelector('#evidence-video').currentTime - (end - .15)) < .1, detail.query_time_sec);
  await page.locator('#play-button').click();
  await page.waitForFunction(end => {
    const video = document.querySelector('#evidence-video');
    return video.paused && Math.abs(video.currentTime - end) <= .03;
  }, detail.query_time_sec);
  checked('Real video decodes and playback stops at the observed query boundary');

  await page.locator('#blind-toggle').check();
  const blindOptions = await page.locator('#model-select option').allTextContents();
  for (const prediction of detail.predictions) {
    assert.ok(!blindOptions.some(text => text.includes(prediction.model)), 'Blind menu exposes ' + prediction.model);
  }
  assert.equal(await page.locator('#auto-toggle').isChecked(), false);
  checked('Model names hidden and automated scores hidden');

  const observedMax = Number(await page.locator('#video-scrubber').getAttribute('max'));
  assert.ok(observedMax <= detail.query_time_sec + 1e-6);
  assert.equal(await page.locator('#future-toggle').getAttribute('aria-pressed'), 'false');
  await page.locator('#future-toggle').click();
  assert.equal(await page.locator('#future-toggle').getAttribute('aria-pressed'), 'true');
  assert.match(await page.locator('#evidence-badge').innerText(), /FUTURE/i);
  const futureMax = Number(await page.locator('#video-scrubber').getAttribute('max'));
  assert.ok(futureMax > observedMax);
  await page.locator('#future-toggle').click();
  assert.match(await page.locator('#evidence-badge').innerText(), /OBSERVED/i);
  checked('Observed and future evidence have distinct explicit bounds');

  const frameButton = page.locator('#filmstrip button').first();
  await frameButton.click();
  await page.locator('#modal[open]').waitFor();
  assert.ok((await page.locator('#modal img').count()) > 0);
  await page.locator('#modal-close').click();
  checked('Filmstrip opens enlarged evidence');

  // Frontend controls are deliberately exercised, rather than injecting saved
  // labels through the API. The fake reviewer is confined to the temporary DB.
  const selectedModel = await page.locator('#model-select').inputValue();
  const [saved] = await Promise.all([
    page.waitForResponse(response => response.url().endsWith('/api/labels') && response.request().method() === 'POST'),
    page.locator('#prediction-content button[data-review-field="why"][data-label="correct"]').click(),
  ]);
  assert.equal(saved.status(), 200, await saved.text());
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.locator('#review-workspace').waitFor({ state: 'visible' });
  const labelResponse = await context.request.get(url + '/api/example?id=' + encodeURIComponent(identity) + '&reviewer=Browser%20smoke%20reviewer');
  const labeled = await labelResponse.json();
  const label = labeled.labels.find(value => value.model === selectedModel && value.field === 'why');
  assert.ok(label, 'WHY label missing after reload');
  assert.equal(label.label, 'correct');
  assert.match(label.answer_sha256, /^[a-f0-9]{64}$/);
  assert.match(label.semantic_packet.prompt_sha256, /^[a-f0-9]{64}$/);
  assert.equal(label.evidence, 'future', 'Hiding future footage must not erase prior future exposure');
  checked('Human decision persists and is hash-bound after reload');

  await page.locator('.field-tab[data-field="goal"]').click();
  const [goalSave] = await Promise.all([
    page.waitForResponse(response => response.url().endsWith('/api/labels') && response.request().method() === 'POST' && response.request().postDataJSON().field === 'goal'),
    page.locator('button[data-review-field="goal"][data-label="partial"]').click(),
  ]);
  assert.equal(goalSave.status(), 200, await goalSave.text());
  await page.locator('.field-tab[data-field="next"]').click();
  assert.equal(await page.locator('button[data-review-field^="next_"][data-label="correct"]').count(), 3);
  for (const component of ['verb', 'noun']) {
    const value = component === 'verb' ? 'false' : 'true';
    const [response] = await Promise.all([
      page.waitForResponse(response => response.url().endsWith('/api/labels') && response.request().method() === 'POST' && response.request().postDataJSON().field === 'next_1'),
      page.locator(`button[data-review-field="next_1"][data-component="${component}"][data-value="${value}"]`).click(),
    ]);
    assert.equal(response.status(), 200, await response.text());
  }
  const [flagSave] = await Promise.all([
    page.waitForResponse(response => response.url().endsWith('/api/labels') && response.request().method() === 'POST' && response.request().postDataJSON().flags.mapper_issue === true),
    page.locator('input[data-review-field="next_1"][data-flag="mapper_issue"]').check(),
  ]);
  assert.equal(flagSave.status(), 200, await flagSave.text());
  const notes = 'Browser smoke only: verify separate verb/noun and mapper issue persistence.';
  const [notesSave] = await Promise.all([
    page.waitForResponse(response => response.url().endsWith('/api/labels') && response.request().method() === 'POST' && response.request().postDataJSON().notes === notes),
    page.locator('textarea[data-review-field="next_1"]').fill(notes),
  ]);
  assert.equal(notesSave.status(), 200, await notesSave.text());
  const decisionResponse = await context.request.get(url + '/api/example?id=' + encodeURIComponent(identity) + '&reviewer=Browser%20smoke%20reviewer');
  const decisions = (await decisionResponse.json()).labels;
  assert.equal(decisions.find(value => value.model === selectedModel && value.field === 'goal').label, 'partial');
  const nextLabel = decisions.find(value => value.model === selectedModel && value.field === 'next_1');
  assert.equal(nextLabel.label, 'incorrect');
  assert.equal(nextLabel.verb_correct, false);
  assert.equal(nextLabel.noun_correct, true);
  assert.equal(nextLabel.flags.mapper_issue, true);
  assert.equal(nextLabel.notes, notes);
  checked('GOAL and ordered NEXT support semantic and component review');

  await page.locator('#next-button').click();
  await page.waitForFunction(previous => document.querySelector('#example-id').textContent.trim() !== previous, identity);
  await page.locator('#previous-button').click();
  await page.waitForFunction(previous => document.querySelector('#example-id').textContent.trim() === previous, identity);
  await page.locator('.field-tab[data-field="next"]').click();
  assert.equal(await page.locator('textarea[data-review-field="next_1"]').inputValue(), notes);
  checked('Save-and-next navigation restores annotation notes');

  await page.locator('#export-button').click();
  await page.locator('#modal[open]').waitFor();
  assert.match(await page.locator('#modal-content').innerText(), /export|JSONL|CSV/i);
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.locator('#modal a[href*="format=jsonl"]').click(),
  ]);
  assert.equal(download.suggestedFilename(), 'human_labels.jsonl');
  await download.saveAs(path.join(artifactRoot, 'human-labels-smoke.jsonl'));
  const exported = await context.request.get(url + '/api/export?format=jsonl&reviewer=Browser%20smoke%20reviewer');
  assert.equal(exported.status(), 200);
  const exportedRows = (await exported.text()).trim().split('\n').map(line => JSON.parse(line));
  assert.ok(exportedRows.some(value => value.id === identity && value.model === selectedModel && value.field === 'why' && value.label === 'correct'));
  assert.ok(exportedRows.some(value => value.id === identity && value.field === 'next_1' && value.verb_correct === false && value.noun_correct === true));
  checked('Review export contains the saved human decision');
  await page.locator('#modal-close').click();

  for (const field of ['next_2', 'next_3']) {
    const [response] = await Promise.all([
      page.waitForResponse(response => response.url().endsWith('/api/labels') && response.request().method() === 'POST' && response.request().postDataJSON().field === field),
      page.locator(`button[data-review-field="${field}"][data-label="correct"]`).click(),
    ]);
    assert.equal(response.status(), 200, await response.text());
  }
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.locator('#review-workspace').waitFor({ state: 'visible' });
  const stateResponse = await context.request.get(url + '/api/progress?reviewer=Browser%20smoke%20reviewer');
  const progress = await stateResponse.json();
  assert.equal(progress.examples[identity].by_model[selectedModel].complete, true);
  assert.ok(progress.reviewed >= 1);
  await page.locator('#status-filter').selectOption('reviewed');
  assert.ok(await page.locator('#example-list button').count());
  assert.ok((await page.locator('#example-list button').allTextContents()).some(text => text.includes('move mug')));
  checked('Completed prediction review survives reload and reviewed filtering');
  await page.locator('#status-filter').selectOption('');

  await page.locator('#question-heading').click();
  await page.keyboard.press('k');
  await page.waitForFunction(previous => document.querySelector('#example-id').textContent.trim() !== previous, identity);
  await page.keyboard.press('j');
  await page.waitForFunction(previous => document.querySelector('#example-id').textContent.trim() === previous, identity);
  await page.locator('.field-tab[data-field="goal"]').click();
  await page.locator('#question-heading').click();
  const [keyboardLabel] = await Promise.all([
    page.waitForResponse(response => response.url().endsWith('/api/labels') && response.request().method() === 'POST' && response.request().postDataJSON().field === 'goal'),
    page.keyboard.press('p'),
  ]);
  assert.equal(keyboardLabel.status(), 200, await keyboardLabel.text());
  assert.equal((await keyboardLabel.json()).saved.label, 'partial');
  await page.waitForFunction(() => document.querySelector('#evidence-video').readyState >= 2);
  await page.keyboard.press('Space');
  await page.waitForFunction(() => !document.querySelector('#evidence-video').paused);
  await page.keyboard.press('Space');
  await page.waitForFunction(() => document.querySelector('#evidence-video').paused);
  checked('Keyboard navigation, semantic labels and playback work');

  await page.locator('#task-filter').selectOption('explain');
  const rapidTarget = await page.locator('#example-list button').nth(1).getAttribute('data-example-id');
  await page.route('**/api/examples?reviewer=Rapid%20reviewer', async route => {
    await new Promise(resolve => setTimeout(resolve, 250));
    await route.continue();
  });
  await page.locator('#reviewer').fill('Rapid reviewer');
  const rapidOverview = page.waitForResponse(response => response.url().includes('/api/examples?reviewer=Rapid%20reviewer'));
  await page.locator('#reviewer').press('Tab');
  await page.locator('#example-list button').nth(1).click();
  await rapidOverview;
  await page.waitForFunction(target => document.querySelector('#example-id').textContent.trim() === target && document.querySelector('#workspace').getAttribute('aria-busy') === 'false', rapidTarget);
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  assert.equal((await page.locator('#example-id').innerText()).trim(), rapidTarget);
  assert.equal(await page.locator('.field-tab[data-field="why"]').isEnabled(), true);
  checked('Fast reviewer changes cannot override a newer example selection');

  await page.locator('#reviewer').fill('Browser smoke reviewer');
  await Promise.all([
    page.waitForResponse(response => response.url().includes('/api/example?') && response.url().includes('reviewer=Browser%20smoke%20reviewer')),
    page.locator('#reviewer').press('Tab'),
  ]);
  await page.locator('#example-list button').first().click();
  await page.waitForFunction(target => document.querySelector('#example-id').textContent.trim() === target, identity);
  await page.locator('.field-tab[data-field="next"]').click();
  if (await page.locator('#frames-toggle').getAttribute('aria-expanded') === 'false') await page.locator('#frames-toggle').click();
  await page.waitForFunction(() => Array.from(document.querySelectorAll('#filmstrip img')).every(image => image.complete && image.naturalWidth > 0));
  await page.screenshot({ path: path.join(artifactRoot, 'review-desktop.png'), fullPage: true });
  assert.deepEqual(pageErrors, [], 'Browser JavaScript errors');
  const result = { passed: true, checks, browser: browser.version(), isolated_state: true, page_errors: pageErrors };
  fs.writeFileSync(path.join(artifactRoot, 'browser-smoke.json'), JSON.stringify(result, null, 2));
  console.log('Artifacts:', artifactRoot);
}

main().catch(error => {
  fs.writeFileSync(path.join(artifactRoot, 'browser-smoke.json'), JSON.stringify({ passed: false, checks, error: error.stack, page_errors: pageErrors, server_output: serverOutput }, null, 2));
  console.error(error.stack);
  process.exitCode = 1;
}).finally(async () => {
  if (browser) await browser.close();
  if (server) {
    const stopped = new Promise(resolve => server.once('exit', resolve));
    server.kill('SIGTERM');
    await Promise.race([stopped, new Promise(resolve => setTimeout(resolve, 3000))]);
  }
  fs.rmSync(temporary, { recursive: true, force: true });
});
