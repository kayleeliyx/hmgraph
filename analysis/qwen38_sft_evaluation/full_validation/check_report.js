// Run from the project after: source scripts/hdepic_data_storage.sh
// Reuses the existing browser installation without downloading anything.
// Reads the generated report; never creates or changes evaluation results.
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

const TOTAL = 228;
const MODELS = ['epoch1', 'epoch2'];
const FIELDS = ['why', 'goal', 'next'];

function dataPath(value, label) {
  assert(value, `${label} is unset; source scripts/hdepic_data_storage.sh first`);
  const resolved = fs.realpathSync(value);
  assert(resolved.startsWith('/data/'), `${label} resolves outside /data: ${resolved}`);
  return resolved;
}

function outputPath(name) {
  const output = path.join(dataPath(__dirname, 'Report directory'), name);
  if (fs.existsSync(output)) dataPath(output, name);
  return output;
}

function changeFor(example, model, field) {
  if (field === 'why' && !example.why_eligible) return 'excluded';
  const correct = name => field === 'next'
    ? example[name].next_correct_positions[0]
    : example[name][`${field}_judgment`].label === 'correct';
  const before = correct('base');
  const after = correct(model);
  return !before && after ? 'gain' : before && !after ? 'loss' : 'same';
}

async function main() {
  for (const key of ['TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_RUNTIME_DIR']) {
    dataPath(process.env[key], key);
  }
  dataPath(__dirname, 'Report directory');
  const disk = fs.statfsSync(__dirname);
  assert(disk.bavail * disk.bsize >= 256 * 1024 * 1024, 'Less than 256 MiB free on /data; browser check stopped');

  for (const name of ['report.html', 'analysis.json', 'examples.json']) {
    const input = path.join(__dirname, name);
    assert(fs.existsSync(input), `Generate the completed report first: missing ${input}`);
    dataPath(input, name);
  }
  const analysis = JSON.parse(fs.readFileSync(path.join(__dirname, 'analysis.json'), 'utf8'));
  const examples = JSON.parse(fs.readFileSync(path.join(__dirname, 'examples.json'), 'utf8'));
  assert.equal(analysis.scope.evaluated_examples, TOTAL);
  assert.equal(analysis.scope.full_generation_complete_for_requested_models, true);
  assert.equal(examples.length, TOTAL);
  assert.equal(new Set(examples.map(example => example.id)).size, TOTAL);
  assert.deepEqual([...examples.map(example => example.id)].sort(), [...analysis.scope.selected_query_ids].sort());
  const videos = [...new Set(examples.map(example => example.video_id))].sort();
  assert.equal(videos.length, 9);
  assert.equal(examples.filter(example => example.why_eligible).length, 226);

  // These existing /tmp assets are read only. Playwright's temporary profiles
  // inherit TMPDIR and the XDG directories checked above, all physically /data.
  const install = fs.realpathSync('/tmp/hdepic-human-review-browser');
  const { chromium } = require(path.join(install, 'node_modules/playwright'));
  const executablePath = fs.realpathSync(path.join(install, 'browsers/chromium-1248/chrome-linux64/chrome'));
  const browser = await chromium.launch({
    headless: true,
    executablePath,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-breakpad', '--disable-background-networking'],
  });
  const result = {
    status: 'passed',
    checked_utc: new Date().toISOString(),
    examples: TOTAL,
    videos: videos.length,
    video_filters: [],
    paired_filters: [],
    combined_filters: [],
    viewport_checks: [],
    page_errors: [],
    console_errors: [],
  };
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
    page.setDefaultTimeout(10000);
    await page.route('**/*', route => route.request().url().startsWith('file:') ? route.continue() : route.abort());
    page.on('pageerror', error => result.page_errors.push(error.message));
    page.on('console', message => {
      if (message.type() === 'error') result.console_errors.push(message.text());
    });
    await page.goto(pathToFileURL(path.join(__dirname, 'report.html')).href);
    assert.equal(await page.locator('#cases > .case').count(), TOTAL);

    const indices = predicate => examples.flatMap((example, index) => predicate(example) ? [index] : []);
    const allIndices = indices(() => true);
    async function shown(expected, label) {
      const actual = await page.locator('#cases > .case').evaluateAll(rows => rows.flatMap((row, index) => !row.hidden ? [index] : []));
      assert.deepEqual(actual, expected, `${label}: visible case identities`);
      assert.equal(await page.locator('#cases > .case:visible').count(), expected.length, `${label}: CSS visibility`);
      assert.equal(await page.locator('#shown').textContent(), `${expected.length} of ${TOTAL} examples`, `${label}: count output`);
    }
    async function overflow(width, state) {
      const measurement = await page.evaluate(() => ({
        viewport_width: innerWidth,
        document_width: document.documentElement.scrollWidth,
        body_width: document.body.scrollWidth,
      }));
      assert.equal(measurement.viewport_width, width);
      result.viewport_checks.push({ width, state, ...measurement });
      assert(measurement.document_width <= width && measurement.body_width <= width,
        `Document overflow at ${width}px (${state}): ${JSON.stringify(measurement)}`);
    }

    await shown(allIndices, 'Initial report');
    assert.equal(await page.locator('.case[open]').count(), 0);
    await overflow(1440, 'collapsed');
    await page.screenshot({ path: outputPath('report_preview.png'), fullPage: false });

    const rendered = await page.locator('#cases > .case').evaluateAll(rows => rows.map(row => ({
      video: row.dataset.video,
      title: row.querySelector(':scope > summary').textContent,
      changes: Object.fromEntries(['epoch1', 'epoch2'].flatMap(model => ['why', 'goal', 'next'].map(field =>
        [`${model}-${field}`, row.getAttribute(`data-${model}-${field}`)]))),
    })));
    for (const [index, example] of examples.entries()) {
      assert.equal(rendered[index].video, example.video_id);
      assert(rendered[index].title.includes(example.question), `Question missing from case ${index}`);
      for (const model of MODELS) {
        for (const field of FIELDS) {
          assert.equal(rendered[index].changes[`${model}-${field}`], changeFor(example, model, field),
            `Incorrect ${model}/${field} attribute for ${example.id}`);
        }
      }
    }

    assert.deepEqual(await page.locator('#video option').evaluateAll(options => options.map(option => option.value)), ['', ...videos]);
    for (const video of videos) {
      await page.locator('#video').selectOption(video);
      const expected = indices(example => example.video_id === video);
      await shown(expected, `Video ${video}`);
      result.video_filters.push({ video, matches: expected.length });
    }
    await page.locator('#video').selectOption('');
    await shown(allIndices, 'Video reset');

    const firstVideoIndices = indices(example => example.video_id === videos[0]);
    await page.locator('#search').fill(`  ${videos[0].toLowerCase()}  `);
    await shown(firstVideoIndices, 'Trimmed case-insensitive search');
    await page.locator('#video').selectOption(videos[0]);
    await shown(firstVideoIndices, 'Matching search and video');
    await page.locator('#video').selectOption(videos[1]);
    await shown([], 'Conflicting search and video');
    await page.locator('#video').selectOption('');
    await page.locator('#search').fill('no-example-matches-this-string-8cebbf');
    await shown([], 'Empty search result');
    await page.locator('#search').fill('');
    await shown(allIndices, 'Search reset');
    result.search = { case_insensitive_and_trimmed: true, video_intersection: true, empty_result: true, reset: true };

    for (const model of MODELS) {
      await page.locator('#model').selectOption(model);
      for (const field of FIELDS) {
        await page.locator('#field').selectOption(field);
        await page.locator('#change').selectOption('');
        await shown(allIndices, `${model}/${field}: all changes include excluded sources`);
        const paired = analysis.paired_transitions[model][field === 'next' ? 'next_1_action' : field];
        assert.equal(paired.denominator, field === 'why' ? 226 : TOTAL);
        assert.equal(paired.both_correct + paired.gained_correctness + paired.lost_correctness + paired.neither_correct, paired.denominator);
        for (const change of ['gain', 'loss', 'same']) {
          const expected = indices(example => changeFor(example, model, field) === change);
          const pairedCount = change === 'gain' ? paired.gained_correctness
            : change === 'loss' ? paired.lost_correctness : paired.both_correct + paired.neither_correct;
          assert.equal(expected.length, pairedCount, `analysis.json versus examples.json for ${model}/${field}/${change}`);
          await page.locator('#change').selectOption(change);
          await shown(expected, `${model}/${field}/${change}`);
          result.paired_filters.push({ model, field, change, matches: pairedCount });

          await page.locator('#video').selectOption(videos[0]);
          const intersection = expected.filter(index => examples[index].video_id === videos[0]);
          await shown(intersection, `${model}/${field}/${change} intersect video`);
          await page.locator('#search').fill(videos[0]);
          await shown(intersection, `${model}/${field}/${change} intersect video and search`);
          result.combined_filters.push({ model, field, change, video: videos[0], matches: intersection.length });
          await page.locator('#search').fill('');
          await page.locator('#video').selectOption('');
        }
      }
    }

    await page.locator('#change').selectOption('');
    await page.locator('#video').selectOption(videos[0]);
    await page.locator('.case:visible > summary').first().click();
    assert.equal(await page.locator('.case[open]').count(), 1, 'Manual expand');
    await page.locator('.case:visible > summary').first().click();
    assert.equal(await page.locator('.case[open]').count(), 0, 'Manual collapse');
    await page.locator('#expand').click();
    assert.equal(await page.locator('.case:not([hidden])[open]').count(), firstVideoIndices.length, 'Expand every visible case');
    assert.equal(await page.locator('.case[hidden][open]').count(), 0, 'Expand leaves hidden cases closed');
    await page.locator('#video').selectOption(videos[1]);
    await page.locator('#expand').click();
    assert.equal(await page.locator('.case[hidden][open]').count(), firstVideoIndices.length, 'Previously expanded cases remain open while hidden');
    await page.locator('#collapse').click();
    assert.equal(await page.locator('.case[open]').count(), 0, 'Collapse closes visible and hidden cases');
    await page.locator('#video').selectOption('');
    await shown(allIndices, 'All controls reset');
    await page.locator('#expand').click();
    assert.equal(await page.locator('.case[open]').count(), TOTAL, 'Expand all 228 cases');
    await overflow(1440, 'all expanded');
    await page.locator('#gallery').scrollIntoViewIfNeeded();
    await page.screenshot({ path: outputPath('report_gallery_preview.png'), fullPage: false });
    result.expand_collapse = { manual: true, visible_only: true, collapse_hidden: true, all_cases: true };

    await page.setViewportSize({ width: 390, height: 844 });
    await overflow(390, 'all expanded');
    await page.locator('#collapse').click();
    await overflow(390, 'collapsed');
    await page.locator('.case > summary').first().click();
    await page.locator('.case[open] .technical > summary').click();
    await overflow(390, 'case and technical details expanded');
    await page.locator('#gallery').scrollIntoViewIfNeeded();
    await page.screenshot({ path: outputPath('report_mobile_preview.png'), fullPage: false });
    assert.deepEqual(result.page_errors, [], 'Browser page errors');
    assert.deepEqual(result.console_errors, [], 'Browser console errors');
    fs.writeFileSync(outputPath('report_ui_check.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
