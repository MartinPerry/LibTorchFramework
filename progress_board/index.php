<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PyTorch Training Metrics</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: rgba(22, 30, 51, 0.88);
      --panel-strong: #18223a;
      --border: #2b3958;
      --text: #edf2ff;
      --muted: #9aa8c7;
      --accent: #6ee7b7;
      --accent-2: #60a5fa;
      --danger: #fb7185;
      --shadow: 0 18px 45px rgba(0, 0, 0, 0.26);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 15% -10%, rgba(59, 130, 246, 0.22), transparent 34rem),
        radial-gradient(circle at 92% 0%, rgba(16, 185, 129, 0.15), transparent 30rem),
        var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    button, select { font: inherit; }

    .shell {
      width: min(1440px, calc(100% - 32px));
      margin: 0 auto;
      padding: 34px 0 56px;
    }

    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 24px;
    }

    .eyebrow {
      margin: 0 0 8px;
      color: var(--accent);
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.15em;
      text-transform: uppercase;
    }

    h1 {
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.7rem);
      line-height: 0.98;
      letter-spacing: -0.05em;
    }

    .subtitle {
      max-width: 650px;
      margin: 13px 0 0;
      color: var(--muted);
      line-height: 1.6;
    }

    .button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      padding: 0 18px;
      border: 1px solid rgba(110, 231, 183, 0.5);
      border-radius: 12px;
      color: #06261d;
      background: var(--accent);
      font-weight: 800;
      cursor: pointer;
      box-shadow: 0 8px 24px rgba(110, 231, 183, 0.15);
    }

    .button:hover { filter: brightness(1.06); }
    .button.secondary {
      color: var(--text);
      background: transparent;
      border-color: var(--border);
      box-shadow: none;
    }

    #folderInput { display: none; }

    .source-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(360px, 0.85fr);
      gap: 14px;
      align-items: stretch;
    }

    .server-panel {
      padding: 18px 20px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: rgba(18, 27, 48, 0.72);
    }

    .source-heading {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 13px;
    }

    .source-heading h2 { margin: 0; font-size: 1rem; }
    .source-heading p { margin: 4px 0 0; color: var(--muted); font-size: 0.82rem; }
    .source-heading .button { min-height: 36px; padding: 0 12px; font-size: 0.78rem; }

    .run-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 9px;
      max-height: 230px;
      overflow-y: auto;
    }

    .run-card {
      min-width: 0;
      padding: 12px 13px;
      border: 1px solid var(--border);
      border-radius: 12px;
      color: var(--text);
      background: var(--panel-strong);
      text-align: left;
      cursor: pointer;
      transition: 140ms ease;
    }

    .run-card:hover { border-color: var(--accent-2); transform: translateY(-1px); }
    .run-card.active { border-color: var(--accent); background: rgba(16, 185, 129, 0.12); }
    .run-card:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .run-card strong { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .run-card-meta { display: block; margin-top: 5px; color: var(--muted); font-size: 0.74rem; }
    .run-empty { padding: 11px 2px; color: var(--muted); font-size: 0.84rem; }

    .drop-zone {
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 18px;
      padding: 18px 20px;
      border: 1px dashed #52658d;
      border-radius: 18px;
      background: rgba(18, 27, 48, 0.72);
      transition: 160ms ease;
    }

    .source-grid .drop-zone { height: 100%; }

    .drop-zone.dragging {
      border-color: var(--accent);
      background: rgba(16, 185, 129, 0.1);
      transform: translateY(-2px);
    }

    .drop-icon {
      display: grid;
      place-items: center;
      width: 46px;
      height: 46px;
      border-radius: 14px;
      color: var(--accent);
      background: rgba(110, 231, 183, 0.1);
      font-size: 1.35rem;
    }

    .drop-copy strong { display: block; margin-bottom: 4px; }
    .drop-copy span { color: var(--muted); font-size: 0.9rem; }

    .status {
      min-height: 22px;
      margin: 11px 3px 0;
      color: var(--muted);
      font-size: 0.88rem;
    }

    .status.error { color: var(--danger); }

    .dashboard { display: none; }
    .dashboard.visible { display: block; }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: flex-end;
      gap: 13px;
      margin: 25px 0 18px;
      padding: 16px;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: var(--panel);
    }

    .field { display: grid; gap: 7px; min-width: 190px; }
    .field label {
      color: var(--muted);
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    select {
      height: 40px;
      padding: 0 36px 0 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      color: var(--text);
      background: var(--panel-strong);
      outline: none;
    }

    select:focus { border-color: var(--accent-2); }
    .toolbar-summary { margin-left: auto; color: var(--muted); font-size: 0.88rem; }
    .source-label { color: var(--accent); font-size: 0.82rem; font-weight: 750; }

    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }

    .stat, .chart-card, .table-card {
      border: 1px solid var(--border);
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .stat { padding: 17px; border-radius: 16px; }
    .stat-label { color: var(--muted); font-size: 0.78rem; font-weight: 700; }
    .stat-value { margin-top: 6px; font-size: 1.58rem; font-weight: 850; letter-spacing: -0.03em; }
    .stat-change { min-height: 18px; margin-top: 4px; font-size: 0.78rem; }
    .better { color: var(--accent); }
    .worse { color: var(--danger); }
    .neutral { color: var(--muted); }

    .charts {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 480px), 1fr));
      gap: 14px;
    }

    .chart-card { min-width: 0; padding: 18px; border-radius: 18px; }
    .chart-heading { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
    .chart-heading h2 { margin: 0; font-size: 1rem; }
    .chart-heading p { margin: 4px 0 0; color: var(--muted); font-size: 0.78rem; }
    .chart-wrap { position: relative; height: 290px; }
    canvas { display: block; width: 100%; height: 100%; }

    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      min-height: 22px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.73rem;
    }

    .legend-item { display: inline-flex; align-items: center; gap: 6px; }
    .legend-line { width: 16px; height: 3px; border-radius: 99px; }

    .tooltip {
      position: fixed;
      z-index: 10;
      display: none;
      max-width: 300px;
      padding: 10px 12px;
      border: 1px solid #425273;
      border-radius: 10px;
      color: var(--text);
      background: #111a2d;
      box-shadow: var(--shadow);
      pointer-events: none;
      font-size: 0.78rem;
      line-height: 1.45;
    }

    .tooltip strong { display: block; margin-bottom: 5px; }
    .tooltip-row { display: flex; justify-content: space-between; gap: 16px; }

    .table-card { margin-top: 14px; padding: 18px; border-radius: 18px; overflow: hidden; }
    .table-card h2 { margin: 0 0 13px; font-size: 1rem; }
    .table-scroll { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th, td { padding: 10px 12px; border-bottom: 1px solid rgba(43, 57, 88, 0.75); text-align: right; white-space: nowrap; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
    th { color: var(--muted); font-size: 0.7rem; letter-spacing: 0.06em; text-transform: uppercase; }
    tbody tr:hover { background: rgba(96, 165, 250, 0.06); }
    .file-cell { max-width: 290px; overflow: hidden; text-overflow: ellipsis; }

    @media (max-width: 900px) {
      header { align-items: flex-start; flex-direction: column; }
      .source-grid { grid-template-columns: 1fr; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .toolbar-summary { width: 100%; margin-left: 0; }
    }

    @media (max-width: 560px) {
      .shell { width: min(100% - 20px, 1440px); padding-top: 24px; }
      .drop-zone { grid-template-columns: auto 1fr; }
      .drop-zone .button { grid-column: 1 / -1; }
      .stats { grid-template-columns: 1fr 1fr; gap: 9px; }
      .stat { padding: 14px; }
      .stat-value { font-size: 1.25rem; }
      .chart-card { padding: 14px 10px; }
      .chart-wrap { height: 250px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <p class="eyebrow">Experiment dashboard</p>
        <h1>Training progress</h1>
        <p class="subtitle">Inspect model convergence, image quality, and segmentation skill from checkpoint JSON files. Your files stay in this browser.</p>
      </div>
      <button class="button secondary" id="demoButton" type="button">Load demo data</button>
    </header>

    <section class="source-grid" aria-label="Metric data sources">
      <article class="server-panel">
        <div class="source-heading">
          <div><h2>Runs on this server</h2><p>Select a folder from data/&lt;run_id&gt;/</p></div>
          <button class="button secondary" id="refreshRunsButton" type="button">Refresh list</button>
        </div>
        <div class="run-list" id="runList"><div class="run-empty">Checking server run folders…</div></div>
      </article>

      <section class="drop-zone" id="dropZone" aria-label="Local JSON folder uploader">
        <div class="drop-icon" aria-hidden="true">↥</div>
        <div class="drop-copy">
          <strong>Load from this PC</strong>
          <span>Select or drag a folder / multiple .json files</span>
        </div>
        <label class="button" for="folderInput">Choose folder</label>
        <input id="folderInput" type="file" accept=".json,application/json" webkitdirectory directory multiple>
      </section>
    </section>
    <p class="status" id="status" role="status">Choose a server run or load a folder from this PC.</p>

    <section class="dashboard" id="dashboard">
      <div class="toolbar">
        <div><div class="source-label">Active source</div><div id="activeSourceLabel">—</div></div>
        <div class="field">
          <label for="modelSelect">Model</label>
          <select id="modelSelect"></select>
        </div>
        <div class="field">
          <label for="typeSelect">Run type</label>
          <select id="typeSelect"></select>
        </div>
        <div class="toolbar-summary" id="toolbarSummary"></div>
      </div>

      <div class="stats" id="stats"></div>

      <div class="charts">
        <article class="chart-card">
          <div class="chart-heading"><div><h2>Optimization</h2><p>Loss should generally fall as training converges.</p></div></div>
          <div class="chart-wrap"><canvas id="lossChart"></canvas></div>
          <div class="legend" id="lossLegend"></div>
        </article>
        <article class="chart-card">
          <div class="chart-heading"><div><h2>Pixel error</h2><p>RMSE and MAE — lower is better.</p></div></div>
          <div class="chart-wrap"><canvas id="errorChart"></canvas></div>
          <div class="legend" id="errorLegend"></div>
        </article>
        <article class="chart-card">
          <div class="chart-heading"><div><h2>Image quality</h2><p>PSNR in dB — higher is better.</p></div></div>
          <div class="chart-wrap"><canvas id="qualityChart"></canvas></div>
          <div class="legend" id="qualityLegend"></div>
        </article>
        <article class="chart-card">
          <div class="chart-heading"><div><h2>CSI across scales</h2><p>Event detection skill; pooled scores show spatial tolerance.</p></div></div>
          <div class="chart-wrap"><canvas id="csiChart"></canvas></div>
          <div class="legend" id="csiLegend"></div>
        </article>
        <article class="chart-card">
          <div class="chart-heading"><div><h2>Jaccard / IoU</h2><p>Positive and macro overlap reveal class performance.</p></div></div>
          <div class="chart-wrap"><canvas id="jaccardChart"></canvas></div>
          <div class="legend" id="jaccardLegend"></div>
        </article>
        <article class="chart-card">
          <div class="chart-heading"><div><h2>Classification</h2><p>Accuracy and MCR (misclassification rate).</p></div></div>
          <div class="chart-wrap"><canvas id="classChart"></canvas></div>
          <div class="legend" id="classLegend"></div>
        </article>
      </div>

      <article class="table-card">
        <h2>Recent checkpoints</h2>
        <div class="table-scroll">
          <table>
            <thead><tr id="metricsTableHead"></tr></thead>
            <tbody id="metricsTable"></tbody>
          </table>
        </div>
      </article>
    </section>
  </main>

  <div class="tooltip" id="tooltip"></div>

  <script>
    const COLORS = ["#6ee7b7", "#60a5fa", "#fbbf24", "#c084fc", "#fb7185", "#22d3ee"];
    const chartDefinitions = [
      { id: "loss", canvas: "lossChart", legend: "lossLegend", fields: [{ key: "loss", label: "Loss" }] },
      { id: "error", canvas: "errorChart", legend: "errorLegend", fields: [{ key: "rmse", label: "RMSE" }, { key: "mae", label: "MAE" }, { key: "mse", label: "MSE" }] },
      { id: "quality", canvas: "qualityChart", legend: "qualityLegend", fields: [{ key: "psnr", label: "PSNR (dB)" }] },
      { id: "csi", canvas: "csiChart", legend: "csiLegend", fixedRange: [0, 1], fields: [
        { key: "csi", label: "CSI" }, { key: "csi_mean_pool1", label: "Mean pool 1" },
        { key: "csi_mean_pool4", label: "Mean pool 4" }, { key: "csi_mean_pool16", label: "Mean pool 16" }
      ]},
      { id: "jaccard", canvas: "jaccardChart", legend: "jaccardLegend", fixedRange: [0, 1], fields: [
        { key: "jaccard_positive", label: "Positive" }, { key: "jaccard_macro", label: "Macro" }, { key: "jaccard_inverted", label: "Inverted" }
      ]},
      { id: "class", canvas: "classChart", legend: "classLegend", fixedRange: [0, 1], fields: [{ key: "acc", label: "Accuracy" }, { key: "mcr", label: "MCR" }] }
    ];

    const state = { records: [], filtered: [], charts: new Map(), activeSource: "" };
    function el(id) { return document.getElementById(id); }

    function parseFilename(fileName) {
      const name = fileName.split(/[\\/]/).pop();
      const match = name.match(/^(.*?)_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_([^_]+)_(\d+)\.json$/i);
      if (!match) {
        return { model: "Unclassified", type: "unknown", runId: null, savedAt: null, savedLabel: "Unknown", matched: false };
      }
      const [, model, year, month, day, hour, minute, type, runId] = match;
      const savedAt = new Date(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute));
      return {
        model, type, runId: Number(runId), savedAt,
        savedLabel: `${year}-${month}-${day} ${hour}:${minute}`,
        matched: true
      };
    }

    function normalizeRecord(data, fileName, path = fileName, runGroup = "Selected folder") {
      if (!data || Array.isArray(data) || typeof data !== "object") throw new Error("root must be a JSON object");
      const meta = parseFilename(fileName);
      const metrics = {};
      for (const [key, value] of Object.entries(data)) {
        const numeric = typeof value === "number" ? value : Number(value);
        if (Number.isFinite(numeric)) metrics[key] = numeric;
      }
      if (!Object.keys(metrics).length) throw new Error("no numeric metrics found");
      return { ...meta, metrics, fileName, path, runGroup };
    }

    async function readFiles(fileList) {
      const files = [...fileList].filter(function isJsonFile(file) {
        return file.name.toLowerCase().endsWith(".json");
      });
      if (!files.length) return setStatus("No .json files were found.", true);
      setStatus(`Reading ${files.length} JSON file${files.length === 1 ? "" : "s"}…`);
      const records = [];
      const failures = [];
      const localGroup = files[0]?.webkitRelativePath?.split(/[\\/]/)[0] || "PC selection";
      await Promise.all(files.map(async function readJsonFile(file) {
        try {
          const data = JSON.parse(await file.text());
          records.push(normalizeRecord(data, file.name, file.webkitRelativePath || file.name, localGroup));
        } catch (error) {
          failures.push(`${file.name}: ${error.message}`);
        }
      }));
      if (!records.length) return setStatus(`Could not load the files. ${failures[0] || ""}`, true);
      activateRecords(records, `This PC / ${localGroup}`);
      clearRunSelection();
      const unmatched = records.filter(function hasUnmatchedFilename(record) {
        return !record.matched;
      }).length;
      const notes = [
        `Loaded ${records.length} checkpoint${records.length === 1 ? "" : "s"}`,
        failures.length ? `${failures.length} invalid file${failures.length === 1 ? "" : "s"} skipped` : "",
        unmatched ? `${unmatched} filename${unmatched === 1 ? "" : "s"} did not match the expected pattern` : ""
      ].filter(Boolean);
      setStatus(notes.join(" · "), failures.length > 0);
    }

    function fileFromEntry(entry) {
      return new Promise(function getEntryFile(resolve, reject) {
        entry.file(resolve, reject);
      });
    }

    async function filesFromEntry(entry) {
      if (entry.isFile) return [await fileFromEntry(entry)];
      if (!entry.isDirectory) return [];
      const reader = entry.createReader();
      const children = [];
      while (true) {
        const batch = await new Promise(function readDirectoryEntries(resolve, reject) {
          reader.readEntries(resolve, reject);
        });
        if (!batch.length) break;
        children.push(...batch);
      }
      return (await Promise.all(children.map(filesFromEntry))).flat();
    }

    async function droppedFiles(dataTransfer) {
      const entries = [...dataTransfer.items]
        .map(function getDroppedEntry(item) { return item.webkitGetAsEntry?.(); })
        .filter(Boolean);
      return entries.length ? (await Promise.all(entries.map(filesFromEntry))).flat() : [...dataTransfer.files];
    }

    function setStatus(message, isError = false) {
      el("status").textContent = message;
      el("status").classList.toggle("error", isError);
    }

    function compareText(a, b) { return a.localeCompare(b); }
    function uniqueSorted(values) { return [...new Set(values)].sort(compareText); }

    function fillSelect(select, values) {
      select.innerHTML = "";
      for (const value of values) {
        const option = document.createElement("option");
        option.value = option.textContent = value;
        select.append(option);
      }
    }

    function populateFilters() {
      const previous = el("modelSelect").value;
      const models = uniqueSorted(state.records.map(function getRecordModel(record) { return record.model; }));
      fillSelect(el("modelSelect"), models);
      if (models.includes(previous)) el("modelSelect").value = previous;
      updateTypeFilter();
      el("dashboard").classList.add("visible");
      el("activeSourceLabel").textContent = state.activeSource;
    }

    function activateRecords(records, sourceLabel) {
      state.records = records;
      state.activeSource = sourceLabel;
      populateFilters();
      applyFilters();
    }

    function updateTypeFilter() {
      const model = el("modelSelect").value;
      const previous = el("typeSelect").value;
      const types = uniqueSorted(state.records
        .filter(function matchesModel(record) { return record.model === model; })
        .map(function getRecordType(record) { return record.type; }));
      fillSelect(el("typeSelect"), types);
      if (types.includes(previous)) el("typeSelect").value = previous;
    }

    function recordSort(a, b) {
      if (a.runId != null && b.runId != null && a.runId !== b.runId) return a.runId - b.runId;
      if (a.savedAt && b.savedAt) return a.savedAt - b.savedAt;
      return a.fileName.localeCompare(b.fileName);
    }

    function applyFilters() {
      const model = el("modelSelect").value;
      const type = el("typeSelect").value;
      const matching = state.records.filter(function matchesFilters(record) {
        return record.model === model && record.type === type;
      });
      const newestByRun = new Map();
      const unnumbered = [];
      matching.forEach(function collectLatestRun(record) {
        if (record.runId == null) return unnumbered.push(record);
        const previous = newestByRun.get(record.runId);
        if (!previous || (record.savedAt?.getTime() || 0) >= (previous.savedAt?.getTime() || 0)) newestByRun.set(record.runId, record);
      });
      state.filtered = [...newestByRun.values(), ...unnumbered].sort(recordSort);
      render();
    }

    function fmt(value, digits = 5) {
      if (!Number.isFinite(value)) return "—";
      if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
      return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
    }

    function renderStats() {
      const definitions = [
        { key: "loss", label: "Latest loss", lower: true },
        { key: "csi", label: "Latest CSI", lower: false },
        { key: "psnr", label: "Latest PSNR", lower: false, suffix: " dB" },
        { key: "rmse", label: "Latest RMSE", lower: true }
      ].filter(function statMetricAvailable(definition) {
        return state.filtered.some(function recordHasStatMetric(record) {
          return Number.isFinite(record.metrics[definition.key]);
        });
      });
      el("stats").innerHTML = definitions.map(function renderStatCard(def) {
        const observations = state.filtered.filter(function recordHasMetric(record) {
          return Number.isFinite(record.metrics[def.key]);
        });
        const current = observations.at(-1)?.metrics[def.key];
        const initial = observations[0]?.metrics[def.key];
        const delta = Number.isFinite(current) && Number.isFinite(initial) ? current - initial : NaN;
        const improved = Number.isFinite(delta) && delta !== 0 ? (def.lower ? delta < 0 : delta > 0) : null;
        const changeClass = improved === null ? "neutral" : improved ? "better" : "worse";
        const arrow = Number.isFinite(delta) && delta !== 0 ? (delta > 0 ? "↑" : "↓") : "";
        const change = Number.isFinite(delta) ? `${arrow} ${fmt(Math.abs(delta))} from first checkpoint` : "No comparison available";
        return `<article class="stat"><div class="stat-label">${def.label}</div><div class="stat-value">${fmt(current)}${Number.isFinite(current) ? def.suffix || "" : ""}</div><div class="stat-change ${changeClass}">${change}</div></article>`;
      }).join("");
    }

    function niceRange(values, fixedRange) {
      if (fixedRange) return fixedRange;
      let min = Math.min(...values), max = Math.max(...values);
      if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
      if (min === max) {
        const padding = Math.abs(min || 1) * 0.1;
        return [Math.max(0, min - padding), max + padding];
      }
      const padding = (max - min) * 0.12;
      return [Math.max(0, min - padding), max + padding];
    }

    function createChart(definition) {
      const canvas = el(definition.canvas);
      const ctx = canvas.getContext("2d");
      const availableFields = definition.fields.filter(function chartMetricAvailable(field) {
        return state.filtered.some(function recordHasChartMetric(record) {
          return Number.isFinite(record.metrics[field.key]);
        });
      });
      const series = availableFields.map(function buildSeries(field, index) {
        return {
          ...field,
          color: COLORS[index],
          values: state.filtered.map(function getSeriesValue(record) { return record.metrics[field.key]; })
        };
      });
      canvas.closest(".chart-card").hidden = series.length === 0;
      const allValues = series.flatMap(function getSeriesValues(item) { return item.values; }).filter(Number.isFinite);
      const range = niceRange(allValues, definition.fixedRange);
      const chart = { canvas, ctx, definition, series, range, points: [] };
      state.charts.set(definition.id, chart);
      renderLegend(definition.legend, series);
      drawChart(chart);
    }

    function renderLegend(id, series) {
      el(id).innerHTML = series.length
        ? series.map(function renderLegendItem(item) {
            return `<span class="legend-item"><i class="legend-line" style="background:${item.color}"></i>${item.label}</span>`;
          }).join("")
        : `<span>No matching metric in these files</span>`;
    }

    function drawChart(chart) {
      const { canvas, ctx, series, range } = chart;
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const width = rect.width, height = rect.height;
      const pad = { left: 55, right: 17, top: 15, bottom: 39 };
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      ctx.clearRect(0, 0, width, height);
      chart.points = [];

      if (!series.length || !state.filtered.length) {
        ctx.fillStyle = "#9aa8c7";
        ctx.font = "13px system-ui";
        ctx.textAlign = "center";
        ctx.fillText("Metric not available", width / 2, height / 2);
        return;
      }

      const [minY, maxY] = range;
      function yFor(value) {
        return pad.top + plotH - ((value - minY) / (maxY - minY || 1)) * plotH;
      }
      function xFor(index) {
        return pad.left + (state.filtered.length === 1 ? plotW / 2 : index / (state.filtered.length - 1) * plotW);
      }

      ctx.lineWidth = 1;
      ctx.font = "11px system-ui";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let tick = 0; tick <= 4; tick++) {
        const ratio = tick / 4;
        const y = pad.top + plotH * ratio;
        const value = maxY - (maxY - minY) * ratio;
        ctx.strokeStyle = "rgba(82, 101, 141, 0.28)";
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
        ctx.fillStyle = "#8e9dbc";
        ctx.fillText(fmt(value, 3), pad.left - 8, y);
      }

      const labelCount = Math.min(6, state.filtered.length);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let labelIndex = 0; labelIndex < labelCount; labelIndex++) {
        const index = labelCount === 1 ? 0 : Math.round(labelIndex * (state.filtered.length - 1) / (labelCount - 1));
        const record = state.filtered[index];
        ctx.fillStyle = "#8e9dbc";
        ctx.fillText(record.runId == null ? String(index + 1) : String(record.runId), xFor(index), height - pad.bottom + 11);
      }
      ctx.fillStyle = "#667695";
      ctx.fillText("run ID", pad.left + plotW / 2, height - 10);

      series.forEach(function drawSeries(item) {
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 2.25;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.beginPath();
        let started = false;
        item.values.forEach(function addLinePoint(value, index) {
          if (!Number.isFinite(value)) { started = false; return; }
          const x = xFor(index), y = yFor(value);
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        });
        ctx.stroke();

        item.values.forEach(function drawPoint(value, index) {
          if (!Number.isFinite(value)) return;
          const x = xFor(index), y = yFor(value);
          ctx.fillStyle = "#131d32";
          ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
          ctx.strokeStyle = item.color; ctx.lineWidth = 2; ctx.stroke();
          chart.points.push({ x, y, recordIndex: index });
        });
      });
    }

    function renderTable() {
      const preferred = ["loss", "csi", "psnr", "rmse", "mae", "mse", "jaccard_positive", "jaccard_macro", "jaccard_inverted", "acc", "mcr"];
      const keys = [...new Set(state.filtered.flatMap(function getMetricKeys(record) {
        return Object.keys(record.metrics);
      }))];
      keys.sort(function compareMetricKeys(a, b) {
        const ai = preferred.indexOf(a), bi = preferred.indexOf(b);
        if (ai !== -1 || bi !== -1) return (ai === -1 ? preferred.length : ai) - (bi === -1 ? preferred.length : bi);
        return a.localeCompare(b);
      });
      el("metricsTableHead").innerHTML = ["Run", "Saved", ...keys, "File"].map(function renderTableHeading(label) {
        return `<th>${escapeHtml(label.replaceAll("_", " "))}</th>`;
      }).join("");
      el("metricsTable").innerHTML = [...state.filtered].reverse().slice(0, 25).map(function renderTableRow(record) {
        const metricCells = keys.map(function renderMetricCell(key) {
          return `<td>${fmt(record.metrics[key])}</td>`;
        }).join("");
        return `<tr><td>${record.runId ?? "—"}</td><td>${record.savedLabel}</td>${metricCells}<td class="file-cell" title="${escapeHtml(record.path)}">${escapeHtml(record.fileName)}</td></tr>`;
      }).join("");
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>'"]/g, function replaceUnsafeCharacter(character) {
        return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character];
      });
    }

    function render() {
      const first = state.filtered[0], last = state.filtered.at(-1);
      const rangeText = first && last ? `runs ${first.runId ?? 1}–${last.runId ?? state.filtered.length}` : "";
      el("toolbarSummary").textContent = `${state.filtered.length} checkpoint${state.filtered.length === 1 ? "" : "s"}${rangeText ? ` · ${rangeText}` : ""}`;
      renderStats();
      chartDefinitions.forEach(createChart);
      renderTable();
    }

    function showTooltip(event, chart) {
      const rect = chart.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left, y = event.clientY - rect.top;
      let closest = null, distance = Infinity;
      for (const point of chart.points) {
        const candidate = Math.hypot(point.x - x, point.y - y);
        if (candidate < distance) { closest = point; distance = candidate; }
      }
      const tooltip = el("tooltip");
      if (!closest || distance > 18) { tooltip.style.display = "none"; return; }
      const record = state.filtered[closest.recordIndex];
      const rows = chart.series
        .filter(function tooltipMetricAvailable(item) { return Number.isFinite(item.values[closest.recordIndex]); })
        .map(function renderTooltipMetric(item) {
          return `<div class="tooltip-row"><span style="color:${item.color}">${item.label}</span><b>${fmt(item.values[closest.recordIndex], 6)}</b></div>`;
        }).join("");
      tooltip.innerHTML = `<strong>Run ${record.runId ?? closest.recordIndex + 1} · ${record.savedLabel}</strong>${rows}`;
      tooltip.style.display = "block";
      const tipWidth = tooltip.offsetWidth, tipHeight = tooltip.offsetHeight;
      tooltip.style.left = `${Math.min(window.innerWidth - tipWidth - 8, event.clientX + 14)}px`;
      tooltip.style.top = `${Math.max(8, Math.min(window.innerHeight - tipHeight - 8, event.clientY + 14))}px`;
    }

    function loadDemo() {
      const base = { loss: .48, csi: .055, psnr: 22.9, rmse: .073, mae: .037, mse: .0053, acc: .9962, mcr: .0038, jaccard_positive: .055, jaccard_macro: .526, jaccard_inverted: .996, csi_mean_pool1: .33, csi_mean_pool4: .36, csi_mean_pool16: .42 };
      const records = Array.from({ length: 18 }, function createDemoRecord(unused, index) {
        const run = index + 1, progress = index / 17, wobble = Math.sin(index * 1.7) * .008;
        const metrics = {
          loss: base.loss * Math.exp(-progress * 1.25) + wobble,
          rmse: base.rmse - progress * .019 + wobble / 5,
          mae: base.mae - progress * .012 + wobble / 7,
          mse: base.mse - progress * .0023 + wobble / 18,
          psnr: base.psnr + progress * 4.1 - wobble * 15,
          csi: base.csi + progress * .19 + wobble,
          csi_mean_pool1: base.csi_mean_pool1 + progress * .17 + wobble,
          csi_mean_pool4: base.csi_mean_pool4 + progress * .20 + wobble,
          csi_mean_pool16: base.csi_mean_pool16 + progress * .21 + wobble,
          jaccard_positive: base.jaccard_positive + progress * .19 + wobble,
          jaccard_macro: base.jaccard_macro + progress * .095 + wobble / 2,
          jaccard_inverted: base.jaccard_inverted + progress * .002,
          acc: base.acc + progress * .0022,
          mcr: base.mcr - progress * .0022
        };
        const fileName = `exPreCastModel_2026_08_27_20_${String(index + 1).padStart(2, "0")}_train_${run}.json`;
        return normalizeRecord(metrics, fileName, fileName, "Demo run");
      });
      activateRecords(records, "Demo data");
      clearRunSelection();
      setStatus("Demo data loaded · choose a server run or PC folder to replace it");
    }

    function clearRunSelection() {
      document.querySelectorAll(".run-card.active").forEach(function clearActiveRun(card) {
        card.classList.remove("active");
      });
    }

    function renderRunList(runs) {
      const runList = el("runList");
      if (!runs.length) {
        runList.innerHTML = `<div class="run-empty">No run folders containing JSON files were found.</div>`;
        return;
      }
      runList.innerHTML = runs.map(function renderRunCard(run) {
        const modified = run.modifiedAt ? ` · updated ${new Date(run.modifiedAt).toLocaleString()}` : "";
        return `<button class="run-card" type="button" data-run-id="${escapeHtml(run.id)}" ${run.fileCount ? "" : "disabled"}>
          <strong title="${escapeHtml(run.label)}">${escapeHtml(run.label)}</strong>
          <span class="run-card-meta">${run.fileCount} JSON file${run.fileCount === 1 ? "" : "s"}${escapeHtml(modified)}</span>
        </button>`;
      }).join("");
      runList.querySelectorAll(".run-card").forEach(function bindRunCard(button) {
        const run = runs.find(function findRun(item) { return item.id === button.dataset.runId; });
        button.addEventListener("click", function selectServerRun() { loadServerRun(run, button); });
      });
    }

    async function fetchApi(url) {
      const response = await fetch(url, { cache: "no-store" });
      const text = await response.text();
      let payload;
      try {
        payload = JSON.parse(text);
      } catch (error) {
        const preview = text.trim().slice(0, 120).replace(/\s+/g, " ");
        throw new Error(`API did not return JSON${preview ? `: ${preview}` : " (empty response)"}`);
      }
      if (!response.ok) {
        const apiMessage = payload.error || payload.errors?.[0]?.error;
        throw new Error(apiMessage || `server returned ${response.status}`);
      }
      return payload;
    }

    async function loadServerRuns() {
      el("runList").innerHTML = `<div class="run-empty">Checking server run folders…</div>`;
      try {
        const payload = await fetchApi("api.php?action=runs");
        const runs = Array.isArray(payload.runs) ? payload.runs : [];
        renderRunList(runs);
        setStatus(runs.length
          ? `Found ${runs.length} server run${runs.length === 1 ? "" : "s"} · select one to load, or choose a folder from this PC`
          : "No server runs found · you can still load a folder from this PC");
      } catch (error) {
        el("runList").innerHTML = `<div class="run-empty">Could not list server runs.</div>`;
        setStatus(`Server run list unavailable: ${error.message}. You can still load files from this PC.`, true);
      }
    }

    async function loadServerRun(run, button) {
      if (!run) return;
      setStatus(`Loading server run ${run.label}…`);
      button.disabled = true;
      try {
        const payload = await fetchApi(`api.php?action=metrics&run=${encodeURIComponent(run.id)}`);
        const records = [];
        const failures = [...(payload.errors || [])];
        for (const file of payload.files || []) {
          try {
            records.push(normalizeRecord(file.data, file.name, file.path, file.runGroup));
          } catch (error) {
            failures.push({ name: file.name, error: error.message });
          }
        }
        if (!records.length) throw new Error("this run has no valid metric JSON files");
        activateRecords(records, `Server / ${run.label}`);
        clearRunSelection();
        button.classList.add("active");
        const notes = [`Loaded ${records.length} checkpoint${records.length === 1 ? "" : "s"} from ${run.label}`];
        if (failures.length) notes.push(`${failures.length} invalid file${failures.length === 1 ? "" : "s"} skipped`);
        setStatus(notes.join(" · "), failures.length > 0);
      } catch (error) {
        setStatus(`Could not load ${run.label}: ${error.message}`, true);
      } finally {
        button.disabled = false;
      }
    }

    function onFolderInputChange(event) { readFiles(event.target.files); }
    function onModelChange() { updateTypeFilter(); applyFilters(); }
    el("folderInput").addEventListener("change", onFolderInputChange);
    el("modelSelect").addEventListener("change", onModelChange);
    el("typeSelect").addEventListener("change", applyFilters);
    el("demoButton").addEventListener("click", loadDemo);
    el("refreshRunsButton").addEventListener("click", loadServerRuns);

    const dropZone = el("dropZone");
    function showDropTarget(event) { event.preventDefault(); dropZone.classList.add("dragging"); }
    function hideDropTarget(event) { event.preventDefault(); dropZone.classList.remove("dragging"); }
    function bindShowDropTarget(type) { dropZone.addEventListener(type, showDropTarget); }
    function bindHideDropTarget(type) { dropZone.addEventListener(type, hideDropTarget); }
    ["dragenter", "dragover"].forEach(bindShowDropTarget);
    ["dragleave", "drop"].forEach(bindHideDropTarget);
    dropZone.addEventListener("drop", async function onFolderDrop(event) {
      try {
        await readFiles(await droppedFiles(event.dataTransfer));
      } catch (error) {
        setStatus(`Could not read the dropped folder: ${error.message}`, true);
      }
    });

    chartDefinitions.forEach(function bindChartEvents(definition) {
      el(definition.canvas).addEventListener("mousemove", function onChartMouseMove(event) {
        const chart = state.charts.get(definition.id);
        if (chart) showTooltip(event, chart);
      });
      el(definition.canvas).addEventListener("mouseleave", function onChartMouseLeave() {
        el("tooltip").style.display = "none";
      });
    });

    let resizeTimer;
    function redrawAllCharts() { state.charts.forEach(drawChart); }
    window.addEventListener("resize", function onWindowResize() {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(redrawAllCharts, 100);
    });

    loadServerRuns();
  </script>
</body>
</html>
