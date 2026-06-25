import { fetchBenchData } from "./data.js";
import { createBenchmarkChart, updateChartTheme } from "./charts.js";
import "./style.css";

const charts = {};

const n = (v) => {
  return v.toLocaleString();
};

const ok = (v) => {
  return v
    ? `<span class="check">\u2713 Pass</span>`
    : `<span class="cross">\u2717 Fail</span>`;
};

const spd = (v) => {
  return `<span class="${v >= 1 ? "speedup-green" : "speedup-red"}">${v.toFixed(2)}\u00d7</span>`;
};

const initCharts = (data) => {
  charts.scan = createBenchmarkChart("scan-chart", data.scan);
  charts.histogram = createBenchmarkChart("histogram-chart", data.histogram);
  charts.spmv = createBenchmarkChart("spmv-chart", data.spmv);
};

const populateMetrics = (data) => {
  const specs = {
    scan: [
      ["Elements", n(data.scan.elements)],
      ["Workgroup", String(data.scan.workgroup_size)],
      ["Speedup", spd(data.scan.speedup)],
      ["Bandwidth", `${data.scan.bandwidth_gbps.toFixed(2)} GB/s`],
      ["Result", ok(data.scan.correct)],
    ],
    histogram: [
      ["Elements", n(data.histogram.elements)],
      ["Buckets", String(data.histogram.buckets)],
      ["Speedup", spd(data.histogram.speedup)],
      ["Bandwidth", `${data.histogram.bandwidth_gbps.toFixed(2)} GB/s`],
      ["Result", ok(data.histogram.correct)],
    ],
    spmv: [
      ["Rows", n(data.spmv.rows)],
      ["Non-zeros", n(data.spmv.nnz)],
      ["Speedup", spd(data.spmv.speedup)],
      ["Bandwidth", `${data.spmv.bandwidth_gbps.toFixed(2)} GB/s`],
      ["Result", ok(data.spmv.correct)],
    ],
  };

  for (const [key, metrics] of Object.entries(specs)) {
    const el = document.getElementById(`${key}-metrics`);
    el.innerHTML = metrics
      .map(
        ([label, html]) =>
          `<div class="metric"><span class="metric-label">${label}</span><span class="metric-value">${html}</span></div>`,
      )
      .join("");
  }
};

const populateTable = (data) => {
  const rows = [
    ["Parallel Prefix Sum", data.scan],
    ["Histogram", data.histogram],
    ["SpMV", data.spmv],
  ];

  const tbody = document.querySelector("#summary-table tbody");

  tbody.innerHTML = rows
    .map(([name, d]) => {
      const elems = d.elements
        ? n(d.elements)
        : `${n(d.rows)} \u00d7 ${n(d.cols)}`;

      return `<tr>
        <td>${name}</td>
        <td>${elems}</td>
        <td>${d.cpu_ms.toFixed(2)}</td>
        <td>${d.gpu_ms.toFixed(2)}</td>
        <td>${spd(d.speedup)}</td>
        <td>${d.bandwidth_gbps.toFixed(2)}</td>
        <td>${ok(d.correct)}</td>
      </tr>`;
    })
    .join("");
};

const populateSystemInfo = (data) => {
  const gpu = data.scan?.device || data.histogram?.device || "Unknown GPU";
  document.getElementById("gpu-name").textContent = gpu;
};

const initTheme = () => {
  const saved = localStorage.getItem("theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved);
  const btn = document.getElementById("theme-toggle");
  btn.textContent = saved === "dark" ? "\u{1F319}" : "\u2600\uFE0F";

  btn.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    btn.textContent = next === "dark" ? "\u{1F319}" : "\u2600\uFE0F";
    Object.values(charts).forEach((c) => updateChartTheme(c));
  });
};

const start = async () => {
    initTheme();
    let data;

    try {
        data = await fetchBenchData();
    } 
    catch (error) {
        document.body.innerHTML =
        `<div style="display:flex;align-items:center;justify-content:center;height:100vh;flex-direction:column;gap:1rem;color:var(--error)">` +
        `<h2>Failed to load benchmark data</h2>` +
        `<p>${error.message}</p>` +
        `<p style="font-size:0.85rem;color:var(--text-muted)">Run <code>cargo test run_all_benchmarks</code> first</p></div>`;
        return;
    }

    populateSystemInfo(data);
    initCharts(data);
    populateMetrics(data);
    populateTable(data);
};

if(document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", start);
else
    start();
