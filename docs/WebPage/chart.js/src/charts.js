import Chart from "chart.js/auto";

function getCSS(name) {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
}

function themeColors() {
  return {
    cpu: getCSS("--cpu-color"),
    gpu: getCSS("--gpu-color"),
    grid: getCSS("--chart-grid"),
    tick: getCSS("--text-secondary"),
  };
}

export function createBenchmarkChart(canvasId, data) {
  const colors = themeColors();

  return new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: {
      labels: ["CPU", "GPU"],
      datasets: [
        {
          data: [data.cpu_ms, data.gpu_ms],
          backgroundColor: [colors.cpu, colors.gpu],
          borderRadius: 4,
          barThickness: 44,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 800, easing: "easeOutQuart" },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: getCSS("--bg-card"),
          titleColor: getCSS("--text"),
          bodyColor: getCSS("--text-secondary"),
          borderColor: getCSS("--border"),
          borderWidth: 1,
          padding: 10,
          displayColors: false,
          callbacks: {
            label: (ctx) => `${ctx.parsed.y.toFixed(2)} ms`,
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: colors.grid },
          ticks: { color: colors.tick },
          title: {
            display: true,
            text: "Time (ms)",
            color: colors.tick,
          },
        },
        x: {
          grid: { display: false },
          ticks: { color: colors.tick },
        },
      },
    },
  });
}

export function updateChartTheme(chart) {
  const colors = themeColors();
  chart.data.datasets[0].backgroundColor = [colors.cpu, colors.gpu];
  chart.options.scales.y.grid.color = colors.grid;
  chart.options.scales.y.ticks.color = colors.tick;
  chart.options.scales.y.title.color = colors.tick;
  chart.options.scales.x.ticks.color = colors.tick;
  chart.update("none");
}
