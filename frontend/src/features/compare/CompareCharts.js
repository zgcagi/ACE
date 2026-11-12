const { ref, onMounted, onUnmounted, watch } = Vue;

const colors = ["#72b4ff", "#4ad7ae", "#f9ba69", "#d7a7ff", "#f98fa8", "#9db9cc", "#f1e38e", "#77e4f2"];
const radarCategories = ["arith", "code", "instr", "longctx", "math", "read", "tool", "truth"];
const textColor = "#d4d4d8";

function chartData(tree) {
  const names = tree.models.map(({ model_key }) => model_key);
  const categories = tree.categories;
  const scores = categories.map((category) => names.map((name) => {
    const rate = category.models?.[name]?.pass_rate;
    return Number.isFinite(rate) ? rate * 100 : null;
  }));
  return { names, categories, scores };
}

function heatmapOption({ names, categories, scores }) {
  const cells = scores.flatMap((row, i) => row.flatMap((score, j) => (
    score === null ? [] : [{
      value: [j, i, score],
      label: { color: score >= 70 ? "#102238" : "#f4f4f5" },
    }]
  )));
  return {
    animation: false,
    textStyle: { color: textColor },
    grid: { top: 34, bottom: 53, left: 68, right: 20 },
    tooltip: {
      renderMode: "richText",
      formatter: ({ value }) => `${names[value[0]]} · ${categories[value[1]].id} · ${value[2].toFixed(2)}%`,
    },
    xAxis: {
      type: "category", position: "top", data: names,
      axisLabel: { color: textColor, interval: 0, overflow: "truncate", width: 180 },
      axisTick: { show: false },
    },
    yAxis: {
      type: "category", inverse: true, data: categories.map(({ id }) => id),
      axisLabel: { color: textColor, interval: 0 }, axisTick: { show: false },
    },
    visualMap: {
      min: 0, max: 100, orient: "horizontal", left: "center", bottom: 6,
      calculable: false, text: ["100%", "0%"], textStyle: { color: textColor },
      inRange: { color: ["#24243e", "#386287", "#72b8cf", "#9feacb"] },
    },
    series: [{
      type: "heatmap",
      data: cells,
      label: { show: true, formatter: ({ value }) => `${value[2].toFixed(1)}%` },
      itemStyle: { borderWidth: 1.5, borderColor: "#111114" },
    }],
  };
}

function radarOption({ names, categories, scores }) {
  const selected = radarCategories.map((id) => categories.findIndex((category) => category.id === id))
    .filter((index) => index >= 0 && scores[index].every((score) => score !== null));
  if (selected.length < 3) {
    return {
      title: {
        text: "至少需要 3 个共同计分类别",
        left: "center",
        top: "middle",
        textStyle: { color: textColor, fontSize: 13 },
      },
    };
  }
  return {
    animation: false,
    color: colors,
    tooltip: { renderMode: "richText" },
    legend: { type: "scroll", top: 7, data: names, textStyle: { color: textColor } },
    radar: {
      center: ["50%", "55%"],
      radius: "67%",
      indicator: selected.map((i) => ({ name: categories[i].id, max: 100 })),
      axisName: { color: textColor },
      splitNumber: 4,
      splitLine: { lineStyle: { color: "#3f3f46" } },
      axisLine: { lineStyle: { color: "#3f3f46" } },
      splitArea: { show: false },
    },
    series: [{
      type: "radar",
      symbolSize: 5,
      lineStyle: { width: 2 },
      data: names.map((name, j) => ({ name, value: selected.map((i) => scores[i][j]) })),
    }],
  };
}

export default {
  props: ["tree"],
  setup(props) {
    const heatmap = ref(null);
    const radar = ref(null);
    let heatmapChart;
    let radarChart;
    let observer;
    function render() {
      if (!heatmapChart || !radarChart) return;
      const data = chartData(props.tree);
      heatmapChart.setOption(heatmapOption(data), true);
      radarChart.setOption(radarOption(data), true);
    }
    onMounted(() => {
      if (!window.echarts) return;
      heatmapChart = echarts.init(heatmap.value);
      radarChart = echarts.init(radar.value);
      observer = new ResizeObserver(() => {
        heatmapChart.resize();
        radarChart.resize();
      });
      observer.observe(heatmap.value);
      observer.observe(radar.value);
      render();
    });
    watch(() => props.tree, render);
    onUnmounted(() => {
      observer?.disconnect();
      heatmapChart?.dispose();
      radarChart?.dispose();
    });
    return { heatmap, radar, chartsAvailable: Boolean(window.echarts) };
  },
  template: `
    <div v-if="chartsAvailable" class="comparison-charts">
      <section class="card card-pad">
        <h2>类别热力图</h2>
        <div ref="heatmap" class="compare-heatmap" role="img" aria-label="按模型和能力类别显示通过率的热力图"></div>
      </section>
      <section class="card card-pad">
        <h2>能力雷达图</h2>
        <p class="subtle">选定类别的通过率，刻度为 0–100%。</p>
        <div ref="radar" class="compare-radar" role="img" aria-label="各模型在选定类别上的通过率雷达图"></div>
      </section>
    </div>
  `,
};
