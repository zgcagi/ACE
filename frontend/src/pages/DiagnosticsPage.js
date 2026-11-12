import { api, visibleModelsOnly } from "../api/client.js";
import { navigate } from "../router.js";
import { ScoreBadge } from "../components/Common.js";

const { ref, computed, onMounted } = Vue;

export default {
  name: "DiagnosticsPage",
  components: { ScoreBadge },
  setup() {
    const rows = ref([]);
    const q = ref("");
    const error = ref("");
    const filtered = computed(() => rows.value.filter((r) => !q.value || r.tag.includes(q.value)));
    const load = async () => {
      error.value = "";
      try { rows.value = visibleModelsOnly(await api("/analysis/model-stats")); } catch (e) { error.value = e.message; }
    };
    onMounted(load);
    return { rows, q, error, filtered, load, navigate };
  },
  template: `
    <section class="page">
      <div class="page-title"><div><h1>Diagnostics</h1><div class="subtle">准确率之外的 token、latency、截断和效率指标。</div></div><button @click="load">刷新</button></div>
      <div v-if="error" class="error">{{ error }}</div>
      <div class="card card-pad toolbar" style="margin-bottom:12px"><input v-model="q" placeholder="搜索 Model Key" style="max-width:320px"><span class="subtle">{{ filtered.length }} rows</span></div>
      <div class="card" style="overflow:auto">
        <table style="min-width:1200px"><thead><tr><th>Model Key</th><th>Group</th><th>Pass</th><th>Avg Tok</th><th>P95 Tok</th><th>Total Tok</th><th>Avg Lat</th><th>P95 Lat</th><th>Stop%</th><th>Trunc%</th><th>Best</th><th>Worst</th></tr></thead>
        <tbody><tr v-for="r in filtered" :key="r.tag" @click="navigate('/models/'+encodeURIComponent(r.tag))" style="cursor:pointer">
          <td><span class="link mono">{{ r.tag }}</span></td><td><span class="chip">{{ r.group }}</span></td><td><ScoreBadge :value="r.overall_rate" /></td>
          <td class="mono">{{ Math.round(r.avg_tokens || 0) }}</td><td class="mono">{{ Math.round(r.p95_tokens || 0) }}</td><td class="mono">{{ Math.round(r.total_tokens || 0) }}</td>
          <td class="mono">{{ (r.avg_latency || 0).toFixed(2) }}</td><td class="mono">{{ (r.p95_latency || 0).toFixed(2) }}</td>
          <td class="mono">{{ ((r.stop_rate || 0)*100).toFixed(1) }}</td><td class="mono">{{ ((r.truncation_rate || 0)*100).toFixed(1) }}</td>
          <td class="subtle">{{ r.best_category }}</td><td class="subtle">{{ r.worst_category }}</td>
        </tr></tbody></table>
      </div>
    </section>
  `,
};
