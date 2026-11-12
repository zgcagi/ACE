import { api, pct, sampleMode, visibleModelsOnly } from "../api/client.js";
import CompareCharts from "../features/compare/CompareCharts.js";
import CompareCapabilityTree from "../features/compare/CompareCapabilityTree.js";
import { ScoreBadge } from "../components/Common.js";

const { ref, computed, onMounted } = Vue;

export default {
  name: "ComparePage",
  components: { CompareCharts, CompareCapabilityTree, ScoreBadge },
  setup() {
    const leaderboard = ref([]);
    const groups = ref([]);
    const selected = ref([]);
    const group = ref("all");
    const query = ref("");
    const tree = ref(null);
    const error = ref("");
    const busy = ref(false);

    const candidates = computed(() => {
      let list = leaderboard.value;
      if (group.value !== "all") list = list.filter((m) => m.group_name === group.value);
      if (query.value.trim()) {
        const q = query.value.trim().toLowerCase();
        list = list.filter((m) => m.model_tag.toLowerCase().includes(q));
      }
      return list.slice(0, 80);
    });

    const toggle = (key) => {
      const i = selected.value.indexOf(key);
      if (i >= 0) selected.value.splice(i, 1);
      else if (selected.value.length < 8) selected.value.push(key);
    };
    const selectTop = (n) => { selected.value = candidates.value.slice(0, n).map((m) => m.model_tag); };
    const loadBase = async () => {
      const [lb, gs] = await Promise.all([api("/analysis/leaderboard"), api("/models/groups")]);
      leaderboard.value = visibleModelsOnly(lb);
      groups.value = gs.filter((g) => g.group && String(g.group).toLowerCase() !== "deprecated");
      if (!selected.value.length) selected.value = leaderboard.value.slice(0, 3).map((m) => m.model_tag);
    };
    const run = async () => {
      if (!selected.value.length) return;
      busy.value = true;
      error.value = "";
      try {
        tree.value = await api(`/deep/tree-matrix?models=${selected.value.map(encodeURIComponent).join(",")}`);
      } catch (e) {
        error.value = e.message;
      } finally {
        busy.value = false;
      }
    };
    onMounted(async () => {
      try {
        await loadBase();
        if (sampleMode) await run();
      } catch (e) {
        error.value = e.message;
      }
    });
    return { sampleMode, leaderboard, groups, selected, group, query, candidates, tree, error, busy, toggle, selectTop, run, pct };
  },
  template: `
    <section class="page">
      <div class="page-title">
        <div><h1>Compare</h1><div class="subtle">{{ sampleMode ? "选择模型后点击分析。展开类别可查看 L1–L3 通过率。" : "按类别、原子能力、难度和题目逐级对比模型。" }}</div></div>
        <button @click="run" :disabled="busy || !selected.length">{{ busy ? '分析中...' : '分析' }}</button>
      </div>
      <div v-if="error" class="error" style="margin-bottom:12px">{{ error }}</div>
      <div class="card card-pad" style="margin-bottom:12px">
        <div class="toolbar" style="margin-bottom:10px">
          <input v-model="query" placeholder="搜索 Model Key" style="max-width:320px">
          <select v-model="group" style="max-width:180px">
            <option value="all">all groups</option>
            <option v-for="g in groups" :key="g.group" :value="g.group">{{ g.group }}</option>
          </select>
          <button @click="selectTop(3)">Top 3</button>
          <button @click="selectTop(5)">Top 5</button>
          <button @click="selected=[];tree=null">清空</button>
        </div>
        <div class="toolbar">
          <span v-for="m in candidates" :key="m.model_tag" class="chip" :class="{on:selected.includes(m.model_tag)}" @click="toggle(m.model_tag)">
            {{ m.model_tag }} <ScoreBadge :value="m.overall_pass_rate" />
          </span>
        </div>
      </div>
      <CompareCharts v-if="tree" :tree="tree" />
      <CompareCapabilityTree v-if="tree" :tree="tree" />
    </section>
  `,
};
