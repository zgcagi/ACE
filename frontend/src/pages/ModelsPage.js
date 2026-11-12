import { api, sampleMode, visibleModelsOnly } from "../api/client.js";
import { navigate } from "../router.js";
import { ScoreBadge } from "../components/Common.js";

const { ref, computed, onMounted } = Vue;

export default {
  name: "ModelsPage",
  components: { ScoreBadge },
  setup() {
    const error = ref("");
    const leaderboard = ref([]);
    const groups = ref([]);
    const group = ref("all");
    const query = ref("");

    const filtered = computed(() => {
      let list = leaderboard.value;
      if (group.value !== "all") list = list.filter((m) => m.group_name === group.value);
      if (query.value.trim()) {
        const q = query.value.trim().toLowerCase();
        list = list.filter((m) => `${m.model_tag} ${m.model_name || ""} ${m.group_name || ""}`.toLowerCase().includes(q));
      }
      return list;
    });

    const load = async () => {
      error.value = "";
      try {
        const [lb, gs] = await Promise.all([api("/analysis/leaderboard"), api("/models/groups")]);
        leaderboard.value = visibleModelsOnly(lb);
        groups.value = gs.filter((g) => g.group && String(g.group).toLowerCase() !== "deprecated");
      } catch (e) {
        error.value = e.message;
      }
    };
    onMounted(load);
    return { sampleMode, error, leaderboard, groups, group, query, filtered, load, navigate };
  },
  template: `
    <section class="page">
      <div class="page-title">
        <div><h1>Models</h1><div class="subtle">{{ sampleMode ? "点击模型查看各能力类别和难度的通过率。" : "一行一个 Model Key，成绩取最新 completed run。" }}</div></div>
        <button @click="load">刷新</button>
      </div>
      <div v-if="error" class="error">{{ error }}</div>
      <div class="card card-pad toolbar" style="margin-bottom:12px">
        <input v-model="query" placeholder="搜索 Model Key / group" style="max-width:340px">
        <select v-model="group" style="max-width:180px">
          <option value="all">all groups</option>
          <option v-for="g in groups" :key="g.group" :value="g.group">{{ g.group }} ({{ g.count }})</option>
        </select>
        <span class="subtle">{{ filtered.length }} models</span>
      </div>
      <div class="card">
        <table>
          <thead><tr><th>#</th><th>Model Key</th><th>Group</th><th>Score</th><th>Passed (effective)</th><th>{{ sampleMode ? "Snapshot" : "Run" }}</th></tr></thead>
          <tbody>
            <tr v-for="(m,i) in filtered" :key="m.model_tag" @click="navigate('/models/'+encodeURIComponent(m.model_tag))" style="cursor:pointer">
              <td class="mono subtle">{{ i + 1 }}</td>
              <td><span class="link mono">{{ m.model_tag }}</span><div v-if="!sampleMode" class="subtle">{{ m.model_desc || m.model_name }}</div></td>
              <td><span class="chip">{{ m.group_name || '-' }}</span></td>
              <td><ScoreBadge :value="m.overall_pass_rate" /></td>
              <td class="mono">{{ m.n_passed }}/{{ m.n_total - (m.n_excluded || 0) }}</td>
              <td class="mono subtle">{{ sampleMode ? m.completed_at : m.run_id }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  `,
};
