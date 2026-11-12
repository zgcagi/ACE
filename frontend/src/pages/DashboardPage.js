import { api, pct, sampleMode, visibleModelsOnly } from "../api/client.js";
import { navigate } from "../router.js";
import { MetricCard, ScoreBadge } from "../components/Common.js";

const { ref, computed, onMounted } = Vue;

export default {
  name: "DashboardPage",
  components: { MetricCard, ScoreBadge },
  setup() {
    const loading = ref(true);
    const error = ref("");
    const leaderboard = ref([]);
    const groups = ref([]);
    const atoms = ref([]);
    const selectedGroup = ref(localStorage.getItem("ace.dashboard.group") || "all");

    const visibleModels = computed(() => {
      const list = selectedGroup.value === "all"
        ? leaderboard.value
        : leaderboard.value.filter((m) => m.group_name === selectedGroup.value);
      return list.slice(0, 5);
    });

    const overview = computed(() => {
      const lb = leaderboard.value;
      const totalCalls = lb.reduce((sum, m) => sum + (m.n_total || 0), 0);
      return {
        models: lb.length,
        probes: lb[0]?.n_total || 0,
        atoms: sampleMode ? lb[0]?.n_categories || 0 : atoms.value.length,
        groups: groups.value.length,
        calls: totalCalls,
      };
    });

    const load = async () => {
      loading.value = true;
      error.value = "";
      try {
        const [lb, gs, atomList] = await Promise.all([
          api("/analysis/leaderboard"),
          api("/models/groups"),
          sampleMode ? [] : api("/detail/atoms"),
        ]);
        leaderboard.value = visibleModelsOnly(lb);
        groups.value = gs.filter((g) => g.group && String(g.group).toLowerCase() !== "deprecated");
        atoms.value = atomList;
        if (selectedGroup.value !== "all" && !groups.value.some((g) => g.group === selectedGroup.value)) {
          selectedGroup.value = "all";
        }
      } catch (e) {
        error.value = e.message;
      } finally {
        loading.value = false;
      }
    };

    const setGroup = (group) => {
      selectedGroup.value = group;
      localStorage.setItem("ace.dashboard.group", group);
    };

    onMounted(load);
    return { sampleMode, loading, error, leaderboard, groups, selectedGroup, visibleModels, overview, load, setGroup, pct, navigate };
  },
  template: `
    <section class="page">
      <div class="page-title">
        <div>
          <h1>Dashboard</h1>
          <div class="subtle">{{ sampleMode ? "三份完整评测样例，按总通过率排序。" : "按模型组展示最新有效成绩前五的模型。" }}</div>
        </div>
        <button @click="load">刷新</button>
      </div>
      <div v-if="error" class="error">{{ error }}</div>
      <div class="grid grid-4" style="margin-bottom:12px">
        <MetricCard label="Models" :value="overview.models" />
        <MetricCard label="Probes" :value="overview.probes" />
        <MetricCard :label="sampleMode ? 'Scored categories' : 'Capabilities'" :value="overview.atoms" />
        <MetricCard label="Groups" :value="overview.groups" />
      </div>
      <div class="card card-pad" style="margin-bottom:12px">
        <div class="toolbar">
          <span class="subtle">Group</span>
          <span class="chip" :class="{on:selectedGroup==='all'}" @click="setGroup('all')">all</span>
          <span v-for="g in groups" :key="g.group" class="chip" :class="{on:selectedGroup===g.group}" @click="setGroup(g.group)">
            {{ g.group }} <span class="subtle">{{ g.count }}</span>
          </span>
        </div>
      </div>
      <div class="card">
        <table>
          <thead><tr><th style="width:54px">#</th><th>Model Key</th><th>Group</th><th>Current Score</th><th>Passed (effective)</th><th>{{ sampleMode ? "Snapshot" : "Latest Run" }}</th></tr></thead>
          <tbody>
            <tr v-for="(m,i) in visibleModels" :key="m.model_tag" @click="navigate('/models/'+encodeURIComponent(m.model_tag))" style="cursor:pointer">
              <td class="mono subtle">{{ i + 1 }}</td>
              <td><span class="link mono">{{ m.model_tag }}</span><div v-if="!sampleMode" class="subtle">{{ m.model_name }}</div></td>
              <td><span class="chip">{{ m.group_name || '-' }}</span></td>
              <td><ScoreBadge :value="m.overall_pass_rate" /></td>
              <td class="mono">{{ m.n_passed }}/{{ m.n_total - (m.n_excluded || 0) }}</td>
              <td class="subtle">{{ m.completed_at || '-' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  `,
};
