import { api, pct, visibleModelsOnly } from "../api/client.js";
import { navigate } from "../router.js";
import { ScoreBadge } from "../components/Common.js";

const { ref, onMounted, watch } = Vue;

export default {
  name: "CapabilityDetailPage",
  components: { ScoreBadge },
  props: ["atomId"],
  setup(props) {
    const data = ref(null);
    const error = ref("");
    const load = async () => {
      error.value = "";
      try { data.value = await api(`/detail/atoms/${encodeURIComponent(props.atomId)}`); if (data.value?.models) data.value.models = visibleModelsOnly(data.value.models); } catch (e) { error.value = e.message; }
    };
    onMounted(load);
    watch(() => props.atomId, load);
    return { data, error, pct, navigate };
  },
  template: `
    <section class="page">
      <div class="page-title"><div><h1 class="mono">{{ atomId }}</h1><div class="subtle">Capability detail</div></div><button @click="load">刷新</button></div>
      <div v-if="error" class="error">{{ error }}</div>
      <template v-if="data">
        <div class="grid grid-2" style="margin-bottom:12px">
          <div class="card card-pad"><div class="subtle">Category</div><span class="chip">{{ data.category }}</span></div>
          <div class="card card-pad"><div class="subtle">Probes</div><div class="mono" style="font-size:20px">{{ data.n_probes }}</div></div>
        </div>
        <div class="card" style="margin-bottom:12px">
          <table><thead><tr><th>Model Key</th><th>Group</th><th>Pass Rate</th><th>Passed</th></tr></thead>
          <tbody><tr v-for="m in data.models" :key="m.model_tag" @click="navigate('/models/'+encodeURIComponent(m.model_tag))" style="cursor:pointer">
            <td><span class="link mono">{{ m.model_tag }}</span></td><td><span class="chip">{{ m.group_name }}</span></td><td><ScoreBadge :value="m.pass_rate" /></td><td class="mono">{{ m.n_passed }}/{{ m.n_total }}</td>
          </tr></tbody></table>
        </div>
        <div class="card">
          <table><thead><tr><th>Level</th><th>Probe</th><th>Prompt</th></tr></thead>
          <tbody><tr v-for="p in data.probes" :key="p.probe_id" @click="navigate('/probes/'+encodeURIComponent(p.probe_id))" style="cursor:pointer">
            <td><span class="chip">{{ p.level }}</span></td><td><span class="link mono">{{ p.probe_id }}</span></td><td class="subtle">{{ (p.prompt || '').slice(0,160) }}</td>
          </tr></tbody></table>
        </div>
      </template>
    </section>
  `,
};
