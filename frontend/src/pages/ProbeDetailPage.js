import { api } from "../api/client.js";
import { navigate } from "../router.js";

const { ref, onMounted, watch } = Vue;

export default {
  name: "ProbeDetailPage",
  props: ["probeId"],
  setup(props) {
    const data = ref(null);
    const error = ref("");
    const load = async () => {
      error.value = "";
      try {
        data.value = await api(`/detail/probe-detail/${encodeURIComponent(props.probeId)}`);
      } catch (e) { error.value = e.message; }
    };
    onMounted(load);
    watch(() => props.probeId, load);
    return { data, error, navigate, load };
  },
  template: `
    <section class="page">
      <div class="page-title"><div><h1 class="mono">{{ probeId }}</h1><div class="subtle">题目级 Debug</div></div><button @click="load">刷新</button></div>
      <div v-if="error" class="error">{{ error }}</div>
      <template v-if="data">
        <div class="card card-pad toolbar" style="margin-bottom:12px">
          <span class="chip">{{ data.category }}</span><span class="chip">{{ data.level }}</span>
          <span class="link mono" @click="navigate('/capabilities/'+encodeURIComponent(data.canonical_id))">{{ data.canonical_id }}</span>
          <span class="spacer"></span>
          <span class="mono">{{ data.n_passed }}/{{ data.n_models }} pass</span>
        </div>
        <div class="grid grid-2" style="margin-bottom:12px">
          <div class="answer"><div class="subtle">Prompt</div><pre>{{ data.prompt || '-' }}</pre></div>
          <div class="answer"><div class="subtle">Ground Truth</div><pre>{{ data.ground_truth || '-' }}</pre></div>
        </div>
        <div class="answer" style="margin-bottom:12px"><div class="subtle">Scorer</div><pre>{{ data.scorer || '-' }}</pre></div>
        <div class="answer-grid">
          <div v-for="r in data.results" :key="r.model_tag" class="answer">
            <div class="toolbar" style="margin-bottom:6px">
              <span class="link mono" @click="navigate('/models/'+encodeURIComponent(r.model_tag))">{{ r.model_tag }}</span>
              <span :class="r.passed ? 'pass' : 'fail'">{{ r.passed ? 'PASS' : 'FAIL' }}</span>
              <span class="spacer"></span>
              <span class="subtle">{{ r.completion_tokens || '-' }} tok · {{ r.latency_s == null ? '-' : r.latency_s.toFixed(2)+'s' }}</span>
            </div>
            <pre>{{ r.response || '-' }}</pre>
          </div>
        </div>
      </template>
    </section>
  `,
};
