import { api, pct, sampleMode, isDeprecatedModel } from "../api/client.js";
import { navigate } from "../router.js";
import { ScoreBadge } from "../components/Common.js";
import CompareCapabilityTree from "../features/compare/CompareCapabilityTree.js";
import ModelCapabilityTree from "../features/model-detail/ModelCapabilityTree.js";
import RetestModelDialog from "../features/model-detail/RetestModelDialog.js";

const { ref, computed, onMounted, watch } = Vue;

export default {
  name: "ModelDetailPage",
  components: { ScoreBadge, CompareCapabilityTree, ModelCapabilityTree, RetestModelDialog },
  props: ["modelKey"],
  setup(props) {
    const summary = ref(null);
    const tree = ref(null);
    const error = ref("");
    const retestOpen = ref(false);
    const submittedJob = ref(null);
    const retryStats = computed(() => {
      const raw = summary.value?.run?.config;
      if (!raw) return null;
      let cfg = {};
      try { cfg = typeof raw === "string" ? JSON.parse(raw) : raw; } catch (_) { return null; }
      const meta = cfg.runner_meta || cfg;
      if (!meta?.retry && !meta?.first_pass) return null;
      return { retry: meta.retry || {}, first_pass: meta.first_pass || {}, config: cfg };
    });
    const load = async () => {
      error.value = "";
      summary.value = null;
      tree.value = null;
      try {
        const [s, t] = await Promise.all([
          api(`/detail/model/${encodeURIComponent(props.modelKey)}/summary`),
          api(`/detail/model/${encodeURIComponent(props.modelKey)}/tree`),
        ]);
        if (isDeprecatedModel(s)) {
          error.value = "该模型已归档。";
          summary.value = null;
          tree.value = null;
          return;
        }
        summary.value = s;
        tree.value = t;
      } catch (e) {
        error.value = e.message;
      }
    };
    const onSubmitted = (job) => {
      submittedJob.value = job;
      retestOpen.value = false;
    };
    onMounted(load);
    watch(() => props.modelKey, load);
    return { sampleMode, summary, tree, retryStats, error, retestOpen, submittedJob, load, onSubmitted, pct, navigate };
  },
  template: `
    <section class="page">
      <div class="page-title">
        <div>
          <h1 class="mono">{{ modelKey }}</h1>
          <div v-if="!sampleMode" class="subtle">Model Key 作为平台唯一身份；API Model 仅用于调用推理服务。</div>
        </div>
        <div class="toolbar">
          <button v-if="!sampleMode" @click="retestOpen=true" :disabled="!summary">重测模型</button>
          <button @click="load">刷新</button>
        </div>
      </div>
      <div v-if="error" class="error">{{ error }}</div>
      <div v-if="submittedJob" class="card card-pad" style="margin-bottom:12px">
        已提交重测任务 <span class="mono">{{ submittedJob.id }}</span>，
        <span class="link" @click="navigate('/runs')">去 Runs 查看</span>
      </div>
      <template v-if="summary && !sampleMode">
        <div class="grid grid-4" style="margin-bottom:12px">
          <div class="card card-pad"><div class="subtle">Current Score</div><div style="font-size:22px"><ScoreBadge :value="summary.run?.overall_pass_rate" /></div></div>
          <div class="card card-pad"><div class="subtle">API Model</div><div class="mono">{{ summary.model_id }}</div></div>
          <div class="card card-pad"><div class="subtle">Group</div><div><span class="chip">{{ summary.group_name || '-' }}</span></div></div>
          <div class="card card-pad"><div class="subtle">Latest Run</div><div class="mono">{{ summary.run?.id || '-' }}</div></div>
        </div>
        <div class="card card-pad" style="margin-bottom:12px">
          <div class="grid grid-2">
            <div><div class="subtle">Description</div><div>{{ summary.description || '-' }}</div></div>
            <div><div class="subtle">Provider</div><div class="mono">{{ summary.provider_id || '-' }}</div></div>
          </div>
        </div>
        <div v-if="retryStats" class="card card-pad" style="margin-bottom:12px">
          <div class="subtle" style="margin-bottom:8px">Two-stage Retry</div>
          <div class="grid grid-4">
            <div><div class="subtle">Recovery Candidates</div><div class="mono">{{ retryStats.retry.first_pass_failed ?? retryStats.first_pass.n_failed ?? '-' }}</div></div>
            <div><div class="subtle">Retried</div><div class="mono">{{ retryStats.retry.attempted ?? '-' }}</div></div>
            <div><div class="subtle">Recovered</div><div class="mono">{{ retryStats.retry.recovered ?? '-' }}</div></div>
            <div><div class="subtle">Still Failed</div><div class="mono">{{ retryStats.retry.still_failed ?? '-' }}</div></div>
          </div>
          <div class="subtle" style="margin-top:8px">
            retry concurrency={{ retryStats.retry.concurrency ?? '-' }},
            timeout={{ retryStats.retry.request_timeout ?? '-' }}s
          </div>
        </div>
      </template>
      <CompareCapabilityTree v-if="tree && sampleMode" :tree="tree" />
      <ModelCapabilityTree v-else-if="tree && tree.latest_run" :tree="tree" :model-key="modelKey" />
      <div v-else-if="tree && !tree.latest_run" class="card card-pad subtle">该模型还没有 completed run。</div>
      <RetestModelDialog v-if="!sampleMode" :open="retestOpen" :model="summary" @close="retestOpen=false" @submitted="onSubmitted" />
    </section>
  `,
};
