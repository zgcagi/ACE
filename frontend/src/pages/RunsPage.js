import { api } from "../api/client.js";
import { navigate } from "../router.js";

const { ref, onMounted, onUnmounted, watch } = Vue;

export default {
  name: "RunsPage",
  setup() {
    const jobs = ref([]);
    const current = ref(null);
    const error = ref("");
    const createOpen = ref(true);
    const createBusy = ref(false);
    const createError = ref("");
    const createNotice = ref("");
    const limitText = ref("");
    let detailPoll = null;
    const form = ref({
      base_url: "",
      api_key: "",
      reuse_provider_id: "",
      tag: "",
      model_id: "",
      description: "",
      group_name: "custom",
      max_tokens: 16384,
      temperature: 0,
      concurrency: 200,
      skip_categories: "",
      request_timeout: 300,
      retry_failed: true,
      retry_concurrency: 8,
      retry_request_timeout: 300,
      force: true,
      thinking_mode: false,
      no_system_prompt: false,
    });
    const isActiveJob = (job) => job && ["queued", "running", "cancelling"].includes(job.status);
    const stopDetailPoll = () => {
      if (detailPoll) clearInterval(detailPoll);
      detailPoll = null;
    };
    const refreshCurrent = async () => {
      if (!current.value?.id) return;
      try {
        current.value = await api(`/auto-eval/jobs/${current.value.id}`);
        if (!isActiveJob(current.value)) stopDetailPoll();
      } catch (e) {
        error.value = e.message;
        stopDetailPoll();
      }
    };
    const startDetailPoll = () => {
      stopDetailPoll();
      if (!isActiveJob(current.value)) return;
      detailPoll = setInterval(refreshCurrent, 3000);
    };
    const load = async () => {
      error.value = "";
      try { jobs.value = await api("/auto-eval/jobs"); } catch (e) { error.value = e.message; }
    };
    const select = async (id) => {
      try { current.value = await api(`/auto-eval/jobs/${id}`); } catch (e) { error.value = e.message; }
    };
    const cancel = async (id) => {
      try { current.value = await api(`/auto-eval/jobs/${id}/cancel`, { method: "POST" }); await load(); } catch (e) { error.value = e.message; }
    };
    const submit = async () => {
      createBusy.value = true;
      createError.value = "";
      createNotice.value = "";
      try {
        const limit = limitText.value === "" || limitText.value == null ? null : Number(limitText.value);
        const body = { ...form.value, limit: Number.isFinite(limit) ? limit : null };
        if (!body.base_url || (!body.api_key && !body.reuse_provider_id) || !body.tag || !body.model_id) {
          createError.value = "base_url / api_key(或 reuse provider) / Model Key / API Model 必填";
          return;
        }
        const job = await api("/auto-eval/jobs", { method: "POST", body: JSON.stringify(body) });
        await load();
        current.value = job;
        createOpen.value = false;
      } catch (e) {
        createError.value = e.message;
      } finally {
        createBusy.value = false;
      }
    };
    const reuse = async (id) => {
      createError.value = "";
      createNotice.value = "";
      try {
        const config = await api(`/auto-eval/jobs/${id}/rerun-config`);
        createNotice.value = [config.scope_reset_warning, config.resume_warning]
          .filter(Boolean).join(" ");
        form.value = {
          ...form.value,
          base_url: config.base_url || "",
          api_key: "",
          reuse_provider_id: config.reuse_provider_id || "",
          tag: config.tag || "",
          model_id: config.model_id || "",
          description: config.description || "",
          group_name: config.group_name || "custom",
          max_tokens: config.max_tokens || 16384,
          temperature: config.temperature ?? 0,
          concurrency: config.concurrency || 200,
          skip_categories: config.skip_categories ?? "",
          retry_failed: config.retry_failed ?? true,
          retry_concurrency: config.retry_concurrency ?? 8,
          retry_request_timeout: config.retry_request_timeout ?? 300,
          force: config.force ?? true,
          thinking_mode: config.thinking_mode ?? false,
          no_system_prompt: config.no_system_prompt ?? false,
        };
        limitText.value = config.limit == null ? "" : String(config.limit);
        createOpen.value = true;
      } catch (e) {
        createError.value = e.message;
      }
    };
    watch(() => `${current.value?.id || ""}:${current.value?.status || ""}`, startDetailPoll);
    onMounted(load);
    onUnmounted(stopDetailPoll);
    return { jobs, current, error, createOpen, createBusy, createError, createNotice, limitText, form, load, select, cancel, submit, reuse, navigate };
  },
  template: `
    <section class="page">
      <div class="page-title"><div><h1>Runs</h1><div class="subtle">评测任务中心。Dashboard 不展示任务，模型详情可快捷重测。</div></div><button @click="load">刷新</button></div>
      <div v-if="error" class="error" style="margin-bottom:12px">{{ error }}</div>
      <div class="card card-pad" style="margin-bottom:12px">
        <div class="toolbar" style="margin-bottom:10px">
          <h1 style="font-size:14px;margin:0">新建评测</h1>
          <span class="subtle">创建新的 Model Key 或复测已有 Model Key。</span>
          <span class="spacer"></span>
          <button @click="createOpen=!createOpen">{{ createOpen ? '收起' : '展开' }}</button>
        </div>
        <template v-if="createOpen">
          <div v-if="createError" class="error" style="margin-bottom:10px">{{ createError }}</div>
          <div v-if="createNotice" style="margin-bottom:10px;color:#f59e0b">{{ createNotice }}</div>
          <div class="grid grid-3">
            <label><span class="subtle">Model Key</span><input v-model="form.tag" placeholder="平台唯一模型名"></label>
            <label><span class="subtle">API Model</span><input v-model="form.model_id" placeholder="推理服务 model 参数"></label>
            <label><span class="subtle">Group</span><input v-model="form.group_name" placeholder="custom"></label>
            <label><span class="subtle">Base URL</span><input v-model="form.base_url" placeholder="https://.../v1"></label>
            <label><span class="subtle">API Key</span><input v-model="form.api_key" type="password" placeholder="sk-..."></label>
            <label><span class="subtle">Reuse Provider</span><input v-model="form.reuse_provider_id" placeholder="留空则使用新 key"></label>
            <label><span class="subtle">Max Tokens</span><input v-model.number="form.max_tokens" type="number"></label>
            <label><span class="subtle">Temperature</span><input v-model.number="form.temperature" type="number" step="0.1"></label>
            <label><span class="subtle">Concurrency（默认 200）</span><input v-model.number="form.concurrency" type="number" min="1" max="200"></label>
            <label><span class="subtle">Retry Concurrency</span><input v-model.number="form.retry_concurrency" type="number" min="1" max="200"></label>
            <label><span class="subtle">Retry Timeout</span><input v-model.number="form.retry_request_timeout" type="number" min="1"></label>
            <label><span class="subtle">Limit（非空仅诊断）</span><input v-model="limitText" placeholder="空为正式全量"></label>
            <label><span class="subtle">Skip Categories（非空仅诊断）</span><input v-model="form.skip_categories" placeholder="空为正式全量"></label>
            <label><span class="subtle">Timeout</span><input v-model.number="form.request_timeout" type="number"></label>
          </div>
          <label style="display:block;margin-top:10px"><span class="subtle">Description</span><textarea v-model="form.description" rows="2" placeholder="可选说明"></textarea></label>
          <div class="toolbar" style="margin-top:10px">
            <label class="chip"><input type="checkbox" v-model="form.force" style="width:auto"> force</label>
            <label class="chip"><input type="checkbox" v-model="form.retry_failed" style="width:auto"> retry failed</label>
            <label class="chip" title="候选模型 Thinking；不影响服务端裁判"><input type="checkbox" v-model="form.thinking_mode" style="width:auto"> thinking</label>
            <label class="chip" title="将 system 内容合并到 user 消息"><input type="checkbox" v-model="form.no_system_prompt" style="width:auto"> no system role</label>
            <span class="spacer"></span>
            <button @click="submit" :disabled="createBusy">{{ createBusy ? '提交中...' : '开始评测' }}</button>
          </div>
        </template>
      </div>
      <div class="runs-layout">
        <div class="card">
          <table><thead><tr><th>Status</th><th>Model Key</th><th>Created</th><th></th></tr></thead>
          <tbody><tr v-for="j in jobs" :key="j.id">
            <td><span class="chip status-chip" :class="'status-' + j.status">{{ j.status }}</span></td>
            <td><span class="link mono" @click="j.tag && navigate('/models/'+encodeURIComponent(j.tag))">{{ j.tag }}</span></td>
            <td class="subtle">{{ j.created_at }}</td>
            <td class="toolbar"><button @click="select(j.id)">查看</button><button @click="reuse(j.id)">复用</button></td>
          </tr></tbody></table>
        </div>
        <div class="card card-pad run-detail-card">
          <template v-if="current">
            <div class="toolbar run-detail-head">
              <h1 style="font-size:14px;margin:0">任务详情</h1>
              <span class="chip status-chip" :class="'status-' + current.status">{{ current.status }}</span>
              <span class="spacer"></span>
              <button v-if="['queued','running','cancelling'].includes(current.status)" @click="cancel(current.id)">停止</button>
            </div>
            <div v-if="current.warning" style="margin:10px 0;color:#f59e0b">{{ current.warning }}</div>
            <div class="run-info-grid">
              <div class="info-item"><div class="subtle">Model Key</div><div class="mono">{{ current.tag || '-' }}</div></div>
              <div class="info-item"><div class="subtle">API Model</div><div class="mono">{{ current.model_id || '-' }}</div></div>
              <div class="info-item"><div class="subtle">Group</div><div>{{ current.group_name || '-' }}</div></div>
              <div class="info-item"><div class="subtle">Base URL</div><div class="mono">{{ current.base_url || '-' }}</div></div>
              <div class="info-item"><div class="subtle">API Key</div><div class="mono">{{ current.api_key ? '********' : '-' }}</div></div>
              <div class="info-item"><div class="subtle">Reuse Provider</div><div class="mono">{{ current.reuse_provider_id || '-' }}</div></div>
              <div class="info-item"><div class="subtle">Max Tokens</div><div class="mono">{{ current.max_tokens ?? '-' }}</div></div>
              <div class="info-item"><div class="subtle">Temperature</div><div class="mono">{{ current.temperature ?? '-' }}</div></div>
              <div class="info-item"><div class="subtle">Concurrency</div><div class="mono">{{ current.concurrency ?? '-' }}</div></div>
              <div class="info-item"><div class="subtle">Retry</div><div class="mono">{{ current.retry_failed ? 'yes' : 'no' }}</div></div>
              <div class="info-item"><div class="subtle">Retry Concurrency</div><div class="mono">{{ current.retry_concurrency ?? '-' }}</div></div>
              <div class="info-item"><div class="subtle">Retry Timeout</div><div class="mono">{{ current.retry_request_timeout ?? '-' }}</div></div>
              <div class="info-item"><div class="subtle">Limit</div><div class="mono">{{ current.limit ?? '-' }}</div></div>
              <div class="info-item"><div class="subtle">Skip Categories</div><div>{{ current.skip_categories || '-' }}</div></div>
              <div class="info-item"><div class="subtle">Timeout</div><div class="mono">{{ current.request_timeout ?? '-' }}</div></div>
              <div class="info-item"><div class="subtle">Force</div><div>{{ current.force ? 'yes' : 'no' }}</div></div>
              <div class="info-item"><div class="subtle">Thinking Mode</div><div>{{ current.thinking_mode ? 'yes' : 'no' }}</div></div>
              <div class="info-item"><div class="subtle">No System Prompt</div><div>{{ current.no_system_prompt ? 'yes' : 'no' }}</div></div>
              <div class="info-item"><div class="subtle">Created</div><div>{{ current.created_at || '-' }}</div></div>
              <div class="info-item"><div class="subtle">Phase</div><div>{{ current.phase || '-' }}</div></div>
              <div class="info-item"><div class="subtle">Judge Deferred</div><div>{{ current.n_judge_deferred ?? '-' }}</div></div>
              <div class="info-item info-wide"><div class="subtle">Description</div><div>{{ current.description || '-' }}</div></div>
            </div>
            <div class="answer log-tail-panel"><div class="subtle">Log Tail</div><pre>{{ current.log_tail || '暂无日志' }}</pre></div>
          </template>
          <div v-else class="subtle">选择一个任务查看详情。</div>
        </div>
      </div>
    </section>
  `,
};
