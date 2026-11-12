import { api } from "../../api/client.js";

const { ref, watch } = Vue;

export default {
  name: "RetestModelDialog",
  props: ["model", "open"],
  emits: ["close", "submitted"],
  setup(props, { emit }) {
    const busy = ref(false);
    const error = ref("");
    const limitText = ref("");
    const form = ref({});

    const reset = () => {
      const m = props.model || {};
      form.value = {
        base_url: m.base_url_override || "",
        api_key: "",
        reuse_provider_id: m.provider_id || "",
        model_id: m.model_id || "",
        tag: m.tag || "",
        description: m.description || m.name || "",
        group_name: m.group_name || "custom",
        max_tokens: m.max_tokens || 16384,
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
      };
      limitText.value = "";
      error.value = "";
    };
    watch(() => props.open, (v) => { if (v) reset(); });

    const submit = async () => {
      busy.value = true;
      error.value = "";
      try {
        const limit = limitText.value === "" ? null : Number(limitText.value);
        const body = { ...form.value, limit: Number.isFinite(limit) ? limit : null };
        const job = await api("/auto-eval/jobs", { method: "POST", body: JSON.stringify(body) });
        emit("submitted", job);
      } catch (e) {
        error.value = e.message;
      } finally {
        busy.value = false;
      }
    };
    return { busy, error, form, limitText, submit };
  },
  template: `
    <div v-if="open" class="modal-backdrop" @click.self="$emit('close')">
      <div class="modal">
        <div class="page-title">
          <div><h1>重测模型</h1><div class="subtle">配置绑定到 Model Key，API Model 仅用于推理调用。</div></div>
          <button @click="$emit('close')">关闭</button>
        </div>
        <div v-if="error" class="error" style="margin-bottom:10px">{{ error }}</div>
        <div class="grid grid-2">
          <label>Model Key<input v-model="form.tag"></label>
          <label>API Model<input v-model="form.model_id"></label>
          <label>Group<input v-model="form.group_name"></label>
          <label>Description<input v-model="form.description"></label>
          <label>Base URL<input v-model="form.base_url" placeholder="https://.../v1"></label>
          <label>API Key<input v-model="form.api_key" type="password" placeholder="留空则尝试复用 provider key"></label>
          <label>Reuse Provider<input v-model="form.reuse_provider_id" placeholder="provider id"></label>
          <label>Max Tokens<input v-model.number="form.max_tokens" type="number"></label>
          <label>Temperature<input v-model.number="form.temperature" type="number" step="0.1"></label>
          <label>Concurrency（默认 200）<input v-model.number="form.concurrency" type="number" min="1" max="200"></label>
          <label>Retry Concurrency<input v-model.number="form.retry_concurrency" type="number" min="1" max="200"></label>
          <label>Retry Timeout<input v-model.number="form.retry_request_timeout" type="number" min="1"></label>
          <label>Timeout<input v-model.number="form.request_timeout" type="number" min="1"></label>
          <label>Limit（非空仅诊断）<input v-model="limitText" placeholder="空为正式全量"></label>
          <label>Skip Categories（非空仅诊断）<input v-model="form.skip_categories" placeholder="空为正式全量"></label>
        </div>
        <div class="toolbar" style="margin-top:12px">
          <label class="chip"><input type="checkbox" v-model="form.force" style="width:auto"> force</label>
          <label class="chip"><input type="checkbox" v-model="form.retry_failed" style="width:auto"> retry failed</label>
          <label class="chip" title="候选模型 Thinking；不影响服务端裁判"><input type="checkbox" v-model="form.thinking_mode" style="width:auto"> thinking</label>
          <label class="chip" title="将 system 内容合并到 user 消息"><input type="checkbox" v-model="form.no_system_prompt" style="width:auto"> no system role</label>
          <span class="spacer"></span>
          <button @click="submit" :disabled="busy">{{ busy ? '提交中...' : '开始重测' }}</button>
        </div>
      </div>
    </div>
  `,
};
