import { api, visibleModelsOnly } from "../api/client.js";

const { ref, computed, onMounted } = Vue;

export default {
  name: "PlaygroundPage",
  setup() {
    const models = ref([]);
    const modelKey = ref("");
    const input = ref("");
    const messages = ref([]);
    const temp = ref(0.7);
    const maxTokens = ref(2048);
    const busy = ref(false);
    const error = ref("");
    const history = ref([]);
    const load = async () => {
      try {
        const [lb, hist] = await Promise.all([api("/analysis/leaderboard"), api("/playground/history?limit=20")]);
        models.value = visibleModelsOnly(lb);
        history.value = hist;
        if (!modelKey.value && models.value[0]) modelKey.value = models.value[0].model_tag;
      } catch (e) { error.value = e.message; }
    };
    const send = async () => {
      if (!modelKey.value || !input.value.trim()) return;
      messages.value.push({ role: "user", content: input.value });
      input.value = "";
      busy.value = true;
      try {
        const data = await api("/playground/chat", {
          method: "POST",
          body: JSON.stringify({ model_tag: modelKey.value, messages: messages.value, temperature: temp.value, max_tokens: maxTokens.value }),
        });
        messages.value.push({ role: "assistant", content: data.response?.choices?.[0]?.message?.content || "(empty)" });
        history.value = await api("/playground/history?limit=20");
      } catch (e) {
        messages.value.push({ role: "assistant", content: `Error: ${e.message}` });
      } finally {
        busy.value = false;
      }
    };
    const loadHistory = (h) => {
      try { messages.value = JSON.parse(h.messages); modelKey.value = h.model_tag; } catch {}
    };
    onMounted(load);
    return { models, modelKey, input, messages, temp, maxTokens, busy, error, history, load, send, loadHistory };
  },
  template: `
    <section class="page" style="height:calc(100vh - 45px);display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:12px">
      <div class="card card-pad" style="display:flex;flex-direction:column;min-height:0">
        <div class="toolbar" style="margin-bottom:10px">
          <select v-model="modelKey" style="max-width:360px"><option v-for="m in models" :key="m.model_tag" :value="m.model_tag">{{ m.model_tag }}</option></select>
          <label class="chip">Temp <input v-model.number="temp" type="number" step="0.1" style="width:70px"></label>
          <label class="chip">Max <input v-model.number="maxTokens" type="number" style="width:90px"></label>
          <button @click="messages=[]">清空</button>
        </div>
        <div v-if="error" class="error">{{ error }}</div>
        <div style="flex:1;overflow:auto" class="grid">
          <div v-for="(m,i) in messages" :key="i" class="answer">
            <div class="subtle">{{ m.role }}</div><pre>{{ m.content }}</pre>
          </div>
        </div>
        <div class="toolbar" style="margin-top:10px">
          <textarea v-model="input" rows="3" placeholder="Ctrl+Enter 发送" @keydown.ctrl.enter="send"></textarea>
          <button @click="send" :disabled="busy || !input.trim()">{{ busy ? '发送中...' : '发送' }}</button>
        </div>
      </div>
      <div class="card card-pad" style="overflow:auto">
        <h1 style="font-size:14px;margin-top:0">History</h1>
        <div v-for="h in history" :key="h.id" class="card card-pad" style="margin-bottom:8px;cursor:pointer" @click="loadHistory(h)">
          <div class="mono">{{ h.model_tag }}</div><div class="subtle">{{ h.created_at }}</div>
        </div>
      </div>
    </section>
  `,
};
