const { computed } = Vue;

function parseJson(value) {
  if (typeof value !== "string") return null;
  const text = value.trim();
  if (!text || !/^[\[{]/.test(text)) return null;
  try { return JSON.parse(text); } catch { return null; }
}

function asMessages(value) {
  const parsed = parseJson(value);
  const messages = Array.isArray(parsed) ? parsed : parsed?.messages;
  if (!Array.isArray(messages)) return [];
  return messages
    .filter((m) => m && typeof m === "object" && ("role" in m || "content" in m))
    .map((m) => ({ role: m.role || "message", content: typeof m.content === "string" ? m.content : JSON.stringify(m.content, null, 2) }));
}

function pretty(value) {
  const parsed = parseJson(value);
  if (parsed) return JSON.stringify(parsed, null, 2);
  return value || "-";
}

function valuePreview(value) {
  if (Array.isArray(value)) return `${value.length} items`;
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value ?? "");
}

function scorerSummary(value) {
  const parsed = parseJson(value);
  if (!parsed || typeof parsed !== "object") return null;
  const rubric = parsed.rubric || parsed.criteria || parsed.instruction || "";
  const threshold = parsed.pass_threshold ?? parsed.threshold ?? parsed.min_score ?? null;
  return {
    type: parsed.type || "scorer",
    rubric,
    threshold,
    keys: Object.entries(parsed)
      .filter(([key]) => !["type", "rubric", "criteria", "instruction", "pass_threshold", "threshold", "min_score"].includes(key))
      .slice(0, 6)
      .map(([key, val]) => ({
        key,
        value: valuePreview(val),
      })),
    raw: JSON.stringify(parsed, null, 2),
  };
}

const TextPanel = {
  name: "TextPanel",
  props: ["title", "text", "kind"],
  setup(props) {
    const messages = computed(() => asMessages(props.text));
    const body = computed(() => pretty(props.text));
    return { messages, body };
  },
  template: `
    <div class="probe-section" :class="kind ? 'section-' + kind : ''">
      <div class="section-title">{{ title }}</div>
      <div v-if="messages.length" class="message-list">
        <div v-for="(m,i) in messages" :key="i" class="message-block">
          <div class="message-role">{{ m.role }}</div>
          <pre class="text-block">{{ m.content }}</pre>
        </div>
      </div>
      <pre v-else class="text-block">{{ body }}</pre>
    </div>
  `,
};

export default {
  name: "ProbeRender",
  components: { TextPanel },
  props: ["probe", "modelKeys"],
  setup(props) {
    const scorer = computed(() => scorerSummary(props.probe?.scorer));
    const keys = computed(() => props.modelKeys || []);
    return { scorer, keys };
  },
  template: `
    <div class="probe-detail">
      <div class="probe-meta">
        <span class="mono">{{ probe.probe_id }}</span>
        <span class="chip">{{ probe.level }}</span>
        <span class="chip">{{ probe.category }}</span>
        <span v-if="probe.completion_tokens != null" class="subtle">{{ probe.completion_tokens }} tok</span>
        <span v-if="probe.latency_s != null" class="subtle">{{ probe.latency_s.toFixed(2) }}s</span>
        <span style="flex:1"></span>
      </div>

      <div class="probe-main-grid">
        <TextPanel title="题目 Prompt" :text="probe.prompt" kind="prompt" />
        <TextPanel title="参考答案 / Ground Truth" :text="probe.ground_truth" kind="truth" />
      </div>

      <template v-if="keys.length">
        <div class="answer-grid">
          <div v-for="key in keys" :key="key" class="probe-section output-section" :class="probe.models[key]?.passed ? 'pass-output' : 'fail-output'">
            <div class="output-title">
              <span class="mono">{{ key }}</span>
              <span v-if="probe.models[key]" :class="probe.models[key].passed ? 'pass' : 'fail'">{{ probe.models[key].passed ? 'PASS' : 'FAIL' }}</span>
              <span style="flex:1"></span>
            </div>
            <pre class="text-block output-text">{{ probe.models[key]?.response || '-' }}</pre>
            <div class="subtle output-meta">
              {{ probe.models[key]?.completion_tokens || '-' }} tok ·
              {{ probe.models[key]?.latency_s == null ? '-' : probe.models[key].latency_s.toFixed(2)+'s' }}
              <span v-if="probe.models[key]?.score_note"> · {{ probe.models[key].score_note }}</span>
            </div>
          </div>
        </div>
      </template>
      <template v-else>
        <div class="probe-section output-section" :class="probe.passed ? 'pass-output' : 'fail-output'">
          <div class="output-title">
            <span class="section-title" style="margin:0">模型输出</span>
            <span :class="probe.passed ? 'pass' : 'fail'">{{ probe.passed ? 'PASS' : 'FAIL' }}</span>
            <span style="flex:1"></span>
          </div>
          <pre class="text-block output-text">{{ probe.response || '-' }}</pre>
        </div>
        <TextPanel v-if="probe.reasoning" title="Reasoning" :text="probe.reasoning" />
        <div v-if="probe.score_note" class="judge-note">{{ probe.score_note }}</div>
      </template>

      <div class="probe-section scorer-section">
        <div class="section-title">Scorer</div>
        <div v-if="scorer" class="scorer-card">
          <div class="toolbar">
            <span class="chip on">{{ scorer.type }}</span>
            <span v-if="scorer.threshold != null" class="kv-chip"><b>pass threshold</b>{{ scorer.threshold }}</span>
            <span v-for="item in scorer.keys" :key="item.key" class="kv-chip"><b>{{ item.key }}</b>{{ item.value }}</span>
          </div>
          <div v-if="scorer.rubric" class="rubric-block">{{ scorer.rubric }}</div>
          <details class="raw-details">
            <summary>查看原始 scorer JSON</summary>
            <pre class="text-block compact">{{ scorer.raw }}</pre>
          </details>
        </div>
        <pre v-else class="text-block compact">{{ probe.scorer || '-' }}</pre>
      </div>
    </div>
  `,
};
