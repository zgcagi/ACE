import { pct, scoreClass } from "../api/client.js";

export const ScoreBadge = {
  name: "ScoreBadge",
  props: ["value"],
  setup() { return { pct, scoreClass }; },
  template: `<span class="score mono" :class="scoreClass(value)">{{ pct(value, 2) }}</span>`,
};

export const MetricCard = {
  name: "MetricCard",
  props: ["label", "value", "hint"],
  template: `
    <div class="card card-pad">
      <div class="subtle" style="font-size:10px;text-transform:uppercase">{{ label }}</div>
      <div style="font-size:20px;font-weight:750;margin-top:4px">{{ value }}</div>
      <div v-if="hint" class="subtle" style="margin-top:3px">{{ hint }}</div>
    </div>
  `,
};

export const ErrorBlock = {
  name: "ErrorBlock",
  props: ["error"],
  template: `<div v-if="error" class="error">{{ error }}</div>`,
};

export function groupColor(group) {
  const map = {
    openai: "#10b981",
    claude: "#a78bfa",
    gemini: "#60a5fa",
    qwen: "#22d3ee",
    custom: "#94a3b8",
  };
  return map[group] || "#94a3b8";
}
