import { sampleApi } from "./sample.js";

const { ref } = Vue;

export const sampleMode = new URLSearchParams(location.search).get("mode") !== "api";
export const token = ref("");

export function setToken(value) {
  token.value = value.trim();
}

const apiRoot = new URL("../../../api", import.meta.url).pathname;

export async function api(path, options = {}) {
  if (sampleMode) return sampleApi(path, options);
  const headers = new Headers(options.headers);
  headers.set("Content-Type", "application/json");
  if (token.value) headers.set("Authorization", `Bearer ${token.value}`);
  const response = await fetch(`${apiRoot}${path}`, { ...options, headers, redirect: "error" });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  if (response.status === 204) return null;
  return response.json();
}

export function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(digits)}%`;
}

export function scoreClass(value) {
  if (value === null || value === undefined) return "";
  if (value >= 0.85) return "good";
  if (value >= 0.65) return "mid";
  return "bad";
}

export function isDeprecatedModel(model) {
  return String(model?.group_name || model?.group || "").trim().toLowerCase() === "deprecated";
}

export function visibleModelsOnly(models) {
  return (models || []).filter((model) => !isDeprecatedModel(model));
}
