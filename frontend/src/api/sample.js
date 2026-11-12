const sampleDirectory = new URL("../../../examples/results/", import.meta.url);
const sampleFiles = ["deepseek-v4-flash.json", "gpt-4o.json", "zgcm-7b-thinking.json"];
const group = "ACE 2.6.4";

async function loadSamples() {
  return Promise.all(sampleFiles.map(async (filename) => {
    const response = await fetch(new URL(filename, sampleDirectory));
    if (!response.ok) throw new Error(`无法读取样例 ${filename}（HTTP ${response.status}）`);
    return response.json();
  }));
}

let samplesPromise;
function samples() {
  samplesPromise ||= loadSamples().catch((error) => {
    samplesPromise = null;
    throw error;
  });
  return samplesPromise;
}

function leaderboardRow(sample) {
  return {
    model_tag: sample.model_name,
    model_name: sample.model_name,
    group_name: group,
    overall_pass_rate: sample.overall_pass_rate,
    n_passed: sample.n_pass,
    n_total: sample.n_total,
    n_excluded: sample.n_excluded,
    n_categories: Object.keys(sample.by_category).length,
    completed_at: sample.snapshot_date,
  };
}

function comparisonTree(selected) {
  const categories = Object.keys(selected[0]?.by_category || {});
  return {
    summary_only: true,
    models: selected.map(({ model_name }) => ({ model_key: model_name })),
    categories: categories.map((id) => ({
      id,
      models: Object.fromEntries(selected.map(({ model_name, by_category }) => [
        model_name, by_category[id],
      ])),
      levels: ["L1", "L2", "L3"].map((level) => ({
        level,
        models: Object.fromEntries(selected.map(({ model_name, by_category }) => [
          model_name, { pass_rate: by_category[id][level] },
        ])),
      })),
    })),
  };
}

export async function sampleApi(path, options) {
  if ((options.method || "GET") !== "GET") throw new Error("样例模式只读取结果");
  const { pathname, searchParams } = new URL(`.${path}`, sampleDirectory);
  const endpoint = pathname.slice(sampleDirectory.pathname.length);
  const results = await samples();
  if (endpoint === "analysis/leaderboard") {
    return results.map(leaderboardRow).sort((a, b) => b.overall_pass_rate - a.overall_pass_rate);
  }
  if (endpoint === "models/groups") return [{ group, count: results.length }];
  if (endpoint === "deep/tree-matrix") {
    const keys = (searchParams.get("models") || "").split(",");
    return comparisonTree(keys.map((key) => results.find((sample) => sample.model_name === key)).filter(Boolean));
  }
  const detail = endpoint.match(/^detail\/model\/([^/]+)\/(summary|tree)$/);
  if (detail) {
    const sample = results.find((item) => item.model_name === decodeURIComponent(detail[1]));
    if (sample && detail[2] === "tree") return comparisonTree([sample]);
    if (sample) return {
      model_id: sample.model_name,
      group_name: group,
      description: sample.source,
      run: { overall_pass_rate: sample.overall_pass_rate },
    };
  }
  throw new Error("此视图需要评测平台 API");
}
