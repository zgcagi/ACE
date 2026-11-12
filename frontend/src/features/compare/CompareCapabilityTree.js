import { pct, scoreClass } from "../../api/client.js";
import ProbeRender from "../../components/ProbeRender.js";

const { ref, computed } = Vue;

export default {
  name: "CompareCapabilityTree",
  components: { ProbeRender },
  props: ["tree"],
  setup(props) {
    const open = ref(new Set());
    const toggle = (key) => {
      const next = new Set(open.value);
      next.has(key) ? next.delete(key) : next.add(key);
      open.value = next;
    };
    const isOpen = (key) => open.value.has(key);
    const modelKeys = computed(() => (props.tree?.models || []).map((m) => m.model_key));
    const stat = (bucket, key) => bucket.models?.[key] || null;
    return { open, toggle, isOpen, modelKeys, stat, pct, scoreClass };
  },
  template: `
    <div class="card" :style="{'--model-count': modelKeys.length}">
      <div class="tree-row tree-head">
        <div class="tree-cell">{{ tree.summary_only ? "能力类别 / 难度" : "测试点" }}</div>
        <div v-for="m in modelKeys" :key="m" class="tree-cell mono">{{ m }}</div>
      </div>
      <template v-for="cat in tree.categories" :key="cat.id">
        <div class="tree-row">
          <div class="tree-cell indent-0 link" @click="toggle('c:'+cat.id)">{{ isOpen('c:'+cat.id) ? '▾' : '▸' }} {{ cat.id }}</div>
          <div v-for="m in modelKeys" :key="m" class="tree-cell">
            <span v-if="stat(cat,m)" class="score mono" :class="scoreClass(stat(cat,m).pass_rate)">{{ pct(stat(cat,m).pass_rate, 2) }}</span>
            <span v-else class="subtle">-</span>
          </div>
        </div>
        <template v-if="tree.summary_only && isOpen('c:'+cat.id)">
          <div class="tree-row" v-for="lv in cat.levels" :key="lv.level">
            <div class="tree-cell indent-1 subtle">{{ lv.level }}</div>
            <div v-for="m in modelKeys" :key="m" class="tree-cell">
              <span class="score mono" :class="scoreClass(stat(lv,m)?.pass_rate)">{{ pct(stat(lv,m)?.pass_rate, 2) }}</span>
            </div>
          </div>
        </template>
        <template v-if="isOpen('c:'+cat.id)" v-for="atom in cat.atoms" :key="atom.canonical_id">
          <div class="tree-row">
            <div class="tree-cell indent-1 link" @click="toggle('a:'+atom.canonical_id)">{{ isOpen('a:'+atom.canonical_id) ? '▾' : '▸' }} <span class="mono">{{ atom.canonical_id }}</span></div>
            <div v-for="m in modelKeys" :key="m" class="tree-cell">
              <span v-if="stat(atom,m)" class="score mono" :class="scoreClass(stat(atom,m).pass_rate)">{{ pct(stat(atom,m).pass_rate) }}</span>
              <span v-else class="subtle">-</span>
            </div>
          </div>
          <template v-if="isOpen('a:'+atom.canonical_id)" v-for="lv in atom.levels" :key="atom.canonical_id+lv.level">
            <div class="tree-row">
              <div class="tree-cell indent-2 link" @click="toggle('l:'+atom.canonical_id+':'+lv.level)">{{ isOpen('l:'+atom.canonical_id+':'+lv.level) ? '▾' : '▸' }} {{ lv.level }}</div>
              <div v-for="m in modelKeys" :key="m" class="tree-cell">
                <span v-if="stat(lv,m)" class="score mono" :class="scoreClass(stat(lv,m).pass_rate)">{{ pct(stat(lv,m).pass_rate) }}</span>
                <span v-else class="subtle">-</span>
              </div>
            </div>
            <template v-if="isOpen('l:'+atom.canonical_id+':'+lv.level)" v-for="probe in lv.probes" :key="probe.probe_id">
              <div class="tree-row">
                <div class="tree-cell indent-3"><span class="subtle">•</span> <span class="mono">{{ probe.probe_id }}</span></div>
                <div v-for="m in modelKeys" :key="m" class="tree-cell">
                  <span v-if="probe.models[m]" :class="probe.models[m].passed ? 'pass' : 'fail'">{{ probe.models[m].passed ? '✓ PASS' : '✗ FAIL' }}</span>
                  <span v-else class="subtle">missing</span>
                </div>
              </div>
              <div class="detail-panel"><ProbeRender :probe="probe" :model-keys="modelKeys" /></div>
            </template>
          </template>
        </template>
      </template>
    </div>
  `,
};
