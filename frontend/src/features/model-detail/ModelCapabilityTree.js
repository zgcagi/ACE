import { pct, scoreClass } from "../../api/client.js";
import ProbeRender from "../../components/ProbeRender.js";

const { ref } = Vue;

export default {
  name: "ModelCapabilityTree",
  components: { ProbeRender },
  props: ["tree", "modelKey"],
  setup() {
    const open = ref(new Set());
    const toggle = (key) => {
      const next = new Set(open.value);
      next.has(key) ? next.delete(key) : next.add(key);
      open.value = next;
    };
    const isOpen = (key) => open.value.has(key);
    return { open, toggle, isOpen, pct, scoreClass };
  },
  template: `
    <div class="card">
      <div class="tree-row tree-head" style="--model-count:3">
        <div class="tree-cell">测试点</div>
        <div class="tree-cell">通过率</div>
        <div class="tree-cell">Passed</div>
        <div class="tree-cell">Meta</div>
      </div>
      <template v-for="cat in tree.categories" :key="cat.id">
        <div class="tree-row" style="--model-count:3">
          <div class="tree-cell indent-0 link" @click="toggle('c:'+cat.id)">{{ isOpen('c:'+cat.id) ? '▾' : '▸' }} {{ cat.id }}</div>
          <div class="tree-cell"><span class="score mono" :class="scoreClass(cat.pass_rate)">{{ pct(cat.pass_rate) }}</span></div>
          <div class="tree-cell mono">{{ cat.n_passed }}/{{ cat.n_total }}</div>
          <div class="tree-cell subtle">{{ cat.atoms.length }} atoms</div>
        </div>
        <template v-if="isOpen('c:'+cat.id)" v-for="atom in cat.atoms" :key="atom.canonical_id">
          <div class="tree-row" style="--model-count:3">
            <div class="tree-cell indent-1 link" @click="toggle('a:'+atom.canonical_id)">{{ isOpen('a:'+atom.canonical_id) ? '▾' : '▸' }} <span class="mono">{{ atom.canonical_id }}</span></div>
            <div class="tree-cell"><span class="score mono" :class="scoreClass(atom.pass_rate)">{{ pct(atom.pass_rate) }}</span></div>
            <div class="tree-cell mono">{{ atom.n_passed }}/{{ atom.n_total }}</div>
            <div class="tree-cell subtle">{{ atom.levels.length }} levels</div>
          </div>
          <template v-if="isOpen('a:'+atom.canonical_id)" v-for="lv in atom.levels" :key="atom.canonical_id+lv.level">
            <div class="tree-row" style="--model-count:3">
              <div class="tree-cell indent-2 link" @click="toggle('l:'+atom.canonical_id+':'+lv.level)">{{ isOpen('l:'+atom.canonical_id+':'+lv.level) ? '▾' : '▸' }} {{ lv.level }}</div>
              <div class="tree-cell"><span class="score mono" :class="scoreClass(lv.pass_rate)">{{ pct(lv.pass_rate) }}</span></div>
              <div class="tree-cell mono">{{ lv.n_passed }}/{{ lv.n_total }}</div>
              <div class="tree-cell subtle">{{ lv.probes.length }} probes</div>
            </div>
            <template v-if="isOpen('l:'+atom.canonical_id+':'+lv.level)" v-for="probe in lv.probes" :key="probe.probe_id">
              <div class="tree-row" style="--model-count:3">
                <div class="tree-cell indent-3">
                  <span class="subtle">•</span>
                  <span :class="probe.passed ? 'pass' : 'fail'">{{ probe.passed ? '✓' : '✗' }}</span>
                  <span class="mono">{{ probe.probe_id }}</span>
                </div>
                <div class="tree-cell"><span :class="probe.passed ? 'pass' : 'fail'">{{ probe.passed ? 'PASS' : 'FAIL' }}</span></div>
                <div class="tree-cell mono">{{ probe.completion_tokens || '-' }} tok</div>
                <div class="tree-cell mono subtle">{{ probe.latency_s == null ? '-' : probe.latency_s.toFixed(2)+'s' }}</div>
              </div>
              <div class="detail-panel"><ProbeRender :probe="probe" /></div>
            </template>
          </template>
        </template>
      </template>
    </div>
  `,
};
