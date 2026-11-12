import { api } from "../api/client.js";
import { navigate } from "../router.js";

const { ref, computed, onMounted } = Vue;

export default {
  name: "CapabilitiesPage",
  setup() {
    const atoms = ref([]);
    const q = ref("");
    const cat = ref("");
    const error = ref("");
    const categories = computed(() => [...new Set(atoms.value.map((a) => a.category).filter(Boolean))].sort());
    const filtered = computed(() => atoms.value.filter((a) => (!cat.value || a.category === cat.value) && (!q.value || a.canonical_id.includes(q.value))));
    const load = async () => {
      try { atoms.value = await api("/detail/atoms"); } catch (e) { error.value = e.message; }
    };
    onMounted(load);
    return { atoms, q, cat, error, categories, filtered, load, navigate };
  },
  template: `
    <section class="page">
      <div class="page-title"><div><h1>Capabilities</h1><div class="subtle">测试集能力目录，按 category 和 canonical atom 浏览。</div></div><button @click="load">刷新</button></div>
      <div v-if="error" class="error">{{ error }}</div>
      <div class="card card-pad toolbar" style="margin-bottom:12px">
        <input v-model="q" placeholder="搜索 capability" style="max-width:320px">
        <select v-model="cat" style="max-width:180px"><option value="">all categories</option><option v-for="c in categories" :key="c">{{ c }}</option></select>
        <span class="subtle">{{ filtered.length }} atoms</span>
      </div>
      <div class="card">
        <table><thead><tr><th>Category</th><th>Capability</th><th>Probes</th></tr></thead>
        <tbody><tr v-for="a in filtered" :key="a.canonical_id" @click="navigate('/capabilities/'+encodeURIComponent(a.canonical_id))" style="cursor:pointer">
          <td><span class="chip">{{ a.category }}</span></td><td><span class="link mono">{{ a.canonical_id }}</span></td><td class="mono">{{ a.n_probes }}</td>
        </tr></tbody></table>
      </div>
    </section>
  `,
};
