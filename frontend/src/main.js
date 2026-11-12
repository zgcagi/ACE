import { sampleMode } from "./api/client.js";
import AppShell from "./components/AppShell.js";
import { parseRoute, route } from "./router.js";
import DashboardPage from "./pages/DashboardPage.js";
import ModelsPage from "./pages/ModelsPage.js";
import ModelDetailPage from "./pages/ModelDetailPage.js";
import ComparePage from "./pages/ComparePage.js";
import CapabilitiesPage from "./pages/CapabilitiesPage.js";
import CapabilityDetailPage from "./pages/CapabilityDetailPage.js";
import ProbeDetailPage from "./pages/ProbeDetailPage.js";
import DiagnosticsPage from "./pages/DiagnosticsPage.js";
import RunsPage from "./pages/RunsPage.js";
import PlaygroundPage from "./pages/PlaygroundPage.js";

const { createApp, computed } = Vue;

const pages = {
  dashboard: DashboardPage,
  models: ModelsPage,
  "model-detail": ModelDetailPage,
  compare: ComparePage,
  capabilities: CapabilitiesPage,
  "capability-detail": CapabilityDetailPage,
  "probe-detail": ProbeDetailPage,
  diagnostics: DiagnosticsPage,
  runs: RunsPage,
  playground: PlaygroundPage,
};

createApp({
  components: { AppShell },
  setup() {
    const routeInfo = computed(() => parseRoute(route.path));
    const samplePages = new Set(["dashboard", "models", "model-detail", "compare"]);
    const currentPage = computed(() => {
      const name = routeInfo.value.name;
      return sampleMode && !samplePages.has(name) ? DashboardPage : pages[name];
    });
    return { routeInfo, currentPage };
  },
  template: `
    <AppShell :route-info="routeInfo">
      <component :is="currentPage" v-bind="routeInfo.params" />
    </AppShell>
  `,
}).mount("#app");
