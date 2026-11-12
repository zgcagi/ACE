import { sampleMode, token, setToken } from "../api/client.js";
import { navigate, routes } from "../router.js";

export default {
  name: "AppShell",
  props: ["routeInfo"],
  setup() {
    const sampleRoutes = new Set(["dashboard", "models", "compare"]);
    const visibleRoutes = sampleMode ? routes.filter((item) => sampleRoutes.has(item.id)) : routes;
    return { sampleMode, token, setToken, navigate, visibleRoutes };
  },
  template: `
    <div class="app">
      <aside class="sidebar">
        <div class="brand" @click="navigate('/')"><span>ACE Eval Platform</span></div>
        <nav class="nav">
          <div v-for="item in visibleRoutes" :key="item.id" class="nav-item"
            :class="{active: routeInfo.name===item.id || (item.id==='models' && routeInfo.name==='model-detail') || (item.id==='capabilities' && routeInfo.name==='capability-detail')}"
            @click="navigate(item.path)">
            <span>{{ item.icon }}</span><span class="nav-label">{{ item.label }}</span>
          </div>
        </nav>
        <div v-if="!sampleMode" class="token-box">
          <input type="password" :value="token" @input="setToken($event.target.value)" placeholder="API Token">
        </div>
      </aside>
      <main class="main">
        <div class="topbar">
          <span class="link" @click="navigate('/')">Home</span>
          <span class="subtle">/</span>
          <span class="muted">{{ routeInfo.name }}</span>
          <span v-if="sampleMode" class="sample-label">ACE 2.6.4 样例结果</span>
          <slot name="topbar"></slot>
        </div>
        <slot></slot>
      </main>
    </div>
  `,
};
