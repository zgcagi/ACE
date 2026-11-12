const { reactive } = Vue;

export const routes = [
  { path: "/", id: "dashboard", label: "Dashboard", icon: "◈" },
  { path: "/models", id: "models", label: "Models", icon: "▤" },
  { path: "/compare", id: "compare", label: "Compare", icon: "⇄" },
  { path: "/capabilities", id: "capabilities", label: "Capabilities", icon: "◎" },
  { path: "/diagnostics", id: "diagnostics", label: "Diagnostics", icon: "◉" },
  { path: "/runs", id: "runs", label: "Runs", icon: "✚" },
  { path: "/playground", id: "playground", label: "Playground", icon: "▷" },
];

function currentPath() {
  const path = location.hash.slice(1);
  return path.startsWith("/") ? path : "/";
}

export const route = reactive({ path: currentPath() });

export function navigate(path) {
  location.hash = path;
  route.path = currentPath();
}

export function parseRoute() {
  const path = route.path;
  const parts = path.split("/").filter(Boolean);
  if (path === "/") return { name: "dashboard", params: {} };
  if (path === "/models") return { name: "models", params: {} };
  if (parts[0] === "models" && parts[1]) return { name: "model-detail", params: { modelKey: decodeURIComponent(parts.slice(1).join("/")) } };
  if (path === "/compare") return { name: "compare", params: {} };
  if (path === "/capabilities") return { name: "capabilities", params: {} };
  if (parts[0] === "capabilities" && parts[1]) return { name: "capability-detail", params: { atomId: decodeURIComponent(parts.slice(1).join("/")) } };
  if (parts[0] === "probes" && parts[1]) return { name: "probe-detail", params: { probeId: decodeURIComponent(parts.slice(1).join("/")) } };
  if (path === "/diagnostics") return { name: "diagnostics", params: {} };
  if (path === "/runs") return { name: "runs", params: {} };
  if (path === "/playground") return { name: "playground", params: {} };
  return { name: "dashboard", params: {} };
}

window.addEventListener("hashchange", () => {
  route.path = currentPath();
});
