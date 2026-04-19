const state = {
  trees: [],
  filter: 'all'
};

// 1. Classification Logic (Ported from Python)
function classifyHealth(tree) {
  const v = tree.vari || 0;
  const e = tree.greenness_exg || 0;
  if (v < 0 || e < 20) return "stressed";
  if (v < 0.02 || e < 60) return "moderate";
  return "healthy";
}

// 2. File Upload Handler
document.getElementById('jsonUpload').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(event) {
    try {
      const rawData = JSON.parse(event.target.result);
      const data = Array.isArray(rawData) ? rawData : (rawData.trees || []);
      
      // Process Data
      state.trees = data.map(t => ({
        ...t,
        health_status: classifyHealth(t),
        // Simple normalization for map positioning (0-100)
        // Note: In production, use the UTM normalization logic from previous step
        plot_x: ((t.geo_x % 1000) / 1000) * 100, 
        plot_y: ((t.geo_y % 1000) / 1000) * 100
      }));

      document.getElementById('scanDate').textContent = new Date().toLocaleDateString();
      updateDashboard();
      showToast("Report Generated: " + state.trees.length + " trees processed.");
    } catch (err) {
      showToast("Error parsing JSON.");
    }
  };
  reader.readAsText(file);
});

// 3. UI Update Logic
function updateDashboard() {
  const filtered = state.trees.filter(t => 
    state.filter === 'all' || t.health_status === state.filter
  );

  renderKPIs();
  renderMarkers(filtered);
  renderTable(filtered);
}

function renderKPIs() {
  const total = state.trees.length;
  const healthy = state.trees.filter(t => t.health_status === 'healthy').length;
  const stressed = state.trees.filter(t => t.health_status === 'stressed').length;
  
  const avgVari = state.trees.reduce((a, b) => a + (b.vari || 0), 0) / total;
  const healthScore = Math.round((healthy / total) * 100);

  document.getElementById('kpiTotal').textContent = total.toLocaleString();
  document.getElementById('kpiHealthy').textContent = healthScore + "%";
  document.getElementById('kpiStressed').textContent = Math.round((stressed / total) * 100) + "%";
  document.getElementById('kpiVari').textContent = avgVari.toFixed(3);
  document.getElementById('mainScore').textContent = healthScore;
}

function renderMarkers(data) {
  const container = document.getElementById('markers');
  container.innerHTML = data.map(t => `
    <div class="marker-dot ${t.health_status}" style="left:${t.plot_x}%; top:${t.plot_y}%;"></div>
  `).join('');
}

function renderTable(data) {
  const tbody = document.getElementById('treeRows');
  tbody.innerHTML = data.slice(0, 100).map(t => `
    <tr>
      <td>#${t.tree_id}</td>
      <td><span class="badge ${t.health_status}">${t.health_status}</span></td>
      <td>${t.vari < 0 ? 'Chlorosis' : 'Normal'}</td>
      <td>${(t.vari || 0).toFixed(4)}</td>
    </tr>
  `).join('');
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.setAttribute('aria-hidden', 'false');
  setTimeout(() => t.setAttribute('aria-hidden', 'true'), 3000);
}