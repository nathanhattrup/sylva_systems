const state = {
    trees: [],
    filter: 'all'
};

// Classification logic
function classifyHealth(t) {
    const v = t.vari || 0;
    const e = t.greenness_exg || 0;
    if (v < 0 || e < 20) return "stressed";
    if (v < 0.02 || e < 60) return "moderate";
    return "healthy";
}

document.getElementById('jsonUpload').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(event) {
        try {
            const rawData = JSON.parse(event.target.result);
            const data = Array.isArray(rawData) ? rawData : (rawData.trees || []);

            // Calculate map bounds for normalization
            const xCoords = data.map(t => t.geo_x);
            const yCoords = data.map(t => t.geo_y);
            const bounds = {
                minX: Math.min(...xCoords), maxX: Math.max(...xCoords),
                minY: Math.min(...yCoords), maxY: Math.max(...yCoords)
            };

            state.trees = data.map(t => ({
                ...t,
                health_status: classifyHealth(t),
                // Normalize 0-100% for CSS
                plot_x: ((t.geo_x - bounds.minX) / (bounds.maxX - bounds.minX)) * 100,
                plot_y: 100 - (((t.geo_y - bounds.minY) / (bounds.maxY - bounds.minY)) * 100)
            }));

            updateDashboard();
            showToast(`Analysis Complete: ${state.trees.length} trees found.`);
        } catch (err) {
            showToast("Invalid JSON Format");
        }
    };
    reader.readAsText(file);
});

function updateDashboard() {
    const filtered = state.trees.filter(t =>
        state.filter === 'all' || t.health_status === state.filter
    );

    // Update KPIs
    const healthyCount = state.trees.filter(t => t.health_status === 'healthy').length;
    const stressedCount = state.trees.filter(t => t.health_status === 'stressed').length;
    const avgVari = state.trees.reduce((a, b) => a + (b.vari || 0), 0) / state.trees.length;
    const avgExg = state.trees.reduce((a, b) => a + (b.greenness_exg || 0), 0) / state.trees.length;

    document.getElementById('kpiTotal').textContent = state.trees.length.toLocaleString();
    document.getElementById('mainScore').textContent = Math.round((healthyCount / state.trees.length) * 100);
    document.getElementById('kpiHealthy').textContent = Math.round((healthyCount / state.trees.length) * 100) + "%";
    document.getElementById('kpiStressed').textContent = Math.round((stressedCount / state.trees.length) * 100) + "%";
    document.getElementById('kpiVari').textContent = avgVari.toFixed(4);
    document.getElementById('kpiExg').textContent = avgExg.toFixed(1);

    // Render Map
    const markers = document.getElementById('markers');
    markers.innerHTML = filtered.slice(0, 2000).map(t => `
        <div class="marker-dot ${t.health_status}" style="left:${t.plot_x}%; top:${t.plot_y}%"></div>
    `).join('');

    // Render Table
    const rows = document.getElementById('treeRows');
    rows.innerHTML = filtered.slice(0, 50).map(t => `
        <tr>
            <td>#${t.tree_id}</td>
            <td><span class="status-dot ${t.health_status}"></span> ${t.health_status}</td>
            <td>${t.vari < 0.01 ? 'Nutrient Stress' : 'Optimal'}</td>
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