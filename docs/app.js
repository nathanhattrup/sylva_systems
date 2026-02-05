
/* Sylva Systems mock dashboard logic (presentation-friendly) */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  filter: "all",
  layer: "severity",
  query: "",
  selectedId: null,
};

const severityOrder = { high: 0, moderate: 1, low: 2, healthy: 3 };

const sample = {
  lastSync: new Date(),
  coverageConfidence: "93%",
  totalScanned: 1114,
  trees: [
    // id, zone, severity, risk, confidence, lastSeen, notes, lat/lon (fake), trend
    { id:"A-014", zone:"Zone A", severity:"high", risk:"Canker suspected", confidence:0.92, lastSeen:"2h ago", x:18, y:22, trend:"+12% risk", notes:"Active lesions along lower trunk. Recommend pruning + sanitize tools. Consider fungicide per local extension guidance." },
    { id:"C-102", zone:"Zone C", severity:"high", risk:"Severe drought stress", confidence:0.88, lastSeen:"3h ago", x:62, y:28, trend:"+18% risk", notes:"Canopy thinning and leaf curl observed. Check irrigation distribution and soil moisture." },
    { id:"B-033", zone:"Zone B", severity:"moderate", risk:"Nutrient deficiency", confidence:0.81, lastSeen:"Today", x:38, y:55, trend:"+5% risk", notes:"Chlorosis pattern suggests nitrogen deficiency. Recommend soil test and targeted amendment." },
    { id:"D-009", zone:"Zone D", severity:"moderate", risk:"Pest pressure", confidence:0.77, lastSeen:"Yesterday", x:74, y:62, trend:"+6% risk", notes:"Possible mite/aphid clustering. Scout underside of leaves; consider integrated pest management." },
    { id:"A-087", zone:"Zone A", severity:"low", risk:"Minor canopy damage", confidence:0.74, lastSeen:"Today", x:24, y:70, trend:"-2% risk", notes:"Small limb abrasion; monitor after next storm." },
    { id:"B-118", zone:"Zone B", severity:"low", risk:"Early blight indicators", confidence:0.69, lastSeen:"2d ago", x:44, y:18, trend:"+1% risk", notes:"Faint spotting detected. Low severity—monitor and compare next scan." },
    { id:"C-041", zone:"Zone C", severity:"healthy", risk:"No issues detected", confidence:0.95, lastSeen:"Today", x:56, y:42, trend:"stable", notes:"Healthy baseline used for zone reference." },
    { id:"D-077", zone:"Zone D", severity:"healthy", risk:"No issues detected", confidence:0.94, lastSeen:"Yesterday", x:80, y:44, trend:"stable", notes:"Healthy baseline used for model calibration." },
  ],
  flights: [
    { title: "Flight Complete", sub: "Captured multispectral + RGB • 170 acres • 14 minutes", when: "Today • 1:20 PM" },
    { title: "Model Inference", sub: "Risk scoring + severity classification finished", when: "Today • 1:26 PM" },
    { title: "Operator Review", sub: "High severity items validated", when: "Today • 1:41 PM" },
    { title: "Report Published", sub: "Dashboard synced to farm account", when: "Today • 1:45 PM" },
  ]
};

function badge(sev) {
  const label = sev[0].toUpperCase() + sev.slice(1);
  return `<span class="badge ${sev}"><span class="miniDot"></span>${label}</span>`;
}

function riskAction(t) {
  if (t.severity === "high") {
    return {
      title: `Immediate action: ${t.id}`,
      meta: `${t.zone} • ${t.risk} • ${(t.confidence*100).toFixed(0)}% confidence`,
      hint: "Prioritize inspection within 24–48 hours. Consider isolating spread vectors and documenting treatment."
    };
  }
  if (t.severity === "moderate") {
    return {
      title: `Monitor soon: ${t.id}`,
      meta: `${t.zone} • ${t.risk} • ${(t.confidence*100).toFixed(0)}% confidence`,
      hint: "Re-scan in 7–14 days. Spot-check physical symptoms and compare against zone baseline."
    };
  }
  if (t.severity === "low") {
    return {
      title: `Track trend: ${t.id}`,
      meta: `${t.zone} • ${t.risk} • ${(t.confidence*100).toFixed(0)}% confidence`,
      hint: "No urgent action. Keep an eye on progression and weather-related stressors."
    };
  }
  return {
    title: `Healthy baseline: ${t.id}`,
    meta: `${t.zone} • ${t.risk}`,
    hint: "Use as a reference tree for seasonal comparisons."
  };
}

function applyFilters() {
  const q = state.query.trim().toLowerCase();
  const filtered = sample.trees
    .filter(t => state.filter === "all" ? true : t.severity === state.filter)
    .filter(t => {
      if (!q) return true;
      return (
        t.id.toLowerCase().includes(q) ||
        t.zone.toLowerCase().includes(q) ||
        t.risk.toLowerCase().includes(q)
      );
    })
    .sort((a,b) => {
      const d = severityOrder[a.severity] - severityOrder[b.severity];
      if (d !== 0) return d;
      return b.confidence - a.confidence;
    });

  renderKPIs();
  renderRows(filtered);
  renderMarkers(filtered);
  renderActions(filtered);
}

function renderKPIs() {
  $("#lastSync").textContent = sample.lastSync.toLocaleString();
  $("#kpiCoverage").textContent = sample.coverageConfidence;

  $("#kpiScanned").textContent = sample.totalScanned.toString();
  $("#countTotal").textContent = sample.totalScanned.toString();

  const high = sample.trees.filter(t => t.severity==="high").length;
  const mod  = sample.trees.filter(t => t.severity==="moderate").length;
  const low  = sample.trees.filter(t => t.severity==="low").length;

  $("#kpiHigh").textContent = high.toString();
  $("#kpiMod").textContent = mod.toString();
  $("#kpiLow").textContent = low.toString();
}

function renderRows(rows) {
  $("#countShown").textContent = rows.length.toString();

  const tbody = $("#treeRows");
  tbody.innerHTML = rows.map(t => `
    <tr class="rowBtn" data-id="${t.id}">
      <td><b>${t.id}</b></td>
      <td>${t.zone}</td>
      <td>${badge(t.severity)}</td>
      <td>${t.risk}</td>
      <td>${Math.round(t.confidence*100)}%</td>
    </tr>
  `).join("");

  $$("#treeRows tr").forEach(tr => {
    tr.addEventListener("click", () => openDrawer(tr.dataset.id));
  });
}

function renderMarkers(rows) {
  const markers = $("#markers");

  // Only plot non-healthy trees on the map
  const mapRows = rows.filter(t => t.severity !== "healthy");

  markers.innerHTML = mapRows.map(t => `
    <div class="marker" style="left:${t.x}%; top:${t.y}%;" data-id="${t.id}">
      <div class="markerDot ${t.severity}" title="${t.id} • ${t.risk}"></div>
      <div class="markerLabel">${t.id}</div>
    </div>
  `).join("");

  $$("#markers .marker").forEach(m => {
    m.addEventListener("click", () => openDrawer(m.dataset.id));
  });
}

// function renderMarkers(rows) {
//   const markers = $("#markers");
//   markers.innerHTML = rows.map(t => `
//     <div class="marker" style="left:${t.x}%; top:${t.y}%;" data-id="${t.id}">
//       <div class="markerDot ${t.severity}" title="${t.id} • ${t.risk}"></div>
//       <div class="markerLabel">${t.id}</div>
//     </div>
//   `).join("");

//   $$("#markers .marker").forEach(m => {
//     m.addEventListener("click", () => openDrawer(m.dataset.id));
//   });
// }

function renderActions(rows) {
  const prioritized = rows.slice(0, 4);
  $("#actions").innerHTML = prioritized.map(t => {
    const a = riskAction(t);
    return `
      <div class="actionCard">
        <div class="actionTop">
          <div class="actionTitle">${a.title}</div>
          ${badge(t.severity)}
        </div>
        <div class="actionMeta">${a.meta}</div>
        <div class="actionHint">${a.hint}</div>
      </div>
    `;
  }).join("");
}

function renderTimeline() {
  $("#timeline").innerHTML = sample.flights.map(e => `
    <div class="event">
      <div class="eventDot"></div>
      <div class="eventBody">
        <div class="eventTitle">${e.title}</div>
        <div class="eventSub">${e.sub}</div>
        <div class="eventSub">${e.when}</div>
      </div>
    </div>
  `).join("");
}

function openDrawer(id) {
  const t = sample.trees.find(x => x.id === id);
  if (!t) return;

  state.selectedId = id;
  $("#drawerTitle").textContent = `Tree ${t.id}`;
  $("#drawerSub").textContent = `${t.zone} • ${t.risk} • ${Math.round(t.confidence*100)}% confidence`;
  $("#drawerBody").innerHTML = `
    <div class="cardMini">
      <div class="grid2">
        <div class="metric"><span>Severity</span><b>${t.severity.toUpperCase()}</b></div>
        <div class="metric"><span>Trend</span><b>${t.trend}</b></div>
        <div class="metric"><span>Last observed</span><b>${t.lastSeen}</b></div>
        <div class="metric"><span>Zone</span><b>${t.zone}</b></div>
      </div>
    </div>

    <div class="cardMini">
      <div class="metric"><span>Notes</span><b style="font-size:13px; line-height:1.4; font-weight:700;">${t.notes}</b></div>
    </div>

    <div class="cardMini">
      <div class="metric"><span>Suggested next steps</span>
        <b style="font-size:13px; line-height:1.4; font-weight:700;">
          ${t.severity==="high" ? "Inspect in 24–48 hours • Document symptoms • Plan intervention and re-scan" :
            t.severity==="moderate" ? "Spot-check within 7–14 days • Compare against zone baseline • Re-scan" :
            t.severity==="low" ? "Track trend • Include in next scheduled scan • No immediate action" :
            "Use as reference • Continue normal monitoring"}
        </b>
      </div>
    </div>
  `;

  const drawer = $("#drawer");
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
}

function closeDrawer() {
  const drawer = $("#drawer");
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  t.setAttribute("aria-hidden", "false");
  setTimeout(() => {
    t.classList.remove("show");
    t.setAttribute("aria-hidden", "true");
  }, 1600);
}

function wireUI() {
  // Filters
  $$(".chip").forEach(btn => {
    btn.addEventListener("click", () => {
      $$(".chip").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.filter = btn.dataset.filter;
      applyFilters();
    });
  });

  // Layer segmented (mock)
  $$(".segBtn").forEach(btn => {
    btn.addEventListener("click", () => {
      $$(".segBtn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.layer = btn.dataset.layer;
      toast(`Map layer: ${state.layer}`);
    });
  });

  // Search
  $("#search").addEventListener("input", (e) => {
    state.query = e.target.value;
    applyFilters();
  });

  // Ctrl+K focus search
  window.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      $("#search").focus();
    }
    if (e.key === "Escape") closeDrawer();
  });

  // Drawer controls
  $("#drawerClose").addEventListener("click", closeDrawer);
  $("#btnMarkReviewed").addEventListener("click", () => toast("Marked reviewed"));
  $("#btnCreateWorkOrder").addEventListener("click", () => toast("Work order created (mock)"));

  // Topbar buttons (mock)
  $("#btnExport").addEventListener("click", () => toast("Export started (mock)"));
  $("#btnSchedule").addEventListener("click", () => toast("Scan scheduled (mock)"));

  // Demo sync - change a couple severities randomly for “new flight”
  $("#btnDemoSync").addEventListener("click", () => {
    sample.lastSync = new Date();
    // bump 1-2 random trees slightly
    const idxs = [Math.floor(Math.random()*sample.trees.length), Math.floor(Math.random()*sample.trees.length)];
    idxs.forEach(i => {
      const t = sample.trees[i];
      const cycle = ["healthy","low","moderate","high"];
      const pos = cycle.indexOf(t.severity);
      const next = Math.min(3, Math.max(0, pos + (Math.random() > 0.5 ? 1 : -1)));
      t.severity = cycle[next];
      t.lastSeen = "Just now";
      t.trend = (next >= pos) ? "+3% risk" : "-3% risk";
    });
    toast("New flight synced");
    applyFilters();
  });

  // Sidebar nav (mock)
  $$(".navItem").forEach(btn => {
    btn.addEventListener("click", () => {
      $$(".navItem").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      toast(`View: ${btn.dataset.view} (mock)`);
    });
  });
}

function init() {
  renderTimeline();
  wireUI();
  applyFilters();
}

init();
