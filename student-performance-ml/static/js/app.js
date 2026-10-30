// Student Performance AI - Frontend Controller

let actualVsPredChart = null;
let coefficientsChart = null;
let studyTimeChart = null;
let failuresChart = null;

let currentDbPage = 0;
const DB_PAGE_SIZE = 15;
let currentSearch = "";

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initTabs();
  initSliderSync();
  initVisToggle();
  initForm();
  initDatabaseExplorer();
  
  // Load initial backend data
  checkSystemStatus();
  loadMetrics();
  loadChartData();
  loadStudentsFromDb();
});

// Theme Management (Light / Dark Mode)
function initTheme() {
  const toggleBtn = document.getElementById("theme-toggle-btn");
  const sunIcon = document.getElementById("theme-icon-sun");
  const moonIcon = document.getElementById("theme-icon-moon");
  const btnText = document.getElementById("theme-btn-text");

  const savedTheme = localStorage.getItem("theme") || "dark";
  applyTheme(savedTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const isLight = document.body.classList.contains("light-theme");
      const newTheme = isLight ? "dark" : "light";
      applyTheme(newTheme);
      localStorage.setItem("theme", newTheme);
    });
  }

  function applyTheme(theme) {
    if (theme === "light") {
      document.body.classList.add("light-theme");
      if (sunIcon) sunIcon.classList.remove("hidden");
      if (moonIcon) moonIcon.classList.add("hidden");
      if (btnText) btnText.textContent = "Dark Mode";
    } else {
      document.body.classList.remove("light-theme");
      if (sunIcon) sunIcon.classList.add("hidden");
      if (moonIcon) moonIcon.classList.remove("hidden");
      if (btnText) btnText.textContent = "Light Mode";
    }
  }
}


// 1. Tab Switching
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.dataset.tab;
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
      }
    });
  });
}

// 2. Synchronize Form Sliders with Value Labels
function initSliderSync() {
  const sliders = [
    { id: "inp-g1", labelId: "val-g1" },
    { id: "inp-g2", labelId: "val-g2" },
    { id: "inp-failures", labelId: "val-failures" },
    { id: "inp-absences", labelId: "val-absences" },
  ];

  sliders.forEach(({ id, labelId }) => {
    const el = document.getElementById(id);
    const label = document.getElementById(labelId);
    if (el && label) {
      el.addEventListener("input", (e) => {
        label.textContent = e.target.value;
      });
    }
  });

  // Load sample student button
  const sampleBtn = document.getElementById("btn-fill-sample");
  if (sampleBtn) {
    sampleBtn.addEventListener("click", () => {
      document.getElementById("inp-g1").value = 14;
      document.getElementById("val-g1").textContent = "14";
      document.getElementById("inp-g2").value = 15;
      document.getElementById("val-g2").textContent = "15";
      document.getElementById("inp-studytime").value = "3";
      document.getElementById("inp-failures").value = 0;
      document.getElementById("val-failures").textContent = "0";
      document.getElementById("inp-absences").value = 2;
      document.getElementById("val-absences").textContent = "2";
      document.getElementById("inp-higher").value = "yes";
      document.getElementById("inp-internet").value = "yes";
      document.getElementById("prediction-form").dispatchEvent(new Event("submit"));
    });
  }
}

// 3. Toggle between Interactive Charts and Rendered Plots
function initVisToggle() {
  const btnInteractive = document.getElementById("btn-show-interactive");
  const btnPlots = document.getElementById("btn-show-plots");
  const viewInteractive = document.getElementById("interactive-charts-view");
  const viewPlots = document.getElementById("rendered-plots-view");

  if (btnInteractive && btnPlots) {
    btnInteractive.addEventListener("click", () => {
      btnInteractive.classList.add("active");
      btnPlots.classList.remove("active");
      viewInteractive.classList.remove("hidden");
      viewPlots.classList.add("hidden");
    });

    btnPlots.addEventListener("click", () => {
      btnPlots.classList.add("active");
      btnInteractive.classList.remove("active");
      viewPlots.classList.remove("hidden");
      viewInteractive.classList.add("hidden");
    });
  }
}

// 4. System Status & MongoDB Health
async function checkSystemStatus() {
  const badge = document.getElementById("mongo-status-badge");
  const text = document.getElementById("mongo-status-text");

  try {
    const res = await fetch("/api/status");
    const data = await res.json();

    if (data.status === "online" && data.mongodb && data.mongodb.connected) {
      badge.className = "status-pill status-online";
      text.textContent = `MongoDB: ${data.mongodb.database} (${data.mongodb.collections.students} students)`;
    } else {
      badge.className = "status-pill status-error";
      text.textContent = "MongoDB Disconnected";
    }
  } catch (err) {
    badge.className = "status-pill status-error";
    text.textContent = "Backend Offline";
  }
}

// 5. Load Model Metrics & Populate Overview Tab
async function loadMetrics() {
  try {
    const res = await fetch("/api/metrics");
    const data = await res.json();
    if (!data.report) return;

    // Linear Regression metrics
    const lr = data.report.regression.find(m => m.model_name === "LinearRegression");
    if (lr && lr.metrics) {
      document.getElementById("metric-r2").textContent = lr.metrics.r2_score.toFixed(4);
      document.getElementById("metric-mae").textContent = lr.metrics.mae.toFixed(4);
      if (lr.metrics.rmse) {
        document.getElementById("metric-rmse").textContent = lr.metrics.rmse.toFixed(4);
      }
    }

    if (data.report.student_count) {
      document.getElementById("metric-total-students").textContent = data.report.student_count;
    }

    // Intercept and top features
    if (data.linear_regression_details) {
      const details = data.linear_regression_details;
      if (details.intercept !== undefined) {
        document.getElementById("equation-intercept").textContent = (details.intercept >= 0 ? "+" : "") + details.intercept.toFixed(3);
      }

      if (details.top_coefficients && details.top_coefficients.length > 0) {
        const pos = details.top_coefficients.find(c => c.coefficient > 0);
        const neg = details.top_coefficients.find(c => c.coefficient < 0);
        if (pos) {
          document.getElementById("top-pos-feature").textContent = `${pos.feature} (+${pos.coefficient.toFixed(3)})`;
        }
        if (neg) {
          document.getElementById("top-neg-feature").textContent = `${neg.feature} (${neg.coefficient.toFixed(3)})`;
        }
      }
    }

    // Populate comparison table with Classification & Regression
    const tbody = document.getElementById("models-comparison-body");
    if (tbody && data.report) {
      let rowsHtml = "";

      // Regression rows
      data.report.regression.forEach(m => {
        const isLR = m.model_name === "LinearRegression";
        rowsHtml += `
          <tr class="${isLR ? 'highlight-row' : ''}">
            <td><strong>${m.model_name}</strong> (Regression)</td>
            <td class="font-mono ${isLR ? 'text-accent text-bold' : ''}">${m.metrics.r2_score.toFixed(4)}</td>
            <td class="font-mono">${m.metrics.mae.toFixed(4)}</td>
            <td class="font-mono">${(m.metrics.rmse || Math.sqrt(m.metrics.mse || 0)).toFixed(4)}</td>
            <td><span class="badge ${isLR ? 'badge-success' : 'badge-muted'}">${isLR ? 'Linear Reg ($R^2$)' : 'Benchmark'}</span></td>
          </tr>
        `;
      });

      // Classification rows (Accuracy, Precision, Recall, F1)
      if (data.report.classification) {
        data.report.classification.forEach(m => {
          rowsHtml += `
            <tr>
              <td><strong>${m.model_name}</strong> (Classification)</td>
              <td class="font-mono">Acc: ${(m.metrics.accuracy * 100).toFixed(1)}%</td>
              <td class="font-mono">Prec: ${m.metrics.precision_macro.toFixed(3)}</td>
              <td class="font-mono">F1: ${m.metrics.f1_macro.toFixed(3)}</td>
              <td><span class="badge badge-primary">Recall: ${m.metrics.recall_macro.toFixed(3)}</span></td>
            </tr>
          `;
        });
      }

      tbody.innerHTML = rowsHtml;
    }

  } catch (err) {
    console.error("Failed to load metrics", err);
  }
}

// 6. Interactive Visualizations via Chart.js
async function loadChartData() {
  try {
    const res = await fetch("/api/chart-data");
    const data = await res.json();
    if (!data.actual_vs_pred) return;

    renderActualVsPredChart(data.actual_vs_pred);
    renderStudyTimeChart(data.study_data);
    renderFailuresChart(data.failure_data);

    // Fetch and render coefficients from metrics API
    const metricsRes = await fetch("/api/metrics");
    const metricsData = await metricsRes.json();
    if (metricsData.linear_regression_details && metricsData.linear_regression_details.top_coefficients) {
      renderCoefficientsChart(metricsData.linear_regression_details.top_coefficients);
    }

  } catch (err) {
    console.error("Failed to load chart data", err);
  }
}

function renderActualVsPredChart(points) {
  const ctx = document.getElementById("actualVsPredChart").getContext("2d");
  if (actualVsPredChart) actualVsPredChart.destroy();

  const scatterData = points.map(p => ({ x: p.actual, y: p.predicted, id: p.id }));

  actualVsPredChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Test Students (Actual vs Predicted)",
          data: scatterData,
          backgroundColor: "rgba(99, 102, 241, 0.75)",
          borderColor: "#818cf8",
          pointRadius: 5,
          pointHoverRadius: 7,
        },
        {
          label: "Ideal Fit (Actual = Predicted)",
          data: [{ x: 0, y: 0 }, { x: 20, y: 20 }],
          type: "line",
          borderColor: "#ef4444",
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: "Actual Grade G3 (0–20)", color: "#94a3b8" },
          min: 0,
          max: 20,
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" }
        },
        y: {
          title: { display: true, text: "Predicted Grade G3 (0–20)", color: "#94a3b8" },
          min: 0,
          max: 20,
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" }
        }
      },
      plugins: {
        legend: { labels: { color: "#e2e8f0" } },
        tooltip: {
          callbacks: {
            label: (ctx) => `Student: ${ctx.raw.id || 'N/A'} | Actual: ${ctx.raw.x} | Predicted: ${ctx.raw.y}`
          }
        }
      }
    }
  });
}

function renderCoefficientsChart(coefficients) {
  const ctx = document.getElementById("coefficientsChart").getContext("2d");
  if (coefficientsChart) coefficientsChart.destroy();

  const top10 = coefficients.slice(0, 10);
  const labels = top10.map(c => c.feature);
  const values = top10.map(c => c.coefficient);
  const colors = values.map(v => v >= 0 ? "rgba(16, 185, 129, 0.85)" : "rgba(239, 68, 68, 0.85)");

  coefficientsChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Coefficient Weight (Impact on G3)",
        data: values,
        backgroundColor: colors,
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" }
        },
        y: {
          grid: { display: false },
          ticks: { color: "#cbd5e1", font: { family: "'JetBrains Mono', monospace", size: 11 } }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function renderStudyTimeChart(data) {
  const ctx = document.getElementById("studyTimeChart").getContext("2d");
  if (studyTimeChart) studyTimeChart.destroy();

  studyTimeChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: data.labels,
      datasets: [{
        label: "Average G3 (0–20)",
        data: data.values,
        backgroundColor: "rgba(59, 130, 246, 0.8)",
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 20,
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" }
        },
        x: {
          grid: { display: false },
          ticks: { color: "#94a3b8" }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function renderFailuresChart(data) {
  const ctx = document.getElementById("failuresChart").getContext("2d");
  if (failuresChart) failuresChart.destroy();

  failuresChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: data.labels,
      datasets: [{
        label: "Mean G3 by Prior Failures",
        data: data.values,
        backgroundColor: "rgba(239, 68, 68, 0.75)",
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 20,
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" }
        },
        x: {
          grid: { display: false },
          ticks: { color: "#94a3b8" }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// 7. Interactive Prediction Submission
function initForm() {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById("btn-predict-submit");
    const originalText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.textContent = "Calculating...";

    const payload = {
      G1: parseFloat(document.getElementById("inp-g1").value),
      G2: parseFloat(document.getElementById("inp-g2").value),
      studytime: parseInt(document.getElementById("inp-studytime").value),
      failures: parseInt(document.getElementById("inp-failures").value),
      absences: parseInt(document.getElementById("inp-absences").value),
      schoolsup: document.getElementById("inp-schoolsup").value,
      famsup: document.getElementById("inp-famsup").value,
      higher: document.getElementById("inp-higher").value,
      internet: document.getElementById("inp-internet").value,
      paid: document.getElementById("inp-paid").value,
      age: parseInt(document.getElementById("inp-age").value),
      sex: document.getElementById("inp-sex").value,
      school: document.getElementById("inp-school").value,
      address: document.getElementById("inp-address").value,
    };

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = await res.json();
      if (result.error) {
        alert("Prediction Error: " + result.error);
        return;
      }

      updatePredictionDisplay(result);

    } catch (err) {
      alert("Failed to reach server: " + err);
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = originalText;
    }
  });

  // Retrain button
  const retrainBtn = document.getElementById("retrain-btn");
  if (retrainBtn) {
    retrainBtn.addEventListener("click", async () => {
      const btnText = document.getElementById("retrain-btn-text");
      btnText.textContent = "Retraining...";
      retrainBtn.disabled = true;

      try {
        const res = await fetch("/api/retrain", { method: "POST" });
        const data = await res.json();
        alert(data.message || "Model retrained successfully!");
        loadMetrics();
        loadChartData();
        checkSystemStatus();
      } catch (err) {
        alert("Retrain error: " + err);
      } finally {
        btnText.textContent = "Retrain & Sync";
        retrainBtn.disabled = false;
      }
    });
  }
}

function updatePredictionDisplay(res) {
  const grade = res.predicted_grade;
  document.getElementById("result-grade-val").textContent = grade.toFixed(1);

  // Update radial SVG ring: circumference = 2 * PI * 68 ~= 427
  const circumference = 427;
  const progressRatio = Math.min(Math.max(grade / 20, 0), 1);
  const offset = circumference - (circumference * progressRatio);
  const fillBar = document.getElementById("gauge-fill");
  fillBar.style.strokeDashoffset = offset;
  fillBar.style.stroke = res.category_color || "#6366f1";

  // Tier badge
  const badge = document.getElementById("result-tier-badge");
  badge.textContent = res.category_label || res.category;
  badge.className = `tier-pill tier-${(res.category || "medium").toLowerCase()}`;

  // Breakdown values
  document.getElementById("breakdown-raw-val").textContent = res.raw_grade.toFixed(2);
  const diff = grade - 10.42;
  const diffEl = document.getElementById("breakdown-diff-val");
  diffEl.textContent = (diff >= 0 ? "+" : "") + diff.toFixed(2);
  diffEl.style.color = diff >= 0 ? "var(--color-success)" : "var(--color-danger)";

  document.getElementById("breakdown-clf-tier").textContent = res.classifier_tier || res.category;
  document.getElementById("result-timestamp").textContent = new Date().toLocaleTimeString();
}

// 8. MongoDB Explorer Tab
function initDatabaseExplorer() {
  const searchInp = document.getElementById("db-search-input");
  const refreshBtn = document.getElementById("btn-refresh-db");
  const prevBtn = document.getElementById("btn-prev-page");
  const nextBtn = document.getElementById("btn-next-page");

  if (searchInp) {
    searchInp.addEventListener("input", (e) => {
      currentSearch = e.target.value;
      currentDbPage = 0;
      loadStudentsFromDb();
    });
  }

  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      loadStudentsFromDb();
      checkSystemStatus();
    });
  }

  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      if (currentDbPage > 0) {
        currentDbPage--;
        loadStudentsFromDb();
      }
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      currentDbPage++;
      loadStudentsFromDb();
    });
  }

  // Modal close
  const modal = document.getElementById("json-modal");
  const closeBtn = document.getElementById("btn-close-modal");
  if (closeBtn && modal) {
    closeBtn.addEventListener("click", () => modal.classList.add("hidden"));
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.classList.add("hidden");
    });
  }
}

async function loadStudentsFromDb() {
  const tbody = document.getElementById("db-students-tbody");
  tbody.innerHTML = `<tr><td colspan="11" class="text-center">Querying MongoDB documents...</td></tr>`;

  try {
    const skip = currentDbPage * DB_PAGE_SIZE;
    const url = `/api/students?limit=${DB_PAGE_SIZE}&skip=${skip}&search=${encodeURIComponent(currentSearch)}`;
    const res = await fetch(url);
    const data = await res.json();

    if (!data.students || data.students.length === 0) {
      tbody.innerHTML = `<tr><td colspan="11" class="text-center">No student records found in MongoDB.</td></tr>`;
      document.getElementById("pagination-info").textContent = "0 of 0 records";
      return;
    }

    const total = data.total;
    const start = skip + 1;
    const end = Math.min(skip + data.students.length, total);
    document.getElementById("pagination-info").textContent = `Showing ${start}–${end} of ${total} records`;

    document.getElementById("btn-prev-page").disabled = currentDbPage === 0;
    document.getElementById("btn-next-page").disabled = end >= total;

    let rows = "";
    data.students.forEach(st => {
      rows += `
        <tr>
          <td class="font-mono text-accent">${st.student_id}</td>
          <td>${st.school}</td>
          <td>${st.sex}</td>
          <td>${st.age}</td>
          <td>${st.studytime} (${st.studytime * 2.5}h)</td>
          <td>${st.failures}</td>
          <td>${st.absences}</td>
          <td class="font-mono">${st.G1}</td>
          <td class="font-mono">${st.G2}</td>
          <td class="font-mono text-bold">${st.G3}</td>
          <td>
            <button class="btn btn-sm btn-ghost btn-inspect" data-student='${JSON.stringify(st).replace(/'/g, "&apos;")}'>
              Inspect JSON
            </button>
          </td>
        </tr>
      `;
    });

    tbody.innerHTML = rows;

    // Attach inspect listeners
    document.querySelectorAll(".btn-inspect").forEach(btn => {
      btn.addEventListener("click", () => {
        const doc = JSON.parse(btn.dataset.student);
        showJsonModal(doc);
      });
    });

  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="11" class="text-center text-danger">Error querying MongoDB: ${err}</td></tr>`;
  }
}

function showJsonModal(doc) {
  const modal = document.getElementById("json-modal");
  const pre = document.getElementById("modal-json-content");
  document.getElementById("modal-title").textContent = `MongoDB Document: ${doc.student_id}`;
  pre.textContent = JSON.stringify(doc, null, 2);
  modal.classList.remove("hidden");
}
