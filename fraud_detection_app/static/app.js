(function () {
  function formatCurrency(value) {
    const number = Number(value || 0);
    return `₹${number.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
  }

  function buildReasonCards(explanations) {
    return (explanations || []).map((item) => `
      <article class="reason-card">
        <strong>${item.label}</strong>
        <p>${item.reason}</p>
      </article>
    `).join('');
  }

  function buildPredictionHtml(data) {
    const alertBlock = data.alert
      ? `<div class="alert-banner">⚠️ High Risk Transaction Detected — risk score is ${data.risk_score}% which is above the ${data.alert_threshold}% alert threshold.</div>`
      : `<div class="info-banner">✅ Transaction looks normal. Risk score is ${data.risk_score}% and stays below the alert threshold.</div>`;

    const scoreCards = `
      <div class="metric-inline-grid">
        <article><span>Risk score</span><strong>${data.risk_score}%</strong></article>
        <article><span>Confidence</span><strong>${data.confidence}%</strong></article>
        <article><span>Best model</span><strong>${data.model_used}</strong></article>
        <article><span>Random Forest / XGBoost</span><strong>${data.all_model_scores['Random Forest']}% / ${data.all_model_scores['XGBoost']}%</strong></article>
      </div>
    `;

    const notes = (data.heuristic_notes || []).map((note) => `<li>${note}</li>`).join('');
    const profile = data.profile || {};

    return `
      <div class="prediction-card">
        ${alertBlock}
        <div class="result-hero">
          <div>
            <h3>${data.prediction === 'Fraud' ? '⚠️ High Risk Transaction Detected' : '✅ Transaction Marked Safe'}</h3>
            <p>The model scored this transaction in real time using user behavior, velocity, amount, and location context.</p>
          </div>
          <div class="result-badge">
            <span>Decision</span>
            <strong>${data.prediction}</strong>
          </div>
        </div>
        ${scoreCards}
        <section>
          <h3>Explainable AI reasons</h3>
          <div class="reason-list">${buildReasonCards(data.explanations)}</div>
        </section>
        <section>
          <h3>Smart risk notes</h3>
          <ul class="bullet-list">${notes}</ul>
        </section>
        <section class="behavior-grid">
          <div><span>Usual spending baseline</span><strong>${formatCurrency(profile.user_avg_amount)}</strong></div>
          <div><span>Transactions in last 24 hours</span><strong>${profile.num_tx_24h || 1}</strong></div>
          <div><span>Distance from last location</span><strong>${Number(profile.distance_from_last_km || 0).toFixed(1)} km</strong></div>
          <div><span>Travel velocity</span><strong>${Number(profile.geovelocity_kmph || 0).toFixed(1)} km/h</strong></div>
        </section>
      </div>
    `;
  }

  function updateSummary(summary) {
    if (!summary) return;
    const total = document.getElementById('summary-total');
    const fraud = document.getElementById('summary-fraud');
    const safe = document.getElementById('summary-safe');
    const percent = document.getElementById('summary-percent');
    if (total) total.textContent = summary.total;
    if (fraud) fraud.textContent = summary.fraud_count;
    if (safe) safe.textContent = summary.safe_count;
    if (percent) percent.textContent = `${summary.fraud_percentage}%`;
  }

  function updateRecentTable(rows) {
    const body = document.querySelector('#recent-transaction-table tbody');
    if (!body || !rows) return;
    body.innerHTML = rows.map((row) => `
      <tr>
        <td>${String(row.created_at).replace('T', ' ').slice(0, 19)}</td>
        <td>${formatCurrency(row.amount)}</td>
        <td><span class="pill ${row.prediction === 'Fraud' ? 'pill-danger' : 'pill-success'}">${row.prediction}</span></td>
        <td>${row.risk_score}%</td>
      </tr>
    `).join('');
  }

  async function handlePredictForm() {
    const form = document.getElementById('predict-form');
    if (!form) return;
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const formData = new FormData(form);
      if (!formData.get('timestamp')) {
        const now = new Date();
        const local = new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
        formData.set('timestamp', `${local}:00`);
      } else if (String(formData.get('timestamp')).length === 16) {
        formData.set('timestamp', `${formData.get('timestamp')}:00`);
      }
      const payload = Object.fromEntries(formData.entries());

      const button = form.querySelector('button[type="submit"]');
      const oldText = button.textContent;
      button.disabled = true;
      button.textContent = 'Scoring transaction...';

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Prediction failed');

        const placeholder = document.getElementById('prediction-placeholder');
        const result = document.getElementById('prediction-result');
        if (placeholder) placeholder.classList.add('hidden');
        result.classList.remove('hidden');
        result.innerHTML = buildPredictionHtml(data);
        updateSummary(data.summary);
        updateRecentTable(data.recent_transactions);
      } catch (error) {
        alert(error.message);
      } finally {
        button.disabled = false;
        button.textContent = oldText;
      }
    });
  }

  function handleHistorySearch() {
    const search = document.getElementById('historySearch');
    const table = document.getElementById('historyTable');
    if (!search || !table) return;
    search.addEventListener('input', () => {
      const term = search.value.toLowerCase();
      table.querySelectorAll('tbody tr').forEach((row) => {
        row.style.display = row.textContent.toLowerCase().includes(term) ? '' : 'none';
      });
    });
  }

  window.addEventListener('load', () => {
    handlePredictForm();
    handleHistorySearch();
  });
})();
