(function () {
  function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return ctx;
  }

  function setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    const ctx = canvas.getContext('2d');
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    return { ctx, width: rect.width, height: rect.height };
  }

  function palette(index) {
    const colors = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'];
    return colors[index % colors.length];
  }

  function drawLegend(ctx, labels, values, width, startY) {
    labels.forEach((label, index) => {
      const y = startY + index * 24;
      ctx.fillStyle = palette(index);
      ctx.fillRect(width - 160, y - 10, 12, 12);
      ctx.fillStyle = '#0f172a';
      ctx.font = '12px sans-serif';
      ctx.fillText(`${label}: ${values[index]}`, width - 140, y);
    });
  }

  function drawPieChart(canvas, labels, values) {
    const { ctx, width, height } = setupCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    const total = values.reduce((sum, value) => sum + value, 0) || 1;
    const centerX = width * 0.32;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.24;
    let startAngle = -Math.PI / 2;

    values.forEach((value, index) => {
      const sliceAngle = (value / total) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.arc(centerX, centerY, radius, startAngle, startAngle + sliceAngle);
      ctx.closePath();
      ctx.fillStyle = palette(index);
      ctx.fill();
      startAngle += sliceAngle;
    });

    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 20px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`${total}`, centerX, centerY + 6);
    ctx.font = '12px sans-serif';
    ctx.fillText('Transactions', centerX, centerY + 26);
    ctx.textAlign = 'left';
    drawLegend(ctx, labels, values, width, height * 0.28);
  }

  function drawBarChart(canvas, labels, values) {
    const { ctx, width, height } = setupCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    const padding = 42;
    const chartHeight = height - padding * 2;
    const chartWidth = width - padding * 2;
    const maxValue = Math.max(...values, 1);
    const barWidth = chartWidth / Math.max(labels.length, 1) * 0.6;

    ctx.strokeStyle = '#cbd5e1';
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    values.forEach((value, index) => {
      const x = padding + index * (chartWidth / Math.max(labels.length, 1)) + 18;
      const barHeight = (value / maxValue) * (chartHeight - 20);
      const y = height - padding - barHeight;
      ctx.fillStyle = palette(index);
      ctx.fillRect(x, y, barWidth, barHeight);
      ctx.fillStyle = '#334155';
      ctx.font = '11px sans-serif';
      ctx.fillText(value, x + 4, y - 8);
      ctx.save();
      ctx.translate(x + 4, height - padding + 14);
      ctx.rotate(Math.PI / 9);
      ctx.fillText(labels[index], 0, 0);
      ctx.restore();
    });
  }

  function drawLineChart(canvas, labels, values) {
    const { ctx, width, height } = setupCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    const padding = 42;
    const chartHeight = height - padding * 2;
    const chartWidth = width - padding * 2;
    const maxValue = Math.max(...values, 1);

    ctx.strokeStyle = '#cbd5e1';
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    ctx.beginPath();
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 2;
    values.forEach((value, index) => {
      const x = padding + (index / Math.max(values.length - 1, 1)) * chartWidth;
      const y = height - padding - (value / maxValue) * (chartHeight - 12);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    values.forEach((value, index) => {
      const x = padding + (index / Math.max(values.length - 1, 1)) * chartWidth;
      const y = height - padding - (value / maxValue) * (chartHeight - 12);
      ctx.fillStyle = '#2563eb';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#334155';
      ctx.font = '11px sans-serif';
      ctx.fillText(labels[index] || '', x - 16, height - padding + 16);
    });
  }

  function renderCharts() {
    document.querySelectorAll('.chart-canvas').forEach((canvas) => {
      const config = canvas.dataset.chart;
      if (!config) return;
      const parsed = JSON.parse(config);
      const labels = parsed.labels || [];
      const values = parsed.values || [];
      if (parsed.type === 'pie') drawPieChart(canvas, labels, values);
      if (parsed.type === 'bar') drawBarChart(canvas, labels, values);
      if (parsed.type === 'line') drawLineChart(canvas, labels, values);
    });
  }

  window.addEventListener('load', renderCharts);
  window.addEventListener('resize', renderCharts);
  window.renderSimpleCharts = renderCharts;
})();
