/**
 * Benchmark visualization - HTML report generation
 */

import * as fs from 'fs';
import * as path from 'path';
import type { BenchmarkReport, BenchmarkComparison } from './types';
import { groupByCategory, getCategorySummaries, formatDuration, formatRatio } from './analysis';

export function generateHTMLReport(report: BenchmarkReport, outputPath: string): void {
  const html = createHTML(report);
  fs.writeFileSync(outputPath, html, 'utf-8');
}

function createHTML(report: BenchmarkReport): string {
  const { timestamp, environment, results, summary } = report;
  const groups = groupByCategory(results);
  const categorySummaries = getCategorySummaries(results);

  // Prepare data for charts
  const categories = Array.from(groups.keys());
  const categoryAvgSlowdowns = categories.map(
    (cat) => categorySummaries.get(cat)!.avg_slowdown
  );

  // All benchmark names and ratios for detailed chart
  const benchmarkNames = results.map((r) => r.name);
  const benchmarkRatios = results.map((r) => r.ratio);

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NumPy vs numpy-ts Benchmark Results</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.6;
      color: #333;
      background: #f5f5f5;
      padding: 20px;
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
      background: white;
      padding: 30px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    h1 {
      color: #2c3e50;
      margin-bottom: 10px;
      font-size: 2.5em;
    }

    .subtitle {
      color: #7f8c8d;
      margin-bottom: 30px;
      font-size: 1.1em;
    }

    .meta {
      background: #ecf0f1;
      padding: 15px;
      border-radius: 5px;
      margin-bottom: 30px;
      font-family: monospace;
      font-size: 0.9em;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin-bottom: 40px;
    }

    .summary-card {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 20px;
      border-radius: 8px;
      text-align: center;
    }

    .summary-card.best {
      background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }

    .summary-card.worst {
      background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
    }

    .summary-card h3 {
      font-size: 0.9em;
      opacity: 0.9;
      margin-bottom: 10px;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .summary-card .value {
      font-size: 2.5em;
      font-weight: bold;
    }

    .chart-container {
      margin: 40px 0;
      padding: 20px;
      background: white;
      border-radius: 8px;
      border: 1px solid #e0e0e0;
    }

    .chart-container h2 {
      margin-bottom: 20px;
      color: #2c3e50;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
    }

    th, td {
      padding: 12px;
      text-align: left;
      border-bottom: 1px solid #e0e0e0;
    }

    th {
      background: #34495e;
      color: white;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.85em;
      letter-spacing: 0.5px;
    }

    tr:hover {
      background: #f8f9fa;
    }

    .ratio {
      font-weight: bold;
      padding: 4px 8px;
      border-radius: 4px;
    }

    .ratio.good {
      background: #d4edda;
      color: #155724;
    }

    .ratio.ok {
      background: #fff3cd;
      color: #856404;
    }

    .ratio.bad {
      background: #f8d7da;
      color: #721c24;
    }

    .category-section {
      margin: 40px 0;
    }

    .category-section h2 {
      color: #2c3e50;
      border-bottom: 3px solid #3498db;
      padding-bottom: 10px;
      margin-bottom: 20px;
    }

    .footer {
      margin-top: 40px;
      padding-top: 20px;
      border-top: 2px solid #e0e0e0;
      text-align: center;
      color: #7f8c8d;
      font-size: 0.9em;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🚀 NumPy vs numpy-ts Benchmark Results</h1>
    <p class="subtitle">Performance comparison of numpy-ts against Python NumPy</p>

    <div class="meta">
      <div><strong>Timestamp:</strong> ${new Date(timestamp).toLocaleString()}</div>
      <div><strong>Node:</strong> ${environment.node_version}</div>
      ${environment.python_version ? `<div><strong>Python:</strong> ${environment.python_version}</div>` : ''}
      ${environment.numpy_version ? `<div><strong>NumPy:</strong> ${environment.numpy_version}</div>` : ''}
      <div><strong>numpy-ts:</strong> ${environment.numpyjs_version}</div>
      <div><strong>Total Benchmarks:</strong> ${summary.total_benchmarks}</div>
    </div>

    <div class="summary-grid">
      <div class="summary-card">
        <h3>Average Slowdown</h3>
        <div class="value">${formatRatio(summary.avg_slowdown)}</div>
      </div>
      <div class="summary-card">
        <h3>Median Slowdown</h3>
        <div class="value">${formatRatio(summary.median_slowdown)}</div>
      </div>
      <div class="summary-card best">
        <h3>Best Case</h3>
        <div class="value">${formatRatio(summary.best_case)}</div>
      </div>
      <div class="summary-card worst">
        <h3>Worst Case</h3>
        <div class="value">${formatRatio(summary.worst_case)}</div>
      </div>
    </div>

    <div class="chart-container">
      <h2>📊 Average Slowdown by Category</h2>
      <canvas id="categoryChart"></canvas>
    </div>

    <div class="chart-container">
      <h2>📈 Detailed Results (All Benchmarks)</h2>
      <canvas id="detailedChart"></canvas>
    </div>

    ${generateCategoryTables(groups)}

    <div class="footer">
      <p>Generated by numpy-ts Benchmark Suite</p>
      <p>Lower ratios are better (closer to NumPy performance)</p>
    </div>
  </div>

  <script>
    // Category chart
    new Chart(document.getElementById('categoryChart'), {
      type: 'bar',
      data: {
        labels: ${JSON.stringify(categories)},
        datasets: [{
          label: 'Average Slowdown (x times slower than NumPy)',
          data: ${JSON.stringify(categoryAvgSlowdowns)},
          backgroundColor: 'rgba(102, 126, 234, 0.8)',
          borderColor: 'rgba(102, 126, 234, 1)',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            display: true
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Slowdown Ratio'
            }
          }
        }
      }
    });

    // Detailed chart
    new Chart(document.getElementById('detailedChart'), {
      type: 'bar',
      data: {
        labels: ${JSON.stringify(benchmarkNames)},
        datasets: [{
          label: 'Slowdown Ratio (x times slower)',
          data: ${JSON.stringify(benchmarkRatios)},
          backgroundColor: ${JSON.stringify(benchmarkRatios.map((r: number) =>
            r < 2 ? 'rgba(46, 213, 115, 0.8)' : r < 5 ? 'rgba(255, 195, 18, 0.8)' : 'rgba(235, 77, 75, 0.8)'
          ))},
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        indexAxis: 'y',
        plugins: {
          legend: {
            display: false
          }
        },
        scales: {
          x: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Slowdown Ratio'
            }
          }
        }
      }
    });
  </script>
</body>
</html>`;
}

function generateCategoryTables(groups: Map<string, BenchmarkComparison[]>): string {
  let html = '';

  for (const [category, items] of groups) {
    html += `
    <div class="category-section">
      <h2>${category.toUpperCase()}</h2>
      <table>
        <thead>
          <tr>
            <th>Benchmark</th>
            <th>NumPy (ms)</th>
            <th>numpy-ts (ms)</th>
            <th>Ratio</th>
          </tr>
        </thead>
        <tbody>`;

    for (const item of items) {
      const ratioClass = item.ratio < 2 ? 'good' : item.ratio < 5 ? 'ok' : 'bad';
      html += `
          <tr>
            <td>${item.name}</td>
            <td>${formatDuration(item.numpy.mean_ms)}</td>
            <td>${formatDuration(item.numpyjs.mean_ms)}</td>
            <td><span class="ratio ${ratioClass}">${formatRatio(item.ratio)}</span></td>
          </tr>`;
    }

    html += `
        </tbody>
      </table>
    </div>`;
  }

  return html;
}
