document.addEventListener('DOMContentLoaded', () => {
    // 1. Initial Data Fetching
    fetchMetrics();
    fetchSubreddits();

    // 2. Metrics Logic
    async function fetchMetrics() {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/metrics');
            const data = await response.json();
            populateMetricsUI(data);
            createBenchmarkChart(data);
        } catch (e) {
            console.error("Failed to fetch metrics", e);
        }
    }

    function populateMetricsUI(data) {
        const hgnn = data["Proposed HGNN"] || data["Proposed HGNN Model"];
        if (hgnn) {
            document.getElementById('auc-val').innerText = hgnn.AUC.toFixed(4);
            document.getElementById('mrr-val').innerText = hgnn.MRR.toFixed(4);
            document.getElementById('hits-val').innerText = hgnn['Hits@10'].toFixed(4);
        }

        const tbody = document.getElementById('metrics-body');
        tbody.innerHTML = '';
        Object.entries(data).forEach(([model, metrics]) => {
            const tr = document.createElement('tr');
            if (model.includes("Proposed") || model.includes("HGNN")) tr.className = "highlight";
            tr.innerHTML = `
                <td>${model}</td>
                <td>${metrics.AUC.toFixed(4)}</td>
                <td>${metrics.MRR.toFixed(4)}</td>
                <td>${metrics['Hits@10'].toFixed(4)}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // 3. Subreddits Logic
    async function fetchSubreddits() {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/subreddits');
            const subs = await response.json();
            const src = document.getElementById('source-opts');
            const tgt = document.getElementById('target-opts');
            src.innerHTML = ''; tgt.innerHTML = '';
            subs.forEach(s => {
                const opt = `<option value="${s}">r/${s}</option>`;
                src.innerHTML += opt; tgt.innerHTML += opt;
            });
        } catch (e) {
            console.error("Failed to fetch subreddits", e);
        }
    }

    // 4. Prediction Logic
    const predictionForm = document.getElementById('predictionForm');
    const resultArea = document.getElementById('predictionResult');

    if (predictionForm) {
        predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const source = document.getElementById('source').value;
            const target = document.getElementById('target').value;

            resultArea.innerHTML = `<div class="loader"></div>`;

            try {
                const response = await fetch('http://127.0.0.1:5000/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source, target })
                });
                const data = await response.json();

                setTimeout(() => {
                    showPredictionResult(data);
                    window.createTemporalGraph(data.graph_data);
                    window.generateHeatmap(data.similarity);
                    document.getElementById('liveConfidence').innerText = Math.round(data.confidence * 100) + "%";
                }, 1000);
            } catch (error) {
                resultArea.innerHTML = `<p style="color:red">Connection Failed</p>`;
            }
        });
    }

    function showPredictionResult(data) {
        resultArea.innerHTML = `
            <div class="animate-up visible">
                <h2 style="color:var(--secondary); letter-spacing: 2px; font-size: 0.9rem;">INFERENCE SCORE</h2>
                <h1>${data.probability.toFixed(1)}%</h1>
                <p style="color:var(--text-dim); margin-top: 15px;">Target: r/${data.target}</p>
            </div>
        `;
    }

    // --- D3 Visualizations ---

    function createBenchmarkChart(data) {
        d3.select("#benchmark-chart").html("");
        const container = document.getElementById('benchmark-chart');
        const width = container.clientWidth;
        const height = 350;
        const margin = { top: 20, right: 30, bottom: 40, left: 100 };

        const svg = d3.select("#benchmark-chart").append("svg")
            .attr("width", width).attr("height", height)
            .append("g").attr("transform", `translate(${margin.left},${margin.top})`);

        const dataset = Object.entries(data).map(([name, metrics]) => ({ name, auc: metrics.AUC }));

        const x = d3.scaleLinear().domain([0, 1]).range([0, width - margin.left - margin.right]);
        const y = d3.scaleBand().domain(dataset.map(d => d.name)).range([0, height - margin.top - margin.bottom]).padding(0.3);

        svg.append("g").call(d3.axisLeft(y).tickSize(0)).selectAll("text").style("color", "#fff").style("font-size", "10px");
        svg.append("g").attr("transform", `translate(0,${height - margin.top - margin.bottom})`).call(d3.axisBottom(x).ticks(5)).style("color", "#94a3b8");

        svg.selectAll("rect")
            .data(dataset).enter().append("rect")
            .attr("y", d => y(d.name))
            .attr("height", y.bandwidth())
            .attr("fill", d => d.name.includes("HGNN") ? "#c084fc" : "#334155")
            .attr("rx", 5)
            .attr("width", 0)
            .transition().duration(1500).attr("width", d => x(d.auc));
    }

    window.createTemporalGraph = function(values) {
        d3.select("#d3-chart").html("");
        const container = document.getElementById('d3-chart');
        const width = container.clientWidth;
        const height = 350;
        const svg = d3.select("#d3-chart").append("svg").attr("width", width).attr("height", height);
        const x = d3.scaleLinear().domain([0, values.length - 1]).range([50, width - 50]);
        const y = d3.scaleLinear().domain([0, 100]).range([height - 50, 50]);
        const line = d3.line().x((d, i) => x(i)).y(d => y(d)).curve(d3.curveMonotoneX);
        svg.append("path").datum(values).attr("fill", "none").attr("stroke", "#c084fc").attr("stroke-width", 5).attr("d", line);
    };

    window.generateHeatmap = function(matrix) {
        d3.select('#heatmap').html('');
        const container = document.getElementById('heatmap');
        const size = Math.min(container.clientWidth, 350);
        const cellSize = size / matrix.length;
        const svg = d3.select('#heatmap').append('svg').attr('width', size).attr('height', size);
        matrix.forEach((row, i) => {
            row.forEach((value, j) => {
                svg.append('rect').attr('x', j * cellSize).attr('y', i * cellSize).attr('width', cellSize - 2).attr('height', cellSize - 2).attr('fill', `rgba(192, 132, 252, ${value})`);
            });
        });
    };
});
