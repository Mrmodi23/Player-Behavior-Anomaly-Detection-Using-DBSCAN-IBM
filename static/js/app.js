/**
 * app.js
 * Frontend controller for:
 * Player Behavior Anomaly Detection Using DBSCAN
 * Vanilla JavaScript (No React required)
 */

document.addEventListener('DOMContentLoaded', () => {
    // Application State
    const state = {
        currentPage: 'dashboard',
        selectedPlayerId: null,
        playersList: [],
        dashboardData: null,
        mapData: null,
        resultsQuery: {
            page: 1,
            pageSize: 50,
            status: 'ALL',
            playerId: 'ALL',
            search: '',
            sortBy: 'timestamp',
            sortOrder: 'desc'
        },
        selectedFile: null,
        chartInstances: {}
    };

    // Color Palette for Clusters
    const clusterPalette = [
        '#06b6d4', '#8b5cf6', '#10b981', '#3b82f6', '#f59e0b', 
        '#ec4899', '#14b8a6', '#6366f1', '#eab308', '#a855f7',
        '#0ea5e9', '#84cc16', '#f97316', '#22c55e', '#d946ef'
    ];
    const NOISE_COLOR = '#f43f5e';

    // UI Element References
    const elements = {
        navButtons: document.querySelectorAll('.nav-item'),
        pageViews: document.querySelectorAll('.page-view'),
        sidebar: document.getElementById('sidebar'),
        sidebarToggle: document.getElementById('sidebar-toggle'),
        btnRefreshAll: document.getElementById('btn-refresh-all'),
        toastContainer: document.getElementById('toast-container'),

        // Dashboard
        cardTotalPlayers: document.getElementById('card-total-players'),
        cardTotalRecords: document.getElementById('card-total-records'),
        cardNormalRecords: document.getElementById('card-normal-records'),
        cardAnomalousRecords: document.getElementById('card-anomalous-records'),
        cardNumClusters: document.getElementById('card-num-clusters'),
        cardAnomalyPct: document.getElementById('card-anomaly-pct'),
        btnViewFullMap: document.getElementById('btn-view-full-map'),

        // Player Analysis
        playerSelect: document.getElementById('player-select'),
        plId: document.getElementById('pl-id'),
        plTotalMovements: document.getElementById('pl-total-movements'),
        plAvgSpeed: document.getElementById('pl-avg-speed'),
        plMaxSpeed: document.getElementById('pl-max-speed'),
        plAvgDir: document.getElementById('pl-avg-dir'),
        plAnomalousMovements: document.getElementById('pl-anomalous-movements'),
        plAnomalyPct: document.getElementById('pl-anomaly-pct'),
        badgeSuspicionVal: document.getElementById('badge-suspicion-val'),
        playerTelemetryTbody: document.getElementById('player-telemetry-tbody'),

        // Movement Map
        btnMapFilterAll: document.getElementById('btn-map-filter-all'),
        btnMapFilterAnomalies: document.getElementById('btn-map-filter-anomalies'),
        btnResetMapZoom: document.getElementById('btn-reset-map-zoom'),
        mapCustomLegend: document.getElementById('map-custom-legend'),

        // Run Detection
        detectionForm: document.getElementById('detection-form'),
        paramEps: document.getElementById('param-eps'),
        paramMinSamples: document.getElementById('param-min-samples'),
        csvFileInput: document.getElementById('csv-file-input'),
        fileDropzone: document.getElementById('file-dropzone'),
        selectedFileName: document.getElementById('selected-file-name'),
        btnRunDbscan: document.getElementById('btn-run-dbscan'),

        // Results Table
        resTotal: document.getElementById('res-total'),
        resNormal: document.getElementById('res-normal'),
        resAnomalous: document.getElementById('res-anomalous'),
        resClusters: document.getElementById('res-clusters'),
        resPct: document.getElementById('res-pct'),
        resultsSearchInput: document.getElementById('results-search-input'),
        filterStatus: document.getElementById('filter-status'),
        filterPlayer: document.getElementById('filter-player'),
        filterPageSize: document.getElementById('filter-page-size'),
        resultsTableBody: document.getElementById('results-table-body'),
        paginationInfoText: document.getElementById('pagination-info-text'),
        currentPageDisplay: document.getElementById('current-page-display'),
        btnPagePrev: document.getElementById('btn-page-prev'),
        btnPageNext: document.getElementById('btn-page-next'),
        tableSortHeaders: document.querySelectorAll('#main-results-table th.sortable'),

        // Evaluation Page
        evalClusters: document.getElementById('eval-clusters'),
        evalNoise: document.getElementById('eval-noise'),
        evalPct: document.getElementById('eval-pct'),
        evalEps: document.getElementById('eval-eps'),
        evalMinSamples: document.getElementById('eval-min-samples'),
        evalClusterBreakdown: document.getElementById('eval-cluster-breakdown')
    };

    // -------------------------------------------------------------
    // Helper: Toast Notifications
    // -------------------------------------------------------------
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        let icon = 'fa-circle-info';
        if (type === 'success') icon = 'fa-circle-check';
        if (type === 'error') icon = 'fa-triangle-exclamation';

        toast.innerHTML = `
            <i class="fa-solid ${icon}"></i>
            <span>${message}</span>
        `;
        elements.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(20px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3800);
    }

    // -------------------------------------------------------------
    // Navigation Routing
    // -------------------------------------------------------------
    function navigateToPage(pageId) {
        state.currentPage = pageId;
        elements.pageViews.forEach(view => {
            view.classList.toggle('active', view.id === `page-${pageId}`);
        });
        elements.navButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.page === pageId);
        });

        // Trigger on-demand view loads
        if (pageId === 'player-analysis') {
            loadPlayerAnalysis(state.selectedPlayerId);
        } else if (pageId === 'coordinate-map') {
            loadCoordinateMap();
        } else if (pageId === 'results-table') {
            loadResultsTable();
        } else if (pageId === 'theory-evaluation') {
            updateEvaluationStats();
        }

        // Close mobile sidebar if open
        if (elements.sidebar.classList.contains('open')) {
            elements.sidebar.classList.remove('open');
        }
    }

    elements.navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            navigateToPage(btn.dataset.page);
        });
    });

    if (elements.sidebarToggle) {
        elements.sidebarToggle.addEventListener('click', () => {
            elements.sidebar.classList.toggle('open');
        });
    }

    if (elements.btnViewFullMap) {
        elements.btnViewFullMap.addEventListener('click', () => {
            navigateToPage('coordinate-map');
        });
    }

    if (elements.btnRefreshAll) {
        elements.btnRefreshAll.addEventListener('click', () => {
            showToast('Refreshing telemetry and model states...', 'info');
            loadDashboard();
            if (state.currentPage === 'player-analysis') loadPlayerAnalysis(state.selectedPlayerId);
            if (state.currentPage === 'coordinate-map') loadCoordinateMap();
            if (state.currentPage === 'results-table') loadResultsTable();
        });
    }

    // -------------------------------------------------------------
    // Chart Cleanup Helper
    // -------------------------------------------------------------
    function destroyChart(id) {
        if (state.chartInstances[id]) {
            state.chartInstances[id].destroy();
            delete state.chartInstances[id];
        }
    }

    // -------------------------------------------------------------
    // 1. DASHBOARD CONTROLLER
    // -------------------------------------------------------------
    async function loadDashboard() {
        try {
            const res = await fetch('/api/dashboard');
            const data = await res.json();
            if (!data.success) throw new Error(data.error || 'Failed to fetch dashboard data');

            state.dashboardData = data;
            const stats = data.stats;

            // Update Metric Cards
            elements.cardTotalPlayers.textContent = stats.total_players.toLocaleString();
            elements.cardTotalRecords.textContent = stats.total_records.toLocaleString();
            elements.cardNormalRecords.textContent = stats.normal_records.toLocaleString();
            elements.cardAnomalousRecords.textContent = stats.anomalous_records.toLocaleString();
            elements.cardNumClusters.textContent = stats.num_clusters.toLocaleString();
            elements.cardAnomalyPct.textContent = `${stats.anomaly_percentage}%`;

            // Update Evaluation Page numbers as well
            updateEvaluationStats();

            // Render Dashboard Charts
            renderNormalVsAnomaliesChart(stats.normal_records, stats.anomalous_records);
            renderClusterDistributionChart(data.cluster_distribution);
            renderActivityTimelineChart(data.time_series);
            renderCoordinatesPreviewChart(data.sample_coordinates);

            // Populate player select lists
            await loadPlayerDropdownList();

        } catch (err) {
            console.error(err);
            showToast(`Dashboard error: ${err.message}`, 'error');
        }
    }

    function renderNormalVsAnomaliesChart(normalCount, anomalyCount) {
        destroyChart('chart-normal-vs-anomalies');
        const ctx = document.getElementById('chart-normal-vs-anomalies');
        if (!ctx) return;

        state.chartInstances['chart-normal-vs-anomalies'] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Normal Movement', 'Anomalous / Noise'],
                datasets: [{
                    data: [normalCount, anomalyCount],
                    backgroundColor: ['#10b981', '#f43f5e'],
                    borderColor: '#0d1322',
                    borderWidth: 3,
                    hoverOffset: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#94a3b8',
                            font: { family: 'Inter', size: 12 },
                            padding: 16
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const total = normalCount + anomalyCount;
                                const val = ctx.raw || 0;
                                const pct = total > 0 ? ((val / total) * 100).toFixed(1) : 0;
                                return ` ${ctx.label}: ${val.toLocaleString()} (${pct}%)`;
                            }
                        }
                    }
                },
                cutout: '72%'
            }
        });
    }

    function renderClusterDistributionChart(clusterDist) {
        destroyChart('chart-cluster-distribution');
        const ctx = document.getElementById('chart-cluster-distribution');
        if (!ctx || !clusterDist) return;

        const labels = clusterDist.map(c => c.label);
        const dataValues = clusterDist.map(c => c.count);
        const bgColors = clusterDist.map(c => c.cluster === -1 ? NOISE_COLOR : (clusterPalette[c.cluster % clusterPalette.length]));

        state.chartInstances['chart-cluster-distribution'] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Records',
                    data: dataValues,
                    backgroundColor: bgColors,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ` Movement Points: ${ctx.raw.toLocaleString()}`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b', font: { family: 'Inter', size: 10 } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b' }
                    }
                }
            }
        });
    }

    function renderActivityTimelineChart(timeSeries) {
        destroyChart('chart-activity-timeline');
        const ctx = document.getElementById('chart-activity-timeline');
        if (!ctx || !timeSeries) return;

        const labels = timeSeries.map(t => t.time_label);
        const normalData = timeSeries.map(t => t.normal);
        const anomalyData = timeSeries.map(t => t.anomalies);

        state.chartInstances['chart-activity-timeline'] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Normal Movements',
                        data: normalData,
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        fill: true,
                        tension: 0.35,
                        pointRadius: 2
                    },
                    {
                        label: 'Anomalies (Noise)',
                        data: anomalyData,
                        borderColor: '#f43f5e',
                        backgroundColor: 'rgba(244, 63, 94, 0.2)',
                        fill: true,
                        tension: 0.35,
                        pointRadius: 4,
                        pointBackgroundColor: '#f43f5e'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b', font: { size: 10 } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b' }
                    }
                }
            }
        });
    }

    function renderCoordinatesPreviewChart(samples) {
        destroyChart('chart-coordinates-preview');
        const ctx = document.getElementById('chart-coordinates-preview');
        if (!ctx || !samples) return;

        const normalPoints = samples.filter(p => p.cluster !== -1).map(p => ({ x: p.x, y: p.y }));
        const anomalyPoints = samples.filter(p => p.cluster === -1).map(p => ({ x: p.x, y: p.y }));

        state.chartInstances['chart-coordinates-preview'] = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Normal Clustered Coordinates',
                        data: normalPoints,
                        backgroundColor: 'rgba(6, 182, 212, 0.6)',
                        borderColor: 'transparent',
                        pointRadius: 3
                    },
                    {
                        label: 'Anomalous Coordinates (Noise)',
                        data: anomalyPoints,
                        backgroundColor: '#f43f5e',
                        borderColor: '#ffffff',
                        borderWidth: 1,
                        pointRadius: 5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#94a3b8', font: { size: 11 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ` (X: ${ctx.parsed.x}, Y: ${ctx.parsed.y})`
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Player X Coordinate', color: '#64748b' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b' }
                    },
                    y: {
                        title: { display: true, text: 'Player Y Coordinate', color: '#64748b' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b' }
                    }
                }
            }
        });
    }

    // -------------------------------------------------------------
    // 2. PLAYER ANALYSIS CONTROLLER
    // -------------------------------------------------------------
    async function loadPlayerDropdownList() {
        try {
            const res = await fetch('/api/players');
            const data = await res.json();
            if (!data.success) return;

            state.playersList = data.players;
            
            // Populate Dropdowns
            elements.playerSelect.innerHTML = '';
            elements.filterPlayer.innerHTML = '<option value="ALL">All Players</option>';

            data.players.forEach((p, idx) => {
                const opt = document.createElement('option');
                opt.value = p.player_id;
                opt.textContent = `${p.player_id} (${p.anomaly_percentage}% Anomaly - ${p.suspicion_level})`;
                elements.playerSelect.appendChild(opt);

                const filterOpt = document.createElement('option');
                filterOpt.value = p.player_id;
                filterOpt.textContent = p.player_id;
                elements.filterPlayer.appendChild(filterOpt);

                if (idx === 0 && !state.selectedPlayerId) {
                    state.selectedPlayerId = p.player_id;
                }
            });

            if (state.selectedPlayerId) {
                elements.playerSelect.value = state.selectedPlayerId;
            }
        } catch (err) {
            console.error('Error fetching players:', err);
        }
    }

    elements.playerSelect.addEventListener('change', (e) => {
        state.selectedPlayerId = e.target.value;
        loadPlayerAnalysis(state.selectedPlayerId);
    });

    async function loadPlayerAnalysis(playerId) {
        if (!playerId) return;

        try {
            const res = await fetch(`/api/player/${encodeURIComponent(playerId)}`);
            const data = await res.json();
            if (!data.success) throw new Error(data.error || 'Failed to load player details');

            const p = data.player;
            elements.plId.textContent = p.player_id;
            elements.plTotalMovements.textContent = p.total_movements.toLocaleString();
            elements.plAvgSpeed.innerHTML = `${p.avg_speed} <span class="unit">m/s</span>`;
            elements.plMaxSpeed.innerHTML = `${p.max_speed} <span class="unit">m/s</span>`;
            elements.plAvgDir.innerHTML = `${p.avg_direction_change}<span class="unit">°</span>`;
            elements.plAnomalousMovements.textContent = p.anomalous_movements.toLocaleString();
            elements.plAnomalyPct.textContent = `${p.anomaly_percentage}%`;

            // Suspicion Badge
            elements.badgeSuspicionVal.textContent = `${p.suspicion_level} SUSPICION`;
            elements.badgeSuspicionVal.className = 'badge-value';
            if (p.suspicion_level === 'HIGH') elements.badgeSuspicionVal.classList.add('badge-high');
            else if (p.suspicion_level === 'MEDIUM') elements.badgeSuspicionVal.classList.add('badge-medium');
            else elements.badgeSuspicionVal.classList.add('badge-low');

            // Render Trajectory Path Chart
            renderPlayerPathChart(data.trajectory);

            // Populate sample table rows
            renderPlayerTelemetryTable(data.trajectory);

        } catch (err) {
            console.error(err);
            showToast(`Player detail error: ${err.message}`, 'error');
        }
    }

    function renderPlayerPathChart(trajectory) {
        destroyChart('chart-player-path');
        const ctx = document.getElementById('chart-player-path');
        if (!ctx || !trajectory) return;

        // Path line dataset (all points in sequence)
        const pathLineData = trajectory.map(pt => ({ x: pt.x, y: pt.y }));
        
        // Anomalies dataset (isolated points)
        const anomaliesData = trajectory.filter(pt => pt.is_anomaly).map(pt => ({
            x: pt.x,
            y: pt.y,
            speed: pt.speed,
            direction: pt.direction,
            timestamp: pt.timestamp
        }));

        state.chartInstances['chart-player-path'] = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        type: 'line',
                        label: 'Movement Trajectory Path',
                        data: pathLineData,
                        borderColor: 'rgba(6, 182, 212, 0.45)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.1,
                        pointRadius: 3,
                        pointBackgroundColor: 'rgba(6, 182, 212, 0.8)'
                    },
                    {
                        type: 'scatter',
                        label: 'Detected Anomaly Coordinates (DBSCAN Noise)',
                        data: anomaliesData,
                        backgroundColor: '#f43f5e',
                        borderColor: '#ffffff',
                        borderWidth: 2,
                        pointRadius: 8,
                        pointHoverRadius: 10
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const p = ctx.raw;
                                if (ctx.datasetIndex === 1) {
                                    return ` ANOMALY! Pos: (${p.x}, ${p.y}) | Speed: ${p.speed} m/s | Angle: ${p.direction}°`;
                                }
                                return ` Pos: (${p.x}, ${p.y})`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'X Coordinate (In-Game Units)', color: '#64748b' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b' }
                    },
                    y: {
                        title: { display: true, text: 'Y Coordinate (In-Game Units)', color: '#64748b' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b' }
                    }
                }
            }
        });
    }

    function renderPlayerTelemetryTable(trajectory) {
        elements.playerTelemetryTbody.innerHTML = '';
        const previewRows = trajectory.slice(-15); // Show latest 15 ticks

        previewRows.forEach(row => {
            const tr = document.createElement('tr');
            const isAnom = row.is_anomaly;
            tr.innerHTML = `
                <td>${row.timestamp}</td>
                <td>${row.x.toFixed(2)}</td>
                <td>${row.y.toFixed(2)}</td>
                <td class="${row.speed > 30 ? 'text-rose font-semibold' : ''}">${row.speed.toFixed(2)}</td>
                <td>${row.direction.toFixed(1)}°</td>
                <td>${row.cluster === -1 ? '<span class="text-rose font-bold">-1 (Noise)</span>' : 'Cluster ' + row.cluster}</td>
                <td>
                    <span class="status-badge ${isAnom ? 'anomaly' : 'normal'}">
                        ${isAnom ? 'ANOMALY' : 'NORMAL'}
                    </span>
                </td>
            `;
            elements.playerTelemetryTbody.appendChild(tr);
        });
    }

    // -------------------------------------------------------------
    // 3. MOVEMENT VISUALIZATION PAGE (FULL 2D INTERACTIVE MAP)
    // -------------------------------------------------------------
    async function loadCoordinateMap() {
        try {
            const res = await fetch('/api/coordinates');
            const data = await res.json();
            if (!data.success) throw new Error(data.error || 'Failed to load coordinates');

            state.mapData = data.clusters;
            renderFullMapChart(data.clusters);
            buildCustomMapLegend(data.clusters);

        } catch (err) {
            console.error(err);
            showToast(`Map loading error: ${err.message}`, 'error');
        }
    }

    function buildCustomMapLegend(clusters) {
        elements.mapCustomLegend.innerHTML = '';

        // Add anomaly pill
        if (clusters['-1']) {
            const count = clusters['-1'].length;
            const badge = document.createElement('span');
            badge.className = 'legend-badge anomaly-legend';
            badge.innerHTML = `<span class="legend-color-dot" style="background:${NOISE_COLOR}"></span> Noise / Anomaly (${count})`;
            badge.addEventListener('click', () => filterMapToCluster(-1));
            elements.mapCustomLegend.appendChild(badge);
        }

        // Add normal cluster pills
        Object.keys(clusters).sort((a, b) => parseInt(a) - parseInt(b)).forEach(cKey => {
            const c = parseInt(cKey);
            if (c === -1) return;
            const count = clusters[cKey].length;
            const color = clusterPalette[c % clusterPalette.length];
            const badge = document.createElement('span');
            badge.className = 'legend-badge';
            badge.innerHTML = `<span class="legend-color-dot" style="background:${color}"></span> Cluster ${c} (${count})`;
            badge.addEventListener('click', () => filterMapToCluster(c));
            elements.mapCustomLegend.appendChild(badge);
        });
    }

    function renderFullMapChart(clusters, visibleClusters = null) {
        destroyChart('chart-full-map');
        const ctx = document.getElementById('chart-full-map');
        if (!ctx || !clusters) return;

        const datasets = [];

        // Build datasets for each cluster
        Object.keys(clusters).sort((a, b) => parseInt(a) - parseInt(b)).forEach(cKey => {
            const c = parseInt(cKey);
            if (visibleClusters && !visibleClusters.includes(c)) return;

            const isNoise = (c === -1);
            const color = isNoise ? NOISE_COLOR : clusterPalette[c % clusterPalette.length];

            const pts = clusters[cKey].map(pt => ({
                x: pt.x,
                y: pt.y,
                player_id: pt.player_id,
                speed: pt.speed,
                direction: pt.direction,
                status: pt.status
            }));

            datasets.push({
                label: isNoise ? 'Noise / Anomaly (-1)' : `Cluster ${c}`,
                data: pts,
                backgroundColor: isNoise ? NOISE_COLOR : color,
                borderColor: isNoise ? '#ffffff' : 'transparent',
                borderWidth: isNoise ? 1 : 0,
                pointRadius: isNoise ? 6 : 3.5,
                pointHoverRadius: isNoise ? 9 : 6
            });
        });

        state.chartInstances['chart-full-map'] = new Chart(ctx, {
            type: 'scatter',
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const pt = ctx.raw;
                                return ` [${pt.player_id}] Pos: (${pt.x}, ${pt.y}) | Speed: ${pt.speed}m/s | Dir: ${pt.direction}° | ${pt.status}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'In-Game Arena X Coordinates', color: '#94a3b8' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b' },
                        min: 0,
                        max: 1000
                    },
                    y: {
                        title: { display: true, text: 'In-Game Arena Y Coordinates', color: '#94a3b8' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b' },
                        min: 0,
                        max: 1000
                    }
                }
            }
        });
    }

    function filterMapToCluster(clusterNum) {
        if (!state.mapData) return;
        renderFullMapChart(state.mapData, [clusterNum]);
        showToast(`Filtered map to Cluster ${clusterNum === -1 ? 'Noise / Anomaly' : clusterNum}`, 'info');
    }

    elements.btnMapFilterAll.addEventListener('click', () => {
        if (state.mapData) renderFullMapChart(state.mapData);
    });

    elements.btnMapFilterAnomalies.addEventListener('click', () => {
        filterMapToCluster(-1);
    });

    elements.btnResetMapZoom.addEventListener('click', () => {
        if (state.mapData) renderFullMapChart(state.mapData);
    });

    // -------------------------------------------------------------
    // 4. ANOMALY DETECTION ENGINE (PIPELINE EXECUTION)
    // -------------------------------------------------------------
    // File upload drag & drop
    elements.fileDropzone.addEventListener('click', () => elements.csvFileInput.click());
    
    elements.fileDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.fileDropzone.classList.add('dragover');
    });

    elements.fileDropzone.addEventListener('dragleave', () => {
        elements.fileDropzone.classList.remove('dragover');
    });

    elements.fileDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.fileDropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    elements.csvFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    function handleSelectedFile(file) {
        if (!file.name.toLowerCase().endsWith('.csv')) {
            showToast('Please select a valid .csv file.', 'error');
            return;
        }
        state.selectedFile = file;
        elements.selectedFileName.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        elements.selectedFileName.style.color = '#06b6d4';
        showToast(`Loaded CSV: ${file.name}`, 'info');
    }

    // Run DBSCAN Pipeline Form Submission
    elements.detectionForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const eps = parseFloat(elements.paramEps.value);
        const minSamples = parseInt(elements.paramMinSamples.value);

        if (isNaN(eps) || eps <= 0) {
            showToast('Epsilon (eps) must be a positive number.', 'error');
            return;
        }
        if (isNaN(minSamples) || minSamples < 1) {
            showToast('min_samples must be at least 1.', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('eps', eps);
        formData.append('min_samples', minSamples);
        if (state.selectedFile) {
            formData.append('file', state.selectedFile);
        }

        elements.btnRunDbscan.disabled = true;
        elements.btnRunDbscan.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>RUNNING DBSCAN PIPELINE...</span>';

        // Animate pipeline steps sequentially for college viva demonstration
        const stepElements = document.querySelectorAll('.step-item');
        stepElements.forEach(s => s.classList.remove('active'));

        for (let i = 1; i <= 6; i++) {
            const stepEl = document.getElementById(`step-${i}`);
            if (stepEl) stepEl.classList.add('active');
            await new Promise(r => setTimeout(r, 70));
        }

        try {
            const res = await fetch('/api/detect', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            for (let i = 7; i <= 10; i++) {
                const stepEl = document.getElementById(`step-${i}`);
                if (stepEl) stepEl.classList.add('active');
                await new Promise(r => setTimeout(r, 60));
            }

            if (!data.success) throw new Error(data.error || 'DBSCAN Execution failed');

            showToast(`DBSCAN Complete! Found ${data.stats.anomalous_records} anomalies (${data.stats.anomaly_percentage}%).`, 'success');

            // Refresh Dashboard data and active views automatically
            await loadDashboard();
            if (state.currentPage === 'player-analysis') loadPlayerAnalysis(state.selectedPlayerId);
            if (state.currentPage === 'coordinate-map') loadCoordinateMap();
            if (state.currentPage === 'results-table') loadResultsTable();

        } catch (err) {
            console.error(err);
            showToast(`Detection error: ${err.message}`, 'error');
        } finally {
            elements.btnRunDbscan.disabled = false;
            elements.btnRunDbscan.innerHTML = '<i class="fa-solid fa-microchip"></i> <span>RUN DBSCAN DETECTION</span>';
        }
    });

    // -------------------------------------------------------------
    // 5. RESULTS TABLE CONTROLLER
    // -------------------------------------------------------------
    async function loadResultsTable() {
        const q = state.resultsQuery;
        const params = new URLSearchParams({
            page: q.page,
            page_size: q.pageSize,
            status: q.status,
            player_id: q.playerId,
            search: q.search,
            sort_by: q.sortBy,
            sort_order: q.sortOrder
        });

        try {
            const res = await fetch(`/api/results?${params.toString()}`);
            const data = await res.json();
            if (!data.success) throw new Error(data.error || 'Failed to load results table');

            // Summary Strip Above Table
            elements.resTotal.textContent = data.summary.total_records.toLocaleString();
            elements.resNormal.textContent = data.summary.normal_records.toLocaleString();
            elements.resAnomalous.textContent = data.summary.anomalous_records.toLocaleString();
            elements.resClusters.textContent = data.summary.num_clusters.toLocaleString();
            elements.resPct.textContent = `${data.summary.anomaly_percentage}%`;

            // Populate Table
            renderMainResultsTable(data.records);

            // Pagination Controls
            const p = data.pagination;
            elements.paginationInfoText.textContent = `Showing page ${p.page} of ${p.total_pages} (${p.total_records.toLocaleString()} records)`;
            elements.currentPageDisplay.textContent = p.page;
            elements.btnPagePrev.disabled = (p.page <= 1);
            elements.btnPageNext.disabled = (p.page >= p.total_pages);

        } catch (err) {
            console.error(err);
            showToast(`Results table error: ${err.message}`, 'error');
        }
    }

    function renderMainResultsTable(records) {
        elements.resultsTableBody.innerHTML = '';
        if (!records || records.length === 0) {
            elements.resultsTableBody.innerHTML = `
                <tr>
                    <td colspan="8" class="text-center text-muted" style="padding: 24px;">
                        No telemetry records match the current filters.
                    </td>
                </tr>
            `;
            return;
        }

        records.forEach(row => {
            const tr = document.createElement('tr');
            const isAnom = (row.status === 'ANOMALY');
            tr.innerHTML = `
                <td><strong>${row.player_id}</strong></td>
                <td>${row.timestamp}</td>
                <td>${row.x.toFixed(2)}</td>
                <td>${row.y.toFixed(2)}</td>
                <td class="${row.speed > 30 ? 'text-rose font-semibold' : ''}">${row.speed.toFixed(2)}</td>
                <td>${row.direction.toFixed(1)}°</td>
                <td>${row.cluster === -1 ? '<span class="text-rose font-semibold">-1 (Noise)</span>' : 'Cluster ' + row.cluster}</td>
                <td>
                    <span class="status-badge ${isAnom ? 'anomaly' : 'normal'}">
                        ${row.status}
                    </span>
                </td>
            `;
            elements.resultsTableBody.appendChild(tr);
        });
    }

    // Results Filters & Search Event Listeners
    let searchDebounceTimeout = null;
    elements.resultsSearchInput.addEventListener('input', (e) => {
        clearTimeout(searchDebounceTimeout);
        searchDebounceTimeout = setTimeout(() => {
            state.resultsQuery.search = e.target.value;
            state.resultsQuery.page = 1;
            loadResultsTable();
        }, 300);
    });

    elements.filterStatus.addEventListener('change', (e) => {
        state.resultsQuery.status = e.target.value;
        state.resultsQuery.page = 1;
        loadResultsTable();
    });

    elements.filterPlayer.addEventListener('change', (e) => {
        state.resultsQuery.playerId = e.target.value;
        state.resultsQuery.page = 1;
        loadResultsTable();
    });

    elements.filterPageSize.addEventListener('change', (e) => {
        state.resultsQuery.pageSize = parseInt(e.target.value);
        state.resultsQuery.page = 1;
        loadResultsTable();
    });

    elements.btnPagePrev.addEventListener('click', () => {
        if (state.resultsQuery.page > 1) {
            state.resultsQuery.page--;
            loadResultsTable();
        }
    });

    elements.btnPageNext.addEventListener('click', () => {
        state.resultsQuery.page++;
        loadResultsTable();
    });

    // Sorting by column headers
    elements.tableSortHeaders.forEach(th => {
        th.addEventListener('click', () => {
            const col = th.dataset.sort;
            if (state.resultsQuery.sortBy === col) {
                state.resultsQuery.sortOrder = (state.resultsQuery.sortOrder === 'asc' ? 'desc' : 'asc');
            } else {
                state.resultsQuery.sortBy = col;
                state.resultsQuery.sortOrder = 'desc';
            }
            loadResultsTable();
        });
    });

    // -------------------------------------------------------------
    // 6. EVALUATION PAGE CONTROLLER
    // -------------------------------------------------------------
    function updateEvaluationStats() {
        if (!state.dashboardData) return;
        const stats = state.dashboardData.stats;
        const dist = state.dashboardData.cluster_distribution;

        elements.evalClusters.textContent = stats.num_clusters;
        elements.evalNoise.textContent = stats.anomalous_records.toLocaleString();
        elements.evalPct.textContent = `${stats.anomaly_percentage}%`;
        elements.evalEps.textContent = stats.eps;
        elements.evalMinSamples.textContent = stats.min_samples;

        // Render cluster breakdown list
        elements.evalClusterBreakdown.innerHTML = '';
        dist.forEach(c => {
            const item = document.createElement('div');
            item.className = `cluster-eval-item ${c.cluster === -1 ? 'anomaly-item' : ''}`;
            item.innerHTML = `
                <span><strong>${c.label}</strong>:</span>
                <span>${c.count.toLocaleString()} points (${c.percentage}%)</span>
            `;
            elements.evalClusterBreakdown.appendChild(item);
        });
    }

    // -------------------------------------------------------------
    // INITIALIZATION
    // -------------------------------------------------------------
    loadDashboard();
});
