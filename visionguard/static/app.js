/**
 * VisionGuard Frontend Interactivity Script
 * Supports Open-Vocabulary Grounding, 12-Class Claim Taxonomy, 4-State Uncertainty & Constrained Repair
 */

let selectedFile = null;
let currentSampleFeatures = null;

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initFileUploadAndDropzone();
    initPlaygroundPresets();
    loadHQMTable();
    loadLeaderboard();
    loadBaselinesAndAblations();
    initGuardrailDemo();
});

// Tab Navigation Handler
function initNavigation() {
    const tabs = document.querySelectorAll('.nav-tab');
    const panels = document.querySelectorAll('.tab-panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const target = tab.dataset.tab;
            document.getElementById(`tab-${target}`).classList.add('active');
        });
    });
}

// File Upload & Drag and Drop Zone Handler
function initFileUploadAndDropzone() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const dropContent = dropZone.querySelector('.drop-zone-content');
    const previewContainer = document.getElementById('image-preview-container');
    const previewImg = document.getElementById('image-preview');
    const fileNameLabel = document.getElementById('file-name-label');
    const btnRemove = document.getElementById('btn-remove-image');

    dropZone.addEventListener('click', (e) => {
        if (e.target !== btnRemove) {
            fileInput.click();
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    btnRemove.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        currentSampleFeatures = null;
        fileInput.value = '';
        previewImg.src = '';
        previewContainer.classList.add('hidden');
        dropContent.classList.remove('hidden');
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG, PNG, WebP)');
            return;
        }
        selectedFile = file;
        currentSampleFeatures = null;
        const reader = new FileReader();
        reader.onload = (evt) => {
            previewImg.src = evt.target.result;
            fileNameLabel.innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            dropContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
}

// Preset Handler for Playground
function initPlaygroundPresets() {
    const presets = {
        1: {
            text: "A young boy wearing a red shirt is riding a bicycle near a park.",
            features: {
                objects: {"boy": 0.95, "bicycle": 0.92, "shirt": 0.94, "person": 0.95},
                open_vocab_detections: {"young boy": 0.94, "red shirt": 0.92, "bicycle": 0.95},
                color_distribution: {"red": 0.35, "green": 0.40, "blue": 0.15},
                dominant_color: "red",
                scene: {"type": "outdoor park", "is_outdoor": true, "confidence": 0.92},
                attributes: {"red shirt": 0.91, "young boy": 0.89},
                relations: {"riding bicycle": 0.93, "near park": 0.88}
            }
        },
        2: {
            text: "A brown dog is sitting beside a red bicycle in Paris.",
            features: {
                objects: {"dog": 0.92, "bicycle": 0.89, "animal": 0.94},
                open_vocab_detections: {"brown dog": 0.92, "red bicycle": 0.90},
                color_distribution: {"brown": 0.50, "blue": 0.30, "green": 0.15},
                dominant_color: "brown",
                scene: {"type": "outdoor yard", "is_outdoor": true, "confidence": 0.85},
                attributes: {"brown dog": 0.90, "blue bicycle": 0.85},
                relations: {"sitting beside": 0.86}
            }
        },
        3: {
            text: "An alien is driving a flying car above a snowy field.",
            features: {
                objects: {"car": 0.85, "road": 0.88},
                open_vocab_detections: {"snowy road": 0.90},
                color_distribution: {"white": 0.60, "black": 0.25},
                dominant_color: "white",
                scene: {"type": "snowy road", "is_outdoor": true, "confidence": 0.90},
                attributes: {"snowy field": 0.89},
                relations: {"driving": 0.82}
            }
        }
    };

    [1, 2, 3].forEach(num => {
        document.getElementById(`preset-${num}`).addEventListener('click', () => {
            selectedFile = null;
            currentSampleFeatures = presets[num].features;
            document.getElementById('image-preview-container').classList.add('hidden');
            document.querySelector('.drop-zone-content').classList.remove('hidden');
            document.getElementById('response-input').value = presets[num].text;
            runCAMVAnalysis();
        });
    });

    document.getElementById('btn-analyze').addEventListener('click', runCAMVAnalysis);
}

// Run CAM-V Analysis
async function runCAMVAnalysis() {
    const responseText = document.getElementById('response-input').value;
    const fusionMode = document.getElementById('fusion-mode').value;

    if (!selectedFile && !currentSampleFeatures && (!responseText || !responseText.trim())) {
        alert("Please upload an image file or select a sample preset.");
        return;
    }

    const btn = document.getElementById('btn-analyze');
    btn.disabled = true;
    btn.innerText = 'Processing Image & Verifying Claims...';

    try {
        let data;
        if (selectedFile) {
            const formData = new FormData();
            formData.append('file', selectedFile);
            if (responseText && responseText.trim()) {
                formData.append('response_text', responseText);
            }
            formData.append('fusion_mode', fusionMode);

            const res = await fetch('/api/upload_and_analyze', {
                method: 'POST',
                body: formData
            });
            data = await res.json();
        } else {
            const res = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    response_text: responseText,
                    visual_features: currentSampleFeatures,
                    fusion_mode: fusionMode,
                    enable_repair: true
                })
            });
            data = await res.json();
        }

        renderAnalysisResults(data);
    } catch (err) {
        console.error("Analysis API failed:", err);
        alert("Failed to analyze image/response. Check server status.");
    } finally {
        btn.disabled = false;
        btn.innerText = 'Run CAM-V Analysis';
    }
}

// Render Analysis Output with Deep Granular Inspection Details
function renderAnalysisResults(data) {
    const title = document.getElementById('overall-verdict');
    const subtitle = document.getElementById('overall-confidence');

    title.innerText = data.overall_verdict;
    title.className = `verdict-title ${data.overall_verdict}`;
    subtitle.innerText = `Overall Confidence: ${(data.overall_confidence * 100).toFixed(1)}% | Hallucination Risk: ${(data.hallucination_probability * 100).toFixed(1)}%`;

    document.getElementById('main-hal-rate').innerText = `${(data.decomposed_metrics.main_hal_rate * 100).toFixed(0)}%`;
    document.getElementById('extra-hal-count').innerText = data.decomposed_metrics.extra_hal_count.toFixed(1);
    document.getElementById('overall-hal-rate').innerText = `${(data.decomposed_metrics.overall_hal_rate * 100).toFixed(0)}%`;

    const container = document.getElementById('claims-container');
    container.innerHTML = '';

    if (!data.claims || data.claims.length === 0) {
        container.innerHTML = '<p class="placeholder-text">No visual claims extracted.</p>';
        return;
    }

    data.claims.forEach((item) => {
        const card = document.createElement('div');
        card.className = 'claim-card';

        const ling = item.claim.linguistic_structure || {};
        const lingStr = `Subject: ${ling.subject || 'N/A'} | Action: ${ling.predicate || 'N/A'} | Target: ${ling.target_object || 'None'} | Modifiers: [${(ling.modifiers || []).join(', ') || 'None'}]`;
        const depStr = (item.claim.dependencies && item.claim.dependencies.length > 0) ? `Depends on: [${item.claim.dependencies.join(', ')}]` : 'Root Entity';

        const evidencesHtml = (item.evidences || []).map(e => `
            <div class="evidence-pill">
                <span>${e.source}:</span>
                <span class="evidence-score">${(e.score * 100).toFixed(0)}%</span>
            </div>
        `).join('');

        const cotLogsHtml = (item.deep_diagnostics && item.deep_diagnostics.cot_logs || []).map(log => `<li style="margin-bottom:0.3rem;">${log}</li>`).join('');
        const formulaStr = item.fusion_math ? item.fusion_math.formula : 'Score = Multi-verifier weighted sum';

        card.innerHTML = `
            <div class="claim-header">
                <div>
                    <span class="claim-id">${item.claim.claim_id}</span>
                    <span class="claim-type-badge">${item.claim.type}</span>
                    ${item.claim.polarity === 'negative' ? '<span class="claim-type-badge" style="color:var(--contradicted);">[NEGATION]</span>' : ''}
                    <strong style="margin-left:0.5rem;">"${item.claim.text}"</strong>
                </div>
                <span class="verdict-badge ${item.verdict}">${item.verdict}</span>
            </div>

            <p style="font-size:0.82rem; color:var(--text-muted); margin-bottom:0.5rem;">${item.explanation}</p>

            <div class="evidences-list">
                <span style="font-size:0.75rem; color:var(--text-muted); align-self:center;">Multi-Verifier Evidence:</span>
                ${evidencesHtml || '<span style="font-size:0.75rem; color:var(--text-muted);">None (Ungrounded)</span>'}
            </div>

            <!-- Deep Inspection Details Accordion -->
            <details style="margin-top:0.8rem; border-top:1px solid var(--border-color); padding-top:0.6rem;">
                <summary style="font-size:0.8rem; font-weight:600; color:var(--text-main); cursor:pointer;">
                    Show Deep Technical Inspection & Reasoning Logs
                </summary>
                
                <div style="font-size:0.8rem; color:var(--text-muted); margin-top:0.6rem; display:flex; flex-direction:column; gap:0.6rem;">
                    <div>
                        <strong style="color:var(--text-main);">Dependency Graph:</strong>
                        <span style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted); margin-left:0.4rem;">${depStr}</span>
                    </div>

                    <div>
                        <strong style="color:var(--text-main);">Syntactic & Semantic Breakdown:</strong><br>
                        <code style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted);">${lingStr}</code>
                    </div>

                    <div>
                        <strong style="color:var(--text-main);">Evidence Fusion Mathematics:</strong><br>
                        <code style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted);">${formulaStr}</code>
                    </div>

                    <div>
                        <strong style="color:var(--text-main);">Chain-of-Thought (CoT) Verification Prompt Logs:</strong>
                        <ul style="margin-left:1.2rem; font-size:0.78rem; margin-top:0.2rem;">
                            ${cotLogsHtml || '<li>No grounding image provided.</li>'}
                        </ul>
                    </div>
                </div>
            </details>
        `;
        container.appendChild(card);
    });

    // Render Constrained Repaired Response Card if available
    if (data.repaired_response && data.repaired_response.removed_claims.length > 0) {
        const repairCard = document.createElement('div');
        repairCard.className = 'claim-card';
        repairCard.style.borderLeft = '3px solid var(--supported)';
        repairCard.style.marginTop = '1rem';
        repairCard.innerHTML = `
            <div class="claim-header">
                <div>
                    <span class="claim-type-badge" style="background:var(--supported-bg); color:var(--supported);">CONSTRAINED REPAIR</span>
                    <strong style="margin-left:0.5rem;">Factually Grounded Repaired Output</strong>
                </div>
                <span class="verdict-badge Supported">Re-Verified (Score ${(data.repaired_response.re_verified_confidence * 100).toFixed(0)}%)</span>
            </div>
            <div style="background:rgba(255,255,255,0.02); padding:0.8rem; border-radius:6px; border:1px solid var(--border-color); margin-top:0.5rem; font-size:0.88rem; color:var(--text-main);">
                "${data.repaired_response.repaired_text}"
            </div>
            <div style="font-size:0.78rem; color:var(--text-muted); margin-top:0.6rem;">
                <strong>Removed Contradicted Claims:</strong>
                <span style="color:var(--contradicted);">${data.repaired_response.removed_claims.join(', ')}</span>
            </div>
        `;
        container.appendChild(repairCard);
    }
}

// Load HQM Table
async function loadHQMTable() {
    try {
        const res = await fetch('/api/hqm');
        const data = await res.json();
        const tbody = document.querySelector('#hqm-table tbody');
        tbody.innerHTML = '';

        data.benchmarks.forEach(item => {
            const tr = document.createElement('tr');
            if (item.name.includes("HQH")) tr.style.fontWeight = 'bold';
            tr.innerHTML = `
                <td>${item.name}</td>
                <td>${item.task}</td>
                <td>${item.metric}</td>
                <td style="color:${item.test_retest > 0.95 ? 'var(--supported)' : 'var(--contradicted)'}">${item.test_retest.toFixed(4)}</td>
                <td style="color:${item.parallel_forms > 0.90 ? 'var(--supported)' : 'var(--contradicted)'}">${item.parallel_forms.toFixed(4)}</td>
                <td>${(item.content_validity * 100).toFixed(0)}%</td>
                <td style="color:${item.criterion_validity > 0.90 ? 'var(--supported)' : 'var(--uncertain)'}">${item.criterion_validity.toFixed(4)}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("HQM table load failed:", err);
    }
}

// Load Leaderboard & Radar Chart
async function loadLeaderboard() {
    try {
        const res = await fetch('/api/leaderboard');
        const data = await res.json();
        const leaderboard = data.leaderboard;

        const tbody = document.querySelector('#leaderboard-table tbody');
        tbody.innerHTML = '';

        leaderboard.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>#${row.rank}</td>
                <td><strong>${row.model}</strong></td>
                <td>${(row.main_hal_rate * 100).toFixed(1)}%</td>
                <td>${row.extra_hal_count.toFixed(3)}</td>
                <td><span class="verdict-badge ${row.overall_hal_rate < 0.45 ? 'Supported' : 'Contradicted'}">${(row.overall_hal_rate * 100).toFixed(1)}%</span></td>
                <td>${(row.dimensions.existence * 100).toFixed(1)}%</td>
                <td>${(row.dimensions.spatial * 100).toFixed(1)}%</td>
                <td>${(row.dimensions.environment * 100).toFixed(1)}%</td>
            `;
            tbody.appendChild(tr);
        });

        renderRadarChart(leaderboard.slice(0, 4));
    } catch (err) {
        console.error("Leaderboard load failed:", err);
    }
}

function renderRadarChart(topModels) {
    const ctx = document.getElementById('radarChart').getContext('2d');
    const dimensions = ["existence", "count", "color", "action", "spatial", "comparison", "environment", "text"];
    const labels = ["Object Existence", "Count", "Color", "Action", "Spatial Relation", "Comparison Relation", "Environment", "Text"];

    const colors = ['#ffffff', '#a1a1aa', '#71717a', '#52525b'];

    const datasets = topModels.map((m, i) => ({
        label: m.model,
        data: dimensions.map(d => m.dimensions[d]),
        borderColor: colors[i % colors.length],
        backgroundColor: 'transparent',
        borderWidth: 2
    }));

    new Chart(ctx, {
        type: 'radar',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: '#27272a' },
                    grid: { color: '#27272a' },
                    pointLabels: { color: '#a1a1aa', font: { size: 11 } },
                    ticks: { color: '#71717a', backdropColor: 'transparent' }
                }
            },
            plugins: { legend: { labels: { color: '#ffffff' } } }
        }
    });
}

async function loadBaselinesAndAblations() {
    try {
        const [bRes, aRes] = await Promise.all([
            fetch('/api/benchmark/run'),
            fetch('/api/ablation/run')
        ]);
        const bData = await bRes.json();
        const aData = await aRes.json();

        renderBaselineChart(bData.methods);
        renderAblationChart(aData.configurations);
    } catch (err) {
        console.error("Ablation/Baseline load failed:", err);
    }
}

function renderBaselineChart(methods) {
    const ctx = document.getElementById('baselineChart').getContext('2d');
    const labels = Object.keys(methods);
    const f1Scores = labels.map(k => methods[k].f1_score);
    const aurocScores = labels.map(k => methods[k].auroc);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: 'F1 Score', data: f1Scores, backgroundColor: '#ffffff' },
                { label: 'AUROC', data: aurocScores, backgroundColor: '#71717a' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, max: 1.0, grid: { color: '#27272a' }, ticks: { color: '#a1a1aa' } },
                x: { ticks: { color: '#a1a1aa' } }
            },
            plugins: { legend: { labels: { color: '#ffffff' } } }
        }
    });
}

function renderAblationChart(configs) {
    const ctx = document.getElementById('ablationChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: configs.map(c => c.name),
            datasets: [
                { label: 'F1 Score', data: configs.map(c => c.f1), backgroundColor: '#ffffff' }
            ]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { beginAtZero: true, max: 1.0, grid: { color: '#27272a' }, ticks: { color: '#a1a1aa' } },
                y: { ticks: { color: '#a1a1aa' } }
            },
            plugins: { legend: { labels: { color: '#ffffff' } } }
        }
    });
}

function initGuardrailDemo() {
    document.getElementById('btn-apply-guardrail').addEventListener('click', async () => {
        const prompt = document.getElementById('gr-prompt').value;
        const responseText = document.getElementById('gr-response').value;

        try {
            const res = await fetch('/api/guardrail/apply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, response_text: responseText })
            });

            const data = await res.json();
            const box = document.getElementById('gr-output');
            box.innerHTML = `
                <div style="margin-bottom:0.8rem;">
                    <strong>Decorated Prompt (Task ID injected):</strong><br>
                    <code style="color:#ffffff; font-family:var(--font-mono);">${data.decorated_prompt}</code>
                </div>
                <div style="margin-bottom:0.8rem;">
                    <strong>Constrained Repaired Response:</strong><br>
                    <div style="background:var(--supported-bg); border:1px solid var(--supported); padding:0.8rem; border-radius:6px; margin-top:0.3rem; color:var(--text-main);">
                        ${data.mitigated_response}
                    </div>
                </div>
                <div>
                    <strong>Removed Contradicted Claims:</strong>
                    <ul style="color:var(--contradicted); margin-left:1.2rem; font-size:0.82rem; margin-top:0.3rem;">
                        ${data.removed_claims.map(c => `<li>${c}</li>`).join('') || '<li>None</li>'}
                    </ul>
                </div>
            `;
        } catch (err) {
            console.error("Guardrail execution failed:", err);
        }
    });
}
