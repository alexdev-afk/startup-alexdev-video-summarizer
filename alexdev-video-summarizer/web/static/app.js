/* ── Video Summarizer Web UI ──────────────────────────────────── */

const STAGES = [
  'initializing', 'ffmpeg', 'scene_detection', 'demucs_separation',
  'audio_processing', 'frame_extraction', 'video_processing', 'knowledge_generation'
];
const STAGE_LABELS = {
  initializing: 'Setup', ffmpeg: 'FFmpeg', scene_detection: 'Scene Detect',
  demucs_separation: 'Demucs', audio_processing: 'Audio', frame_extraction: 'Frames',
  video_processing: 'VLM', knowledge_generation: 'Knowledge'
};

let selectedFiles = new Set();
let currentDetailBatch = null;

// ── Tabs ─────────────────────────────────────────────────────────

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.screen').forEach(s => s.classList.toggle('active', s.id === 'screen-' + name));
  if (name === 'queue') loadQueue();
  if (name === 'history') loadHistory();
  if (name === 'detail' && currentDetailBatch) loadBatchDetail(currentDetailBatch);
  if (name === 'logs') startLogPolling();
  else stopLogPolling();
}

// ── File Browser ─────────────────────────────────────────────────

const browseInput = document.getElementById('browse-path');
const fileList = document.getElementById('file-list');

document.getElementById('btn-browse').addEventListener('click', () => browsePath(browseInput.value));
document.getElementById('btn-up').addEventListener('click', () => {
  const cur = browseInput.value.replace(/\\/g, '/');
  const parent = cur.substring(0, cur.lastIndexOf('/')) || cur.substring(0, cur.lastIndexOf('\\'));
  if (parent) browsePath(parent);
});
browseInput.addEventListener('keydown', e => { if (e.key === 'Enter') browsePath(browseInput.value); });

document.getElementById('btn-create-batch').addEventListener('click', createBatch);

async function browsePath(path) {
  const params = path ? '?path=' + encodeURIComponent(path) : '';
  try {
    const r = await fetch('/api/browse' + params);
    const data = await r.json();
    if (data.error) { toast(data.error, 'error'); return; }
    browseInput.value = data.path;
    selectedFiles.clear();
    renderFileList(data);
  } catch (e) {
    toast('Failed to browse: ' + e.message, 'error');
  }
}

function renderFileList(data) {
  fileList.innerHTML = '';
  // Parent directory entry
  if (data.parent) {
    const el = createFileItem({ name: '..', path: data.parent, is_dir: true });
    fileList.appendChild(el);
  }
  for (const entry of data.entries) {
    fileList.appendChild(createFileItem(entry));
  }
  updateSelectionCount();
}

function createFileItem(entry) {
  const div = document.createElement('div');
  div.className = 'file-item' + (entry.is_dir ? ' dir' : '');

  if (entry.is_dir) {
    const isParent = entry.name === '..';
    if (isParent) {
      div.innerHTML = `
        <span class="icon">&#128193;</span>
        <span class="name">${esc(entry.name)}</span>
      `;
      div.addEventListener('click', () => browsePath(entry.path));
    } else {
      div.innerHTML = `
        <input type="checkbox" class="folder-cb">
        <span class="icon">&#128193;</span>
        <span class="name">${esc(entry.name)}</span>
        <span class="size folder-status"></span>
      `;
      const cb = div.querySelector('input');
      const statusSpan = div.querySelector('.folder-status');

      // Checkbox toggles recursive selection
      cb.addEventListener('change', async (e) => {
        e.stopPropagation();
        if (cb.checked) {
          statusSpan.textContent = 'scanning...';
          const videos = await fetchFolderVideos(entry.path);
          statusSpan.textContent = videos.length + ' video' + (videos.length !== 1 ? 's' : '');
          for (const v of videos) selectedFiles.add(v.path);
        } else {
          const videos = await fetchFolderVideos(entry.path);
          for (const v of videos) selectedFiles.delete(v.path);
          statusSpan.textContent = '';
        }
        updateSelectionCount();
      });

      // Click on name navigates into folder; click on checkbox toggles selection
      div.addEventListener('click', (e) => {
        if (e.target === cb) return; // let checkbox handle itself
        e.stopPropagation();
        browsePath(entry.path);
      });
    }
  } else {
    const isVideo = entry.is_video;
    const checked = selectedFiles.has(entry.path) ? 'checked' : '';
    div.innerHTML = `
      <input type="checkbox" ${checked} ${isVideo ? '' : 'disabled'}>
      <span class="icon">${isVideo ? '&#127909;' : '&#128196;'}</span>
      <span class="name">${esc(entry.name)}</span>
      <span class="size">${entry.size_mb != null ? entry.size_mb + ' MB' : ''}</span>
    `;
    if (isVideo) {
      const cb = div.querySelector('input');
      cb.addEventListener('change', () => {
        if (cb.checked) selectedFiles.add(entry.path);
        else selectedFiles.delete(entry.path);
        updateSelectionCount();
      });
      div.addEventListener('click', e => {
        if (e.target.tagName !== 'INPUT') {
          cb.checked = !cb.checked;
          cb.dispatchEvent(new Event('change'));
        }
      });
    }
  }
  return div;
}

// Cache for recursive folder scans
const folderVideoCache = {};

async function fetchFolderVideos(folderPath) {
  if (folderVideoCache[folderPath]) return folderVideoCache[folderPath];
  try {
    const r = await fetch('/api/browse/videos?path=' + encodeURIComponent(folderPath));
    const data = await r.json();
    if (data.error) { toast(data.error, 'error'); return []; }
    folderVideoCache[folderPath] = data.videos;
    return data.videos;
  } catch (e) {
    toast('Failed to scan folder: ' + e.message, 'error');
    return [];
  }
}

function updateSelectionCount() {
  const n = selectedFiles.size;
  document.getElementById('selection-count').textContent = n + ' video' + (n !== 1 ? 's' : '') + ' selected';
  document.getElementById('btn-create-batch').disabled = n === 0;
}

async function createBatch() {
  if (selectedFiles.size === 0) return;
  try {
    const r = await fetch('/api/queue/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files: [...selectedFiles], output_dir: document.getElementById('output-path').value || undefined }),
    });
    const data = await r.json();
    if (data.error) { toast(data.error, 'error'); return; }
    selectedFiles.clear();
    updateSelectionCount();
    toast('Batch ' + data.batch_id + ' created', 'success');
    currentDetailBatch = data.batch_id;
    switchTab('detail');
  } catch (e) {
    toast('Failed to create batch: ' + e.message, 'error');
  }
}

// ── Queue ────────────────────────────────────────────────────────

async function loadQueue() {
  try {
    const r = await fetch('/api/queue/list');
    const batches = await r.json();
    const active = batches.filter(b => b.status !== 'completed');
    renderBatchGrid('queue-list', active, true);
  } catch (e) {
    document.getElementById('queue-list').innerHTML = '<div class="empty-state"><p>Failed to load queue</p></div>';
  }
}

async function loadHistory() {
  try {
    const r = await fetch('/api/queue/list');
    const batches = await r.json();
    const completed = batches.filter(b => b.status === 'completed');
    renderBatchGrid('history-list', completed, false);
  } catch (e) {
    document.getElementById('history-list').innerHTML = '<div class="empty-state"><p>Failed to load history</p></div>';
  }
}

function renderBatchGrid(containerId, batches, showActions) {
  const container = document.getElementById(containerId);
  if (batches.length === 0) {
    container.innerHTML = '<div class="empty-state"><div class="icon">&#128230;</div><p>No batches</p></div>';
    return;
  }
  container.innerHTML = '<div class="batch-grid">' + batches.map(b => batchCardHTML(b, showActions)).join('') + '</div>';

  // Wire up card clicks → detail view
  container.querySelectorAll('.batch-card').forEach(card => {
    card.addEventListener('click', e => {
      if (e.target.closest('.actions')) return; // don't navigate when clicking buttons
      currentDetailBatch = card.dataset.batchId;
      switchTab('detail');
    });
  });

  // Wire up action buttons
  container.querySelectorAll('[data-action]').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const action = btn.dataset.action;
      const id = btn.dataset.batchId;
      if (action === 'start') startBatch(id);
      else if (action === 'pause') pauseBatch(id);
      else if (action === 'retry') retryBatch(id);
      else if (action === 'delete') deleteBatch(id);
    });
  });
}

function batchCardHTML(b, showActions) {
  const pct = b.total_videos > 0 ? Math.round((b.completed_count / b.total_videos) * 100) : 0;
  let actions = '';
  if (showActions) {
    actions = '<div class="actions">';
    if (b.status === 'pending' || b.status === 'paused')
      actions += `<button class="btn btn-success btn-sm" data-action="start" data-batch-id="${b.batch_id}">Start</button>`;
    if (b.status === 'processing')
      actions += `<button class="btn btn-secondary btn-sm" data-action="pause" data-batch-id="${b.batch_id}">Pause</button>`;
    if (b.failed_count > 0)
      actions += `<button class="btn btn-secondary btn-sm" data-action="retry" data-batch-id="${b.batch_id}">Retry</button>`;
    actions += `<button class="btn btn-danger btn-sm" data-action="delete" data-batch-id="${b.batch_id}">Delete</button>`;
    actions += '</div>';
  }
  return `
    <div class="batch-card" data-batch-id="${b.batch_id}">
      <div class="batch-header">
        <span class="batch-id">Batch ${b.batch_id}</span>
        <span class="badge badge-${b.status}">${b.status}</span>
      </div>
      <div class="stats">
        <span>${b.completed_count}/${b.total_videos} done</span>
        ${b.failed_count > 0 ? `<span style="color:var(--error)">${b.failed_count} failed</span>` : ''}
      </div>
      <div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>
      ${actions}
    </div>`;
}

// ── Batch Detail ─────────────────────────────────────────────────

document.getElementById('btn-back-queue').addEventListener('click', () => switchTab('queue'));
document.getElementById('detail-start').addEventListener('click', () => { if (currentDetailBatch) startBatch(currentDetailBatch); });
document.getElementById('detail-pause').addEventListener('click', () => { if (currentDetailBatch) pauseBatch(currentDetailBatch); });
document.getElementById('detail-retry').addEventListener('click', () => { if (currentDetailBatch) retryBatch(currentDetailBatch); });

async function loadBatchDetail(batchId) {
  try {
    const r = await fetch('/api/queue/' + batchId);
    const batch = await r.json();
    if (batch.error) { toast(batch.error, 'error'); return; }
    renderBatchDetail(batch);
  } catch (e) {
    toast('Failed to load batch: ' + e.message, 'error');
  }
}

function renderBatchDetail(batch) {
  document.getElementById('detail-title').textContent = 'Batch ' + batch.batch_id;
  const badge = document.getElementById('detail-badge');
  badge.textContent = batch.status;
  badge.className = 'badge badge-' + batch.status;

  // Toggle buttons
  const canStart = batch.status === 'pending' || batch.status === 'paused';
  const canPause = batch.status === 'processing';
  const hasFailed = batch.failed_count > 0;
  document.getElementById('detail-start').style.display = canStart ? '' : 'none';
  document.getElementById('detail-pause').style.display = canPause ? '' : 'none';
  document.getElementById('detail-retry').style.display = hasFailed ? '' : 'none';

  const tbody = document.getElementById('detail-tbody');
  tbody.innerHTML = batch.videos.map((v, i) => {
    const pipeline = renderPipeline(v);
    const timeStr = v.processing_time != null ? formatTime(v.processing_time) : '-';
    return `
      <tr>
        <td>${i + 1}</td>
        <td>${esc(v.filename)}</td>
        <td>${v.size_mb} MB</td>
        <td>${pipeline}</td>
        <td><span class="status-text status-${v.status}">${v.status}${v.error ? ': ' + esc(v.error.substring(0, 60)) : ''}</span></td>
        <td>${timeStr}</td>
      </tr>`;
  }).join('');
}

function renderPipeline(video) {
  if (video.status === 'pending') {
    return '<div class="pipeline">' + STAGES.map(s =>
      `<div class="stage" title="${STAGE_LABELS[s]}"></div>`).join('') + '</div>';
  }

  const currentIdx = video.current_stage ? STAGES.indexOf(video.current_stage) : -1;
  return '<div class="pipeline">' + STAGES.map((s, i) => {
    let cls = 'stage';
    if (video.status === 'completed') cls += ' completed';
    else if (video.status === 'failed') {
      if (i <= currentIdx) cls += (i === currentIdx ? ' failed' : ' completed');
    } else if (video.status === 'in_progress') {
      if (i < currentIdx) cls += ' completed';
      else if (i === currentIdx) cls += ' active';
    }
    return `<div class="${cls}" title="${STAGE_LABELS[s]}"></div>`;
  }).join('') + '</div>';
}

// ── API Actions ──────────────────────────────────────────────────

async function startBatch(id) {
  const r = await fetch('/api/queue/' + id + '/start', { method: 'POST' });
  const data = await r.json();
  if (data.error) toast(data.error, 'error');
  else toast('Batch ' + id + ' starting...', 'info');
}

async function pauseBatch(id) {
  const r = await fetch('/api/queue/' + id + '/pause', { method: 'POST' });
  const data = await r.json();
  if (data.error) toast(data.error, 'error');
  else toast('Pausing after current video...', 'info');
}

async function retryBatch(id) {
  const r = await fetch('/api/queue/' + id + '/retry-failed', { method: 'POST' });
  const data = await r.json();
  if (data.error) toast(data.error, 'error');
  else { toast('Failed videos re-queued', 'success'); if (currentDetailBatch === id) loadBatchDetail(id); }
}

async function deleteBatch(id) {
  if (!confirm('Delete batch ' + id + '?')) return;
  const r = await fetch('/api/queue/' + id, { method: 'DELETE' });
  const data = await r.json();
  if (data.error) toast(data.error, 'error');
  else { toast('Batch ' + id + ' deleted', 'success'); loadQueue(); loadHistory(); }
}

// ── SSE ──────────────────────────────────────────────────────────

let evtSource = null;

function connectSSE() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource('/api/events');

  evtSource.onerror = () => {
    evtSource.close();
    setTimeout(connectSSE, 3000);
  };

  // Listen to each event type
  const eventTypes = [
    'batch_started', 'batch_paused', 'batch_completed', 'batch_created',
    'batch_updated', 'batch_deleted', 'video_started', 'video_completed',
    'video_failed', 'progress', 'circuit_breaker'
  ];

  eventTypes.forEach(type => {
    evtSource.addEventListener(type, e => handleSSE(type, JSON.parse(e.data)));
  });
}

function handleSSE(type, data) {
  // Update worker status indicator
  if (type === 'batch_started') {
    setWorkerStatus(true);
    toast('Batch ' + data.batch_id + ' started', 'info');
  } else if (type === 'batch_paused' || type === 'batch_completed') {
    setWorkerStatus(false);
  }

  if (type === 'batch_completed') {
    toast('Batch ' + data.batch_id + ' completed!', 'success');
  }

  if (type === 'video_completed') {
    toast(data.filename + ' done (' + formatTime(data.processing_time) + ')', 'success');
  }

  if (type === 'video_failed') {
    toast(data.filename + ' failed: ' + (data.error || '').substring(0, 80), 'error');
  }

  if (type === 'circuit_breaker') {
    toast('Circuit breaker: ' + data.consecutive_failures + ' consecutive failures — auto-paused', 'error');
  }

  // Refresh views if relevant batch is visible
  const activeScreen = document.querySelector('.screen.active');
  if (activeScreen) {
    const screenId = activeScreen.id;
    if (screenId === 'screen-queue') loadQueue();
    if (screenId === 'screen-history' && type === 'batch_completed') loadHistory();
    if (screenId === 'screen-detail' && currentDetailBatch === data.batch_id) {
      loadBatchDetail(data.batch_id);
    }
  }
}

function setWorkerStatus(running) {
  const dot = document.getElementById('worker-dot');
  const label = document.getElementById('worker-label');
  dot.classList.toggle('running', running);
  label.textContent = running ? 'Processing' : 'Idle';
}

// ── Utilities ────────────────────────────────────────────────────

function esc(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function formatTime(seconds) {
  if (seconds == null) return '-';
  if (seconds < 60) return Math.round(seconds) + 's';
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m + 'm ' + s + 's';
}

function toast(msg, type) {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = 'toast ' + (type || 'info');
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

// ── Logs ─────────────────────────────────────────────────────────

let logSeq = 0;
let logPollTimer = null;

function startLogPolling() {
  if (logPollTimer) return;
  pollLogs(); // immediate first fetch
  logPollTimer = setInterval(pollLogs, 2000);
}

function stopLogPolling() {
  if (logPollTimer) { clearInterval(logPollTimer); logPollTimer = null; }
}

async function pollLogs() {
  try {
    const r = await fetch('/api/logs?since=' + logSeq);
    const data = await r.json();
    if (data.logs.length > 0) {
      const panel = document.getElementById('log-panel');
      for (const line of data.logs) {
        const div = document.createElement('div');
        div.className = 'log-line';
        if (/ERROR/i.test(line)) div.className += ' error';
        else if (/WARN/i.test(line)) div.className += ' warning';
        div.textContent = line;
        panel.appendChild(div);
      }
      // Cap DOM nodes
      while (panel.children.length > 5000) panel.removeChild(panel.firstChild);
      // Auto-scroll
      if (document.getElementById('log-autoscroll').checked) {
        panel.scrollTop = panel.scrollHeight;
      }
    }
    logSeq = data.seq;
  } catch (e) { /* ignore transient failures */ }
}

document.getElementById('btn-clear-logs').addEventListener('click', () => {
  document.getElementById('log-panel').innerHTML = '';
});

// ── Init ─────────────────────────────────────────────────────────

// Load defaults from config
fetch('/api/config').then(r => r.json()).then(cfg => {
  document.getElementById('output-path').value = cfg.output_dir || 'output';
}).catch(() => {});

browsePath('');
connectSSE();
