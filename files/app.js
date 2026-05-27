/**
 * app.js
 * UI logic — event handlers, DOM updates, history management.
 */

let history     = [];
let totalChecked = 0;
let spamFound   = 0;
let hamFound    = 0;

/* ── Helpers ──────────────────────────────── */

function updateChar() {
  const len = document.getElementById('msgInput').value.length;
  document.getElementById('charCount').textContent = len;
}

function loadSample(el) {
  const ta = document.getElementById('msgInput');
  ta.value = el.textContent.trim();
  updateChar();
  ta.focus();
}

function clearAll() {
  document.getElementById('msgInput').value = '';
  updateChar();
  document.getElementById('resultArea').classList.add('hidden');
}

function clearHistory() {
  history = [];
  renderHistory();
}

/* ── Analyse ──────────────────────────────── */

function analyze() {
  const text = document.getElementById('msgInput').value.trim();
  if (!text) return;

  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Analysing…';

  // Simulate async model call (replace setTimeout with fetch() for real backend)
  setTimeout(() => {
    const { spamProb, hamProb, foundSpam, foundHam } = scoreMessage(text);
    renderResult({ text, spamProb, hamProb, foundSpam, foundHam });

    btn.disabled = false;
    btn.innerHTML = '<i class="ti ti-search"></i> Analyse Message';
  }, 600);
}

/* ── Render result ────────────────────────── */

function renderResult({ text, spamProb, hamProb, foundSpam, foundHam }) {
  const isSpam  = spamProb >= 0.5;
  const spamPct = Math.round(spamProb * 100);
  const hamPct  = Math.round(hamProb  * 100);

  // Show result area
  document.getElementById('resultArea').classList.remove('hidden');

  // Card colour
  const card = document.getElementById('resultCard');
  card.className = 'result-card ' + (isSpam ? 'spam-result' : 'ham-result');

  // Verdict badge
  const badge = document.getElementById('verdictBadge');
  badge.className = 'verdict-badge ' + (isSpam ? 'spam-badge' : 'ham-badge');
  badge.innerHTML = isSpam
    ? '<i class="ti ti-alert-triangle"></i> Spam Detected'
    : '<i class="ti ti-circle-check"></i> Legitimate (Ham)';

  // Confidence label
  document.getElementById('confText').textContent =
    'Confidence: ' + (isSpam ? spamPct : hamPct) + '%';

  // Probability bars
  document.getElementById('hamBar').style.width   = hamPct  + '%';
  document.getElementById('spamBar').style.width  = spamPct + '%';
  document.getElementById('hamPct').textContent   = hamPct  + '%';
  document.getElementById('spamPct').textContent  = spamPct + '%';

  // Token chips
  const chips = document.getElementById('tokenChips');
  chips.innerHTML = '';

  if (foundSpam.length === 0 && foundHam.length === 0) {
    chips.innerHTML = '<span style="font-size:12px;color:var(--text-tertiary)">No strong signals found</span>';
  }

  foundSpam.slice(0, 8).forEach(t => {
    const chip = document.createElement('span');
    chip.className   = 'token-chip spam-token';
    chip.textContent = t;
    chips.appendChild(chip);
  });

  foundHam.slice(0, 6).forEach(t => {
    const chip = document.createElement('span');
    chip.className   = 'token-chip ham-token';
    chip.textContent = t;
    chips.appendChild(chip);
  });

  // Update session stats
  totalChecked++;
  if (isSpam) spamFound++; else hamFound++;
  document.getElementById('totalCount').textContent = totalChecked;
  document.getElementById('spamCount').textContent  = spamFound;
  document.getElementById('hamCount').textContent   = hamFound;

  // Push to history
  history.unshift({
    text  : text.slice(0, 60) + (text.length > 60 ? '…' : ''),
    isSpam,
    conf  : isSpam ? spamPct : hamPct,
  });
  if (history.length > 5) history.pop();
  renderHistory();
}

/* ── Render history ───────────────────────── */

function renderHistory() {
  const section = document.getElementById('historySection');
  const list    = document.getElementById('historyList');

  if (history.length === 0) {
    section.style.display = 'none';
    return;
  }

  section.style.display = 'block';
  list.innerHTML = history.map(h => `
    <div class="history-item">
      <span class="h-badge ${h.isSpam ? 'h-spam' : 'h-ham'}">${h.isSpam ? 'Spam' : 'Ham'}</span>
      <span class="h-text">${escapeHtml(h.text)}</span>
      <span class="h-conf">${h.conf}%</span>
    </div>
  `).join('');
}

/* ── Keyboard shortcut ────────────────────── */

document.getElementById('msgInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyze();
});

/* ── Utility ──────────────────────────────── */

function escapeHtml(str) {
  return str
    .replace(/&/g,  '&amp;')
    .replace(/</g,  '&lt;')
    .replace(/>/g,  '&gt;')
    .replace(/"/g,  '&quot;');
}
