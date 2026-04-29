// voice-bridge: HTTP/SSE server inside an isolated VSCode instance.
// - Spawns a `claude` terminal in this VSCode window
// - POST /prompt {text}  -> terminal.sendText(text)
// - GET  /events (SSE)   -> stream new assistant messages from transcript jsonl
// - GET  /status         -> diagnostic info

const vscode = require('vscode');
const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');

const PORT = parseInt(process.env.VOICE_BRIDGE_PORT || '43117', 10);
const CLAUDE_BIN = process.env.VOICE_BRIDGE_CLAUDE || 'claude';

let claudeTerminal = null;
let httpServer = null;
let sseClients = new Set();
let watcherCleanup = null;
let currentTranscriptPath = null;
let currentProjectDir = null;
let transcriptOffset = 0;
let projectsRoot = path.join(os.homedir(), '.claude', 'projects');
let workspaceFsPath = null;
let outputChannel = null;

function log(msg) {
  const line = `[${new Date().toISOString()}] ${msg}`;
  if (outputChannel) outputChannel.appendLine(line);
  console.log(line);
}

function broadcast(event, data) {
  const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const res of sseClients) {
    try { res.write(payload); } catch (e) { /* ignore */ }
  }
}

// We don't try to predict Claude's project-dir encoding (it varies by drive-letter
// case and PowerShell vs bash startup). Instead we scan ALL jsonl files and use
// each file's own `cwd` field (recorded by claude on every record) to identify
// which session belongs to our workspace.
function listAllSessionFiles() {
  if (!fs.existsSync(projectsRoot)) return [];
  const out = [];
  for (const dirName of fs.readdirSync(projectsRoot)) {
    const dirFull = path.join(projectsRoot, dirName);
    let dstat;
    try { dstat = fs.statSync(dirFull); } catch { continue; }
    if (!dstat.isDirectory()) continue;
    let entries;
    try { entries = fs.readdirSync(dirFull); } catch { continue; }
    for (const f of entries) {
      if (!f.endsWith('.jsonl')) continue;
      const full = path.join(dirFull, f);
      try {
        const fstat = fs.statSync(full);
        out.push({ full, mtime: fstat.mtimeMs });
      } catch { /* race */ }
    }
  }
  out.sort((a, b) => b.mtime - a.mtime);
  return out;
}

function readJsonlCwd(filePath) {
  // Read up to the first ~16 KB and grab the first record's cwd. Records that
  // claude writes from a real chat (user/assistant/attachment) carry a `cwd` field.
  let fd;
  try { fd = fs.openSync(filePath, 'r'); } catch { return null; }
  const buf = Buffer.alloc(16 * 1024);
  let n = 0;
  try { n = fs.readSync(fd, buf, 0, buf.length, 0); } catch {}
  fs.closeSync(fd);
  if (!n) return null;
  for (const line of buf.slice(0, n).toString('utf8').split('\n')) {
    if (!line) continue;
    if (!line.includes('"cwd"')) continue;
    try {
      const rec = JSON.parse(line);
      if (typeof rec.cwd === 'string') return rec.cwd;
    } catch { /* incomplete line */ }
  }
  return null;
}

function normPath(p) {
  if (!p) return '';
  return p.replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase();
}

function findOurTranscript() {
  // Most recent jsonl whose cwd matches our workspace. Falls back to the most
  // recent jsonl overall if none match (defensive — should not happen in practice).
  const sessions = listAllSessionFiles();
  if (!sessions.length) return null;
  const want = normPath(workspaceFsPath);
  if (!want) return sessions[0].full;
  for (const s of sessions) {
    const cwd = readJsonlCwd(s.full);
    if (cwd && normPath(cwd) === want) return s.full;
  }
  return null;
}

function findLatestSessionFile(projDir) {
  if (!fs.existsSync(projDir)) return null;
  const entries = fs.readdirSync(projDir)
    .filter(f => f.endsWith('.jsonl'))
    .map(f => {
      const full = path.join(projDir, f);
      try { return { full, mtime: fs.statSync(full).mtimeMs }; } catch { return null; }
    })
    .filter(Boolean)
    .sort((a, b) => b.mtime - a.mtime);
  return entries.length ? entries[0].full : null;
}

function extractAssistantText(record) {
  // Claude transcript jsonl has lines like:
  // {"type":"assistant","message":{"content":[{"type":"text","text":"..."}]}, ...}
  if (!record || record.type !== 'assistant') return null;
  // Skip API-error synthesized messages — those carry text like
  // "API Error: 400 ... content filtering" which we don't want TTS'd.
  // The watcher emits a separate 'error' SSE event for them instead.
  if (record.isApiErrorMessage) return null;
  const msg = record.message;
  if (!msg || !Array.isArray(msg.content)) return null;
  // Synthetic error messages also carry model="<synthetic>".
  if (msg.model === '<synthetic>') return null;
  const parts = [];
  for (const block of msg.content) {
    if (block && block.type === 'text' && typeof block.text === 'string') {
      parts.push(block.text);
    }
  }
  return parts.length ? parts.join('\n') : null;
}

function extractApiError(record) {
  if (!record || record.type !== 'assistant') return null;
  if (!record.isApiErrorMessage) return null;
  const status = record.apiErrorStatus || 0;
  let msg = '';
  const content = record.message && record.message.content;
  if (Array.isArray(content) && content.length && content[0].text) {
    msg = String(content[0].text);
  }
  // Try to pull the inner error message field for cleaner reporting
  const m = msg.match(/"message"\s*:\s*"([^"]+)"/);
  const detail = m ? m[1] : msg;
  return { status, detail };
}

function readNewLines(filePath) {
  // Read incrementally from transcriptOffset and emit assistant text events.
  let stat;
  try { stat = fs.statSync(filePath); } catch { return; }
  if (stat.size < transcriptOffset) {
    // File was rotated/truncated.
    transcriptOffset = 0;
  }
  if (stat.size === transcriptOffset) return;

  const fd = fs.openSync(filePath, 'r');
  const len = stat.size - transcriptOffset;
  const buf = Buffer.alloc(len);
  fs.readSync(fd, buf, 0, len, transcriptOffset);
  fs.closeSync(fd);
  transcriptOffset = stat.size;

  const lines = buf.toString('utf8').split('\n').filter(Boolean);
  for (const line of lines) {
    let rec;
    try { rec = JSON.parse(line); } catch { continue; }
    const apiErr = extractApiError(rec);
    if (apiErr) {
      log(`api error (${apiErr.status}): ${apiErr.detail}`);
      broadcast('error', { status: apiErr.status, detail: apiErr.detail, ts: Date.now() });
      continue;
    }
    const text = extractAssistantText(rec);
    if (text) {
      log(`assistant message (${text.length} chars)`);
      broadcast('assistant', { text, ts: Date.now() });
    }
  }
}

function attachTranscriptWatcher() {
  if (watcherCleanup) { watcherCleanup(); watcherCleanup = null; }

  let lastScanForNewSession = 0;
  const pollInterval = setInterval(() => {
    // Re-scan for our session every 1s OR if we don't yet have one. Once locked
    // we just tail it; we still re-check periodically so a /clear or new session
    // is picked up.
    const now = Date.now();
    const needScan = !currentTranscriptPath || (now - lastScanForNewSession > 1000);
    if (needScan) {
      lastScanForNewSession = now;
      const found = findOurTranscript();
      if (found && found !== currentTranscriptPath) {
        // Only switch if newer (avoids flapping between similar mtimes).
        let newer = true;
        if (currentTranscriptPath) {
          try {
            newer = fs.statSync(found).mtimeMs > fs.statSync(currentTranscriptPath).mtimeMs;
          } catch { newer = true; }
        }
        if (newer) {
          log(`watching session: ${found}`);
          currentTranscriptPath = found;
          currentProjectDir = path.dirname(found);
          transcriptOffset = 0;
        }
      }
    }
    if (currentTranscriptPath) readNewLines(currentTranscriptPath);
  }, 500);

  watcherCleanup = () => {
    if (pollInterval) clearInterval(pollInterval);
  };
}

function ensureClaudeTerminal() {
  if (claudeTerminal && claudeTerminal.exitStatus === undefined) return claudeTerminal;
  const cwd = workspaceFsPath || undefined;
  log(`creating claude terminal (cwd=${cwd})`);
  claudeTerminal = vscode.window.createTerminal({
    name: 'Claude (voice)',
    cwd,
  });
  claudeTerminal.show(false);
  // Wait long enough for the PowerShell profile to finish initialising.
  // Earlier we saw MCP servers (computer-control etc.) fail to spawn when
  // claude was launched while the shell was still busy with its profile —
  // the cold-start race made the user-level MCP show up as ✘ failed in
  // /mcp even though manual `claude` from the same shell connects fine.
  setTimeout(() => {
    if (claudeTerminal) claudeTerminal.sendText(CLAUDE_BIN);
  }, 3000);
  return claudeTerminal;
}

function sendPromptToClaude(text) {
  ensureClaudeTerminal();
  // Collapse to single line for now (claude CLI input box is single-line submit on Enter).
  const oneLine = text.replace(/\r?\n/g, ' ').trim();
  if (!oneLine) return;
  // Claude CLI uses bracketed paste mode: a single sendText with embedded newline
  // is treated as a paste, not an Enter press. Send the body first WITHOUT a
  // newline, then a separate '\r' which the TUI's keypress handler interprets
  // as Enter and submits the prompt.
  claudeTerminal.sendText(oneLine, false);
  setTimeout(() => {
    if (claudeTerminal) claudeTerminal.sendText('\r', false);
  }, 120);
  log(`sent prompt (${oneLine.length} chars)`);
}

function startHttpServer() {
  httpServer = http.createServer((req, res) => {
    if (req.method === 'GET' && req.url === '/status') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        ok: true,
        port: PORT,
        workspace: workspaceFsPath,
        projectDir: currentProjectDir,
        transcript: currentTranscriptPath,
        terminalAlive: claudeTerminal && claudeTerminal.exitStatus === undefined,
        sseClients: sseClients.size,
      }));
      return;
    }

    if (req.method === 'GET' && req.url === '/events') {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      });
      res.write(': connected\n\n');
      sseClients.add(res);
      req.on('close', () => { sseClients.delete(res); });
      return;
    }

    if (req.method === 'POST' && req.url === '/shutdown') {
      // voicebridge is asking us to close this isolated VSCode window.
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true }));
      log('shutdown requested by voicebridge');
      setTimeout(() => {
        try { vscode.commands.executeCommand('workbench.action.quit'); }
        catch (e) { log(`shutdown command failed: ${e.message}`); }
      }, 50);
      return;
    }

    if (req.method === 'POST' && req.url === '/prompt') {
      let body = '';
      req.on('data', chunk => { body += chunk; });
      req.on('end', () => {
        try {
          const obj = JSON.parse(body);
          const text = String(obj.text || '');
          sendPromptToClaude(text);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: true }));
        } catch (e) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: String(e) }));
        }
      });
      return;
    }

    res.writeHead(404);
    res.end();
  });
  httpServer.listen(PORT, '127.0.0.1', () => {
    log(`bridge listening on http://127.0.0.1:${PORT}`);
  });
  httpServer.on('error', err => {
    log(`http server error: ${err.message}`);
  });
}

function activate(context) {
  outputChannel = vscode.window.createOutputChannel('Voice Bridge');
  context.subscriptions.push(outputChannel);
  log('activating voice-bridge');

  const folders = vscode.workspace.workspaceFolders;
  workspaceFsPath = folders && folders[0] ? folders[0].uri.fsPath : null;
  log(`workspace: ${workspaceFsPath}`);

  startHttpServer();
  ensureClaudeTerminal();
  attachTranscriptWatcher();

  context.subscriptions.push(
    vscode.commands.registerCommand('voice-bridge.status', () => {
      vscode.window.showInformationMessage(
        `Voice Bridge :${PORT} | terminal=${claudeTerminal ? 'alive' : 'none'} | transcript=${currentTranscriptPath || '(none)'}`
      );
      outputChannel.show();
    }),
    vscode.commands.registerCommand('voice-bridge.restartClaude', () => {
      if (claudeTerminal) { try { claudeTerminal.dispose(); } catch {} claudeTerminal = null; }
      ensureClaudeTerminal();
    })
  );

  context.subscriptions.push({
    dispose: () => {
      log('deactivating');
      if (watcherCleanup) watcherCleanup();
      if (httpServer) httpServer.close();
      for (const r of sseClients) { try { r.end(); } catch {} }
      sseClients.clear();
    }
  });
}

function deactivate() {}

module.exports = { activate, deactivate };
