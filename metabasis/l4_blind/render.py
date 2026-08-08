"""The blind page. One self-contained HTML string, zero external requests.

Blindness is structural, not cosmetic: this module is handed a `BlindDeck`
and literally cannot render a source label, because a `BlindDeck` does not
carry one. The leak selftest greps the rendered bytes for every column,
model, side, dose and axis token in the population and requires zero hits.

Ergonomics (brief requirement 3 — "the point"):

    1 / ←      pick Text 1              n     note on this pair
    2 / →      pick Text 2              s     session-notes pane
    u          UNSURE (its own state)   b     back one pair
    Enter      advance                  t     toggle side-by-side/stacked
    ?          keyboard help

Every pick POSTs before the page advances; the pill in the header goes
green only after the server has fsync'd the line. A kill at any moment
loses at most the keystroke in flight.
"""
from __future__ import annotations

import json

from metabasis.l4_blind import TOOL_VERSION
from metabasis.l4_blind.bank import normalise_for_display
from metabasis.l4_blind.models import BlindDeck


def _embed(obj: object) -> str:
    """JSON for a <script> block: no '<' survives, so no tag can close early."""
    return (
        json.dumps(obj, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


_CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg:#12131a; --panel:#191b24; --panel2:#1f2230; --ink:#e8e9ef; --dim:#9297ab;
  --line:#2b2f40; --accent:#7aa2f7; --ok:#7fd88f; --warn:#e0af68; --unsure:#bb9af7;
  --pad:14px; --radius:10px;
}
html,body{height:100%;background:var(--bg)}
body{margin:0;height:100dvh;color:var(--ink);
  font:15.5px/1.62 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  display:flex;flex-direction:column;overflow:hidden}
header{flex:0 0 auto;border-bottom:1px solid var(--line);background:var(--panel);
  padding:8px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.brand{font-weight:650;letter-spacing:.2px}
.pill{font:12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;padding:5px 9px;
  border:1px solid var(--line);border-radius:999px;color:var(--dim);white-space:nowrap}
.pill.ok{color:var(--ok);border-color:#2f5a3a}
.pill.busy{color:var(--warn);border-color:#5a4a2a}
.pill.err{color:#ff7a85;border-color:#5a2a30}
.spacer{flex:1}
#bars{flex:0 0 auto;display:flex;gap:14px;padding:9px 16px 10px;
  border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
.bar{flex:1 1 96px;min-width:96px;padding-left:10px;border-left:1px solid var(--line)}
.bar:first-child{padding-left:0;border-left:none;flex:0 0 150px}
.bar .lab{font:11px/1.5 ui-monospace,Menlo,monospace;color:var(--dim);display:flex;
  justify-content:space-between;gap:8px;letter-spacing:.3px}
.bar .lab .n{color:var(--ink);opacity:.75}
.bar .track{height:7px;background:#0d0e14;border:1px solid var(--line);border-radius:4px;
  overflow:hidden;margin-top:4px}
.bar .fill{height:100%;background:var(--accent);width:0;transition:width .18s ease}
.bar.done .fill{background:var(--ok)}
.bar:first-child .lab{color:var(--ink);font-weight:600}
#reveal{flex:0 0 auto;display:none;margin:10px 16px 0;padding:10px 14px;border-radius:8px;
  background:rgba(122,162,247,.10);border:1px solid rgba(122,162,247,.45);font-size:14.5px}
#reveal.on{display:block}
#reveal b{color:var(--accent)}
#reveal .tag{font:11px/1 ui-monospace,Menlo,monospace;color:var(--dim);letter-spacing:.6px;
  display:block;margin-bottom:5px}
.panel.steered{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset}
#qline{flex:0 0 auto;padding:12px 16px 6px;font-size:19px;font-weight:600}
#qline .trait{color:var(--accent)}
main{flex:1 1 auto;display:grid;grid-template-columns:1fr 1fr;gap:12px;
  padding:6px 16px 12px;min-height:0}
main.stacked{grid-template-columns:1fr;grid-template-rows:1fr 1fr}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
  display:flex;flex-direction:column;min-height:0;cursor:pointer;transition:border-color .12s}
.panel:hover{border-color:#3a4160}
.panel.chosen{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset}
.panel h2{margin:0;padding:8px var(--pad);font:12px/1 ui-monospace,Menlo,monospace;
  color:var(--dim);border-bottom:1px solid var(--line);letter-spacing:.6px}
.panel .body{padding:var(--pad);overflow-y:auto;white-space:pre-wrap;flex:1 1 auto;
  font-size:15px;line-height:1.66;overflow-wrap:anywhere}
footer{flex:0 0 auto;border-top:1px solid var(--line);background:var(--panel);
  padding:8px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;
  font:12.5px/1.4 ui-monospace,Menlo,monospace;color:var(--dim)}
kbd{background:var(--panel2);border:1px solid var(--line);border-bottom-width:2px;
  border-radius:5px;padding:1px 6px;font:11.5px/1.5 ui-monospace,Menlo,monospace;color:var(--ink)}
.overlay{position:fixed;inset:0;background:rgba(6,7,11,.72);display:none;
  align-items:center;justify-content:center;padding:24px;z-index:20}
.overlay.on{display:flex}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
  width:min(760px,100%);max-height:82vh;display:flex;flex-direction:column;overflow:hidden}
.card h3{margin:0;padding:12px 16px;border-bottom:1px solid var(--line);font-size:15px}
.card .in{padding:14px 16px;display:flex;flex-direction:column;gap:10px;overflow:auto}
textarea{width:100%;min-height:150px;background:var(--panel2);color:var(--ink);
  border:1px solid var(--line);border-radius:8px;padding:10px;font:14px/1.55 inherit;resize:vertical}
textarea:focus{outline:none;border-color:var(--accent)}
.hint{color:var(--dim);font:12px/1.5 ui-monospace,Menlo,monospace}
.keys{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;align-items:center}
#done{display:none;flex:1;align-items:center;justify-content:center;flex-direction:column;gap:12px;
  text-align:center;padding:40px}
#done.on{display:flex}
#done .big{font-size:26px;font-weight:650}
.badge{font:11px/1 ui-monospace,Menlo,monospace;color:var(--dim);border:1px dashed var(--line);
  border-radius:6px;padding:5px 8px}
"""


_JS = r"""
const DECK = JSON.parse(document.getElementById('deck-data').textContent);
const PAIRS = DECK.pairs, N = PAIRS.length;
const IDX = new Map(PAIRS.map((p,i)=>[p.pair_id,i]));
const NBLIND = PAIRS.filter(p=>!p.revealed_steered).length;
const state = {i:0, verdicts:new Map(), notes:new Map(), shownAt:0, shownIso:null,
               times:[], saving:0, err:0};

const $ = s => document.querySelector(s);
const q = $('#qline'), main = $('#main'), p1 = $('#p1b'), p2 = $('#p2b');
const h1 = $('#p1h'), h2 = $('#p2h');

function fmt(ms){const s=Math.round(ms/1000);return (s<60? s+'s' : Math.floor(s/60)+'m'+String(s%60).padStart(2,'0')+'s');}

function buildBars(){
  const wrap = $('#bars'); wrap.innerHTML='';
  const all = document.createElement('div'); all.className='bar'; all.id='bar-ALL';
  all.innerHTML='<div class="lab"><span>BLIND</span><span class="n"></span></div>'+
                '<div class="track"><div class="fill"></div></div>';
  wrap.appendChild(all);
  for(const s of DECK.set_labels){
    const d=document.createElement('div'); d.className='bar'; d.id='bar-'+s.replace(/\s/g,'_');
    d.innerHTML='<div class="lab"><span>'+s+'</span><span class="n"></span></div>'+
                '<div class="track"><div class="fill"></div></div>';
    wrap.appendChild(d);
  }
}

function counts(){
  const per={}; for(const s of DECK.set_labels) per[s]=0;
  for(const [pid] of state.verdicts){ const p=PAIRS[IDX.get(pid)]; if(p) per[p.set_label]++; }
  return per;
}

function paintBars(){
  const per=counts(), done=state.verdicts.size;
  const setBar=(id,n,t)=>{const el=document.getElementById(id); if(!el)return;
    el.querySelector('.n').textContent=n+'/'+t;
    el.querySelector('.fill').style.width=(t?100*n/t:0)+'%';
    el.classList.toggle('done', n>=t);};
  setBar('bar-ALL', done, NBLIND);
  for(const s of DECK.set_labels){
    if(s==='Calibration'){
      const seen=PAIRS.filter((p,ix)=>p.set_label==='Calibration'&&ix<state.i).length;
      setBar('bar-Calibration', seen, DECK.set_totals[s]);
    } else setBar('bar-'+s.replace(/\s/g,'_'), per[s], DECK.set_totals[s]);
  }
  const med = state.times.length ? [...state.times].sort((a,b)=>a-b)[Math.floor(state.times.length/2)] : 0;
  $('#pace').textContent = 'pace ' + (med? (med/1000).toFixed(1)+'s/pair' : '—') +
                           '  ·  judged ' + done + '/' + NBLIND +
                           '  ·  remaining ' + (med? fmt(med*(NBLIND-done)) : '—');
}

function render(){
  if(state.i>=N){ $('#done').classList.add('on'); main.style.display='none'; q.style.display='none';
                  $('#donecount').textContent = state.verdicts.size+' of '+N+' judged'; paintBars(); return; }
  $('#done').classList.remove('on'); main.style.display=''; q.style.display='';
  const p=PAIRS[state.i];
  q.innerHTML='Which text is more <span class="trait">'+p.trait+'</span>?';
  p1.textContent=p.text_1; p2.textContent=p.text_2;
  p1.scrollTop=0; p2.scrollTop=0;
  const rv=$('#reveal');
  if(p.revealed_steered){
    const which = p.revealed_steered==='text_1' ? 'TEXT 1' : 'TEXT 2';
    rv.className='on';
    rv.innerHTML='<span class="tag">CALIBRATION — NOT SCORED. This is what a real effect looks like.</span>'+
      '<b>'+which+'</b> is the steered generation; the other is the same prompt with no steering. '+
      'The steering pushed it to be <b>'+p.revealed_direction+' '+p.trait+'</b> '+
      '(on-axis score moved '+p.revealed_delta+' of a possible 1.00). '+
      'Read both, then press <b>Enter</b>.';
    $('#phase').textContent='calibration · steered side revealed';
    $('#phase').style.borderColor='rgba(122,162,247,.55)';
    $('#phase').style.color='var(--accent)';
  } else {
    rv.className='';
    $('#phase').textContent='blind · sources sealed';
    $('#phase').style.borderColor=''; $('#phase').style.color='';
  }
  const v=state.verdicts.get(p.pair_id);
  $('#p1').classList.toggle('chosen', v==='text_1');
  $('#p2').classList.toggle('chosen', v==='text_2');
  $('#p1').classList.toggle('steered', p.revealed_steered==='text_1');
  $('#p2').classList.toggle('steered', p.revealed_steered==='text_2');
  h1.textContent='TEXT 1'+(v==='text_1'?'   ✓ chosen':'');
  h2.textContent='TEXT 2'+(v==='text_2'?'   ✓ chosen':'');
  const note=state.notes.get(p.pair_id)||'';
  $('#pos').textContent=(state.i+1)+' / '+N+'  ·  '+(p.revealed_steered?'CALIBRATION':p.set_label)+
      (v? '  ·  recorded: '+(v==='unsure'?'UNSURE':v.replace('_',' ')) : '') +
      (note? '  ·  ✎ note' : '');
  state.shownAt=performance.now();
  state.shownIso=new Date().toISOString().replace(/(\.\d{3})\d*Z?$/, '$1Z');
  paintBars();
}

function pill(cls,txt){const el=$('#save'); el.className='pill '+cls; el.textContent=txt;}

async function post(url,body){
  state.saving++; pill('busy','saving…');
  try{
    const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
                            body:JSON.stringify(body)});
    if(!r.ok) throw new Error('HTTP '+r.status);
    state.err=0; pill('ok','saved');
  }catch(e){ state.err++; pill('err','SAVE FAILED — do not close'); console.error(e); }
  finally{ state.saving--; }
}

function record(choice){
  if(state.i>=N) return;
  const p=PAIRS[state.i];
  if(p.revealed_steered){ return; }   // calibration: anchor, never gold
  const already=state.verdicts.has(p.pair_id);
  const dt=Math.max(0,Math.round(performance.now()-state.shownAt));
  if(!already) state.times.push(dt);
  state.verdicts.set(p.pair_id, choice);
  post('/verdict',{pair_id:p.pair_id, choice:choice, note:state.notes.get(p.pair_id)||'',
                   elapsed_ms:dt, amends:already, shown_at:state.shownIso});
  render();
  if(!$('#noteOverlay').classList.contains('on')) setTimeout(advance,90);
}

function advance(){ if(state.i<N){ state.i++; render(); } }
function back(){ if(state.i>0){ state.i--; render(); } }

function jumpToFirstUnjudged(){
  for(let i=0;i<N;i++){ if(!state.verdicts.has(PAIRS[i].pair_id)){ state.i=i; return; } }
  state.i=N;
}

// ── overlays ──────────────────────────────────────────────────────────
function openNote(){
  if(state.i>=N) return;
  const p=PAIRS[state.i];
  $('#noteText').value=state.notes.get(p.pair_id)||'';
  $('#noteOverlay').classList.add('on'); $('#noteText').focus();
}
function saveNote(){
  const p=PAIRS[state.i]; const t=$('#noteText').value;
  state.notes.set(p.pair_id,t);
  $('#noteOverlay').classList.remove('on');
  if(state.verdicts.has(p.pair_id)){
    post('/verdict',{pair_id:p.pair_id, choice:state.verdicts.get(p.pair_id), note:t,
                     elapsed_ms:null, amends:true, shown_at:state.shownIso});
  }
  render();
}
function openSession(){ $('#sessOverlay').classList.add('on'); $('#sessText').focus(); }
function saveSession(){
  const t=$('#sessText').value.trim();
  $('#sessOverlay').classList.remove('on');
  if(t){ post('/session-note',{note:t}); $('#sessText').value=''; }
}

document.addEventListener('keydown', e=>{
  const inBox = e.target.tagName==='TEXTAREA';
  if(inBox){
    if(e.key==='Escape'){ e.preventDefault();
      $('#noteOverlay').classList.remove('on'); $('#sessOverlay').classList.remove('on'); return; }
    if(e.key==='Enter' && (e.ctrlKey||e.metaKey)){ e.preventDefault();
      if($('#noteOverlay').classList.contains('on')) saveNote(); else saveSession(); }
    return;
  }
  if($('#helpOverlay').classList.contains('on') && e.key!=='?'){
    $('#helpOverlay').classList.remove('on'); if(e.key==='Escape') return; }
  switch(e.key){
    case '1': case 'ArrowLeft':  e.preventDefault(); record('text_1'); break;
    case '2': case 'ArrowRight': e.preventDefault(); record('text_2'); break;
    case 'u': case 'U':          e.preventDefault(); record('unsure'); break;
    case 'n': case 'N':          e.preventDefault(); openNote(); break;
    case 's': case 'S':          e.preventDefault(); openSession(); break;
    case 'b': case 'B': case 'Backspace': e.preventDefault(); back(); break;
    case 'Enter':                e.preventDefault(); advance(); break;
    case 't': case 'T':          e.preventDefault(); main.classList.toggle('stacked'); break;
    case '?':                    e.preventDefault();
                                 $('#helpOverlay').classList.toggle('on'); break;
  }
});

$('#p1').addEventListener('click', ()=>record('text_1'));
$('#p2').addEventListener('click', ()=>record('text_2'));
$('#noteSave').addEventListener('click', saveNote);
$('#sessSave').addEventListener('click', saveSession);
window.addEventListener('beforeunload', e=>{ if(state.saving>0||state.err>0){ e.preventDefault(); e.returnValue=''; } });

// ── boot: resume from the server's append-only log ────────────────────
(async ()=>{
  buildBars();
  try{
    const r = await fetch('/state'); const s = await r.json();
    for(const [pid,v] of Object.entries(s.verdicts||{})) if(IDX.has(pid)) state.verdicts.set(pid,v);
    for(const [pid,t] of Object.entries(s.notes||{}))    if(IDX.has(pid)) state.notes.set(pid,t);
    if(state.verdicts.size) pill('ok','resumed '+state.verdicts.size+'/'+N);
    else pill('','ready');
  }catch(e){ pill('err','could not read session state'); }
  jumpToFirstUnjudged();
  render();
})();
"""


def render_page(deck: BlindDeck, session_id: str) -> str:
    """The whole page, inlined. No <img>, no <link>, no fetch off-origin."""
    payload = {
        "set_labels": deck.set_labels,
        "set_totals": deck.set_totals,
        "pairs": [
            {
                "pair_id": p.pair_id,
                "set_label": p.set_label,
                "trait": p.trait,
                "text_1": normalise_for_display(p.text_1),
                "text_2": normalise_for_display(p.text_2),
                "revealed_steered": p.revealed_steered,
                "revealed_delta": p.revealed_delta,
                "revealed_direction": p.revealed_direction,
            }
            for p in deck.pairs
        ],
    }
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>L4 micro-gold — blind judging</title>
<style>{_CSS}</style></head>
<body>
<header>
  <span class="brand">L4 micro-gold</span>
  <span class="pill" id="pos">—</span>
  <span class="pill" id="pace">pace —</span>
  <span class="spacer"></span>
  <span class="badge" id="phase">blind · sources sealed</span>
  <span class="pill" id="save">ready</span>
</header>
<div id="bars"></div>
<div id="reveal"></div>
<div id="qline">—</div>
<main id="main">
  <section class="panel" id="p1"><h2 id="p1h">TEXT 1</h2><div class="body" id="p1b"></div></section>
  <section class="panel" id="p2"><h2 id="p2h">TEXT 2</h2><div class="body" id="p2b"></div></section>
</main>
<div id="done">
  <div class="big">Deck complete.</div>
  <div class="hint" id="donecount"></div>
  <div class="hint">Leave this tab open and run the <code>seal</code> command at the desk.</div>
</div>
<footer>
  <span><kbd>1</kbd>/<kbd>←</kbd> text 1</span>
  <span><kbd>2</kbd>/<kbd>→</kbd> text 2</span>
  <span><kbd>u</kbd> unsure</span>
  <span><kbd>n</kbd> note</span>
  <span><kbd>s</kbd> session notes</span>
  <span><kbd>b</kbd> back</span>
  <span><kbd>Enter</kbd> advance</span>
  <span><kbd>t</kbd> layout</span>
  <span><kbd>?</kbd> help</span>
  <span class="spacer"></span>
  <span>{TOOL_VERSION} · session {session_id}</span>
</footer>

<div class="overlay" id="noteOverlay"><div class="card">
  <h3>Note on this pair</h3>
  <div class="in">
    <textarea id="noteText" placeholder="What you noticed — the tell, the hesitation, why it was close…"></textarea>
    <div class="hint"><kbd>Ctrl</kbd>+<kbd>Enter</kbd> save &amp; close · <kbd>Esc</kbd> discard</div>
    <div><button id="noteSave">Save note</button></div>
  </div></div></div>

<div class="overlay" id="sessOverlay"><div class="card">
  <h3>Session notes</h3>
  <div class="in">
    <textarea id="sessText" placeholder="Anything about the sitting as a whole — fatigue, a pattern across sets, a question for the desk…"></textarea>
    <div class="hint">Appended as its own record. <kbd>Ctrl</kbd>+<kbd>Enter</kbd> save · <kbd>Esc</kbd> close</div>
    <div><button id="sessSave">Append note</button></div>
  </div></div></div>

<div class="overlay" id="helpOverlay"><div class="card">
  <h3>Keyboard</h3>
  <div class="in"><div class="keys">
    <kbd>1</kbd><span>pick Text 1 (also <kbd>←</kbd>, or click the panel)</span>
    <kbd>2</kbd><span>pick Text 2 (also <kbd>→</kbd>)</span>
    <kbd>u</kbd><span>UNSURE — recorded as its own state, never coerced to a pick</span>
    <kbd>n</kbd><span>note on the current pair</span>
    <kbd>s</kbd><span>session-notes pane</span>
    <kbd>b</kbd><span>back one pair (re-judging appends an amendment)</span>
    <kbd>Enter</kbd><span>advance without recording</span>
    <kbd>t</kbd><span>side-by-side / stacked</span>
  </div>
  <div class="hint">Every pick is written to disk before the page advances. If the
  pill reads SAVE FAILED, stop — the log is the artifact.</div>
  </div></div></div>

<script id="deck-data" type="application/json">{_embed(payload)}</script>
<script>{_JS}</script>
</body></html>
"""
