"""
gnn_background.py — a live graph-neural-network scene behind the whole UI.

What you see
  * 40-120 nodes drifting slowly, coloured by the project's eight component
    classes (long/short shaft, thick/thin plate, bolt, washer, nut, body),
    linked into a proximity mesh like an assembly graph.
  * "Message passing": glowing pulses hop node -> node -> node, and each hop
    changes colour (cyan -> violet -> magenta) like a 3-layer GNN; every landing
    sends out a ring.
  * A cyan scan band sweeps down the page (scanning a model), lighting nodes.
  * Hovering a node highlights its neighbourhood and fires pulses from it.
  * Hover a node to see its class and embedding bars (and its neighbours').
  * Faint equations the model actually uses (message passing, link predictor, BPR)
    and a small HUD (hop colours + live node/edge/message counts).

How it is wired
  Streamlit cannot run <script> from st.markdown, so a zero-height component
  iframe injects a <script> into the PARENT document (the app already reaches
  the parent the same way for auto-scroll).  That script owns a fixed <canvas>
  at z-index -1 and stays alive across Streamlit reruns; re-injection is a
  no-op unless VERSION/theme change.  The Streamlit containers are made
  transparent in CSS so the canvas shows through, with the sidebar and header
  as frosted glass.

Theme
  Palette follows st.get_option("theme.base") ("dark" or "light").  The rest of
  the UI's inline colours are handled by dark_theme.py.

Accessibility / performance
  * prefers-reduced-motion: one static frame, no animation loop.
  * pointer-events: none, so it never intercepts a click.
  * The node count adapts down if frames get slow.
"""
from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

VERSION = "5"

# name -> palette.  css: page/glass colours.  js: canvas colours (rgb triplets or css colours).
PALETTES = {
    "dark": {
        "css": dict(
            page="linear-gradient(135deg, #070d1b 0%, #0c1830 46%, #071a2a 100%)",
            header_bg="rgba(9,17,32,0.66)", header_line="rgba(56,189,248,0.30)", header_shadow="0 10px 40px rgba(0,0,0,0.45)",
            sidebar_bg="rgba(9,17,32,0.62)", sidebar_line="rgba(56,189,248,0.18)", topbar_bg="rgba(9,17,32,0.45)",
            halo="rgba(6,10,20,0.95)",
        ),
        "js": dict(
            aurora=[[0.16, 0.20, "34,211,238", 0.13], [0.84, 0.30, "139,92,246", 0.17], [0.52, 0.90, "59,130,246", 0.15]],
            dot="rgba(148,163,184,0.13)",
            edge="96,165,250", edgeA=[0.06, 0.26], edgeHot="rgba(34,211,238,0.85)",
            scan="34,211,238", scanA=0.11, scanLine="rgba(34,211,238,0.38)",
            label="rgba(148,163,184,0.72)", eq="rgba(148,163,184,0.45)", hud="rgba(148,163,184,0.62)",
            core="rgba(255,255,255,0.95)", nodeA=0.78, glow="lighter", body="#94a3b8",
        ),
    },
    "light": {
        "css": dict(
            page="linear-gradient(135deg, #e8f1ff 0%, #f3eeff 46%, #e3fbff 100%)",
            header_bg="rgba(255,255,255,0.74)", header_line="rgba(59,130,246,0.26)", header_shadow="0 10px 34px rgba(59,130,246,0.12)",
            sidebar_bg="rgba(255,255,255,0.62)", sidebar_line="rgba(59,130,246,0.20)", topbar_bg="rgba(255,255,255,0.55)",
            halo="rgba(255,255,255,0.95)",
        ),
        "js": dict(
            aurora=[[0.18, 0.22, "34,211,238", 0.16], [0.82, 0.30, "139,92,246", 0.14], [0.55, 0.88, "59,130,246", 0.14]],
            dot="rgba(100,116,139,0.10)",
            edge="59,130,246", edgeA=[0.05, 0.25], edgeHot="rgba(6,182,212,0.65)",
            scan="6,182,212", scanA=0.10, scanLine="rgba(6,182,212,0.30)",
            label="rgba(71,85,105,0.55)", eq="rgba(51,65,85,0.34)", hud="rgba(51,65,85,0.5)",
            core="rgba(255,255,255,0.92)", nodeA=0.55, glow="source-over", body="#64748b",
        ),
    },
}


def _theme() -> str:
    try:
        return "light" if str(st.get_option("theme.base")).lower() == "light" else "dark"
    except Exception:
        return "dark"


def _css(p: dict) -> str:
    return f"""<!--nodark-->
<style>
/* ---- futuristic GNN backdrop: let the canvas (z-index -1) show through ---- */
html {{ background: {p['page']} fixed !important; min-height: 100%; }}
body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {{ background: transparent !important; }}
[data-testid="stHeader"] {{ background: {p['topbar_bg']} !important; backdrop-filter: blur(8px); }}
/* frosted-glass sidebar */
section[data-testid="stSidebar"] {{
    background: {p['sidebar_bg']} !important;
    backdrop-filter: blur(14px) saturate(140%); -webkit-backdrop-filter: blur(14px) saturate(140%);
    border-right: 1px solid {p['sidebar_line']};
}}
section[data-testid="stSidebar"] > div:first-child {{ background: transparent !important; }}
/* frosted-glass project header with a moving light hairline */
#proj-header {{
    left: var(--gnn-sidebar-w, 300px) !important;   /* kept in sync with the real sidebar edge by the injected script */
    background: {p['header_bg']} !important;
    backdrop-filter: blur(14px) saturate(150%); -webkit-backdrop-filter: blur(14px) saturate(150%);
    border-bottom: 1px solid {p['header_line']} !important;
    box-shadow: {p['header_shadow']} !important;
}}
#proj-header::after {{
    content: ""; position: absolute; left: 0; right: 0; bottom: -1px; height: 2px;
    background: linear-gradient(90deg, transparent, #06b6d4, #8b5cf6, #ec4899, transparent);
    background-size: 200% 100%; animation: gnnSweep 6s linear infinite; opacity: .9;
}}
@keyframes gnnSweep {{ 0% {{ background-position: 200% 0; }} 100% {{ background-position: -200% 0; }} }}
@media (prefers-reduced-motion: reduce) {{ #proj-header::after {{ animation: none; }} }}
/* soft halo so body text stays crisp when a node or edge drifts behind it */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] label {{ text-shadow: 0 0 5px {p['halo']}, 0 0 2px {p['halo']}; }}
</style>
"""


# Runs inside the component iframe; builds a script that executes in the parent page.
_INJECTOR = r"""
<script>
(function () {
  var W = window.parent, D = W.document, VERSION = "__VERSION__", PAL = __PAL__;
  if (W.__gnnBg) {
    if (W.__gnnBg.version === VERSION) return;          // already running this version/theme
    try { W.__gnnBg.stop(); } catch (e) {}
  }

  function main(W, D, VERSION, PAL) {
    var CLASSES = [["long_shaft","#3b82f6"],["short_shaft","#06b6d4"],["thick_plate","#8b5cf6"],["thin_plate","#ec4899"],
                   ["bolt","#f59e0b"],["washer","#10b981"],["nut","#ef4444"],["body",PAL.body]];
    var HOP = ["#06b6d4", "#6366f1", "#d946ef"];          // layer 1 / 2 / 3 message colours
    var reduce = W.matchMedia && W.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var DPR = Math.min(W.devicePixelRatio || 1, 1.5);
    var MONO = "'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace";

    var cv = D.createElement("canvas");
    cv.id = "gnn-bg"; cv.setAttribute("aria-hidden", "true");
    cv.style.cssText = "position:fixed;left:0;top:0;width:100vw;height:100vh;z-index:-1;pointer-events:none;";
    D.body.insertBefore(cv, D.body.firstChild);
    var ctx = cv.getContext("2d");

    var sbw = 300;                                        // sidebar right edge (px), kept current by syncSidebar()
    var w = 0, h = 0, LINK = 170, nodes = [], adj = [], edges = [], pulses = [], rings = [];
    var grid = D.createElement("canvas");
    var mouse = { x: -9999, y: -9999 }, hover = -1;
    var t0 = W.performance.now(), last = t0, raf = 0, spawnAcc = 0, slow = 0, stopped = false, labelIdx = [];

    function rnd(a, b) { return a + Math.random() * (b - a); }
    function makeNode() {
      var c = (Math.random() * CLASSES.length) | 0, sp = rnd(6, 16), an = rnd(0, 6.283);
      return { x: rnd(0, w), y: rnd(0, h), vx: Math.cos(an) * sp, vy: Math.sin(an) * sp, r: rnd(2.6, 5.4),
               c: c, ph: rnd(0, 6.283), flash: 0 };
    }
    function targetCount() { return Math.max(36, Math.min(120, Math.round(w * h / 20000))); }

    function buildGrid() {
      grid.width = w; grid.height = h;
      var g = grid.getContext("2d"); g.fillStyle = PAL.dot;
      for (var x = 22; x < w; x += 44) for (var y = 22; y < h; y += 44) g.fillRect(x, y, 1.6, 1.6);
    }
    function resize() {
      w = W.innerWidth; h = W.innerHeight;
      cv.width = Math.round(w * DPR); cv.height = Math.round(h * DPR);
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      LINK = Math.max(120, Math.min(190, w * 0.11));
      buildGrid();
      var n = targetCount();
      while (nodes.length < n) nodes.push(makeNode());
      if (nodes.length > n) nodes.length = n;
      nodes.forEach(function (o) { if (o.x > w) o.x = rnd(0, w); if (o.y > h) o.y = rnd(0, h); });
      labelIdx = [];
      for (var i = 0; i < nodes.length && labelIdx.length < 14; i += 6) labelIdx.push(i + 2);
      if (reduce) { rebuildEdges(); frame(W.performance.now()); }
    }

    function rebuildEdges() {
      edges.length = 0; adj.length = nodes.length;
      for (var i = 0; i < nodes.length; i++) adj[i] = [];
      for (i = 0; i < nodes.length; i++) {
        for (var j = i + 1; j < nodes.length; j++) {
          var dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y, d2 = dx * dx + dy * dy;
          if (d2 < LINK * LINK) {
            var d = Math.sqrt(d2);
            if (adj[i].length < 5 && adj[j].length < 5) { edges.push([i, j, 1 - d / LINK]); adj[i].push(j); adj[j].push(i); }
          }
        }
      }
    }

    function spawn(from, hop, prev) {
      var nb = adj[from]; if (!nb || !nb.length) return;
      var to = nb[(Math.random() * nb.length) | 0];
      if (to === prev && nb.length > 1) to = nb[(nb.indexOf(to) + 1) % nb.length];
      pulses.push({ a: from, b: to, t: 0, dur: rnd(0.7, 1.3), hop: hop });
    }

    function step(dt, t) {
      for (var i = 0; i < nodes.length; i++) {
        var o = nodes[i];
        o.x += (o.vx + Math.sin(t * 0.0004 + o.ph) * 3) * dt; o.y += (o.vy + Math.cos(t * 0.00035 + o.ph) * 3) * dt;
        if (o.x < -20) o.x = w + 20; else if (o.x > w + 20) o.x = -20;
        if (o.y < -20) o.y = h + 20; else if (o.y > h + 20) o.y = -20;
        var mx = mouse.x - o.x, my = mouse.y - o.y, md = mx * mx + my * my;
        if (md < 150 * 150 && md > 1) { o.x += mx * 0.15 * dt; o.y += my * 0.15 * dt; }     // gentle pull toward the cursor
        if (o.flash > 0) o.flash = Math.max(0, o.flash - dt * 1.6);
      }
      rebuildEdges();
      hover = -1; var best = 46 * 46;
      for (i = 0; i < nodes.length; i++) {
        var ex = nodes[i].x - mouse.x, ey = nodes[i].y - mouse.y, ed = ex * ex + ey * ey;
        if (ed < best) { best = ed; hover = i; }
      }
      spawnAcc += dt;
      if (spawnAcc > 0.28 && pulses.length < 26) {
        spawnAcc = 0;
        spawn(hover >= 0 && Math.random() < 0.6 ? hover : (Math.random() * nodes.length) | 0, 0, -1);
      }
      for (i = pulses.length - 1; i >= 0; i--) {
        var p = pulses[i]; p.t += dt / p.dur;
        if (p.t >= 1) {
          nodes[p.b].flash = 1; rings.push({ x: nodes[p.b].x, y: nodes[p.b].y, t: 0, hop: p.hop });
          if (p.hop < 2 && Math.random() < 0.85) spawn(p.b, p.hop + 1, p.a);
          pulses.splice(i, 1);
        }
      }
      for (i = rings.length - 1; i >= 0; i--) { rings[i].t += dt / 0.9; if (rings[i].t >= 1) rings.splice(i, 1); }
    }

    function frame(now) {
      var t = now - t0, ph = t / 1000;
      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = "source-over";
      // aurora: three slow soft light blobs
      var R = Math.max(w, h) * 0.46, bl = PAL.aurora;
      for (var b = 0; b < bl.length; b++) {
        var bx = w * (bl[b][0] + Math.sin(ph * 0.05 + b * 2.1) * 0.06), by = h * (bl[b][1] + Math.cos(ph * 0.043 + b) * 0.06);
        var g = ctx.createRadialGradient(bx, by, 0, bx, by, R);
        g.addColorStop(0, "rgba(" + bl[b][2] + "," + bl[b][3] + ")"); g.addColorStop(1, "rgba(" + bl[b][2] + ",0)");
        ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
      }
      ctx.drawImage(grid, 0, 0);
      // scan band
      var sy = ((ph / 9) % 1.25 - 0.12) * h;
      var sg = ctx.createLinearGradient(0, sy - 90, 0, sy + 90);
      sg.addColorStop(0, "rgba(" + PAL.scan + ",0)"); sg.addColorStop(0.5, "rgba(" + PAL.scan + "," + PAL.scanA + ")"); sg.addColorStop(1, "rgba(" + PAL.scan + ",0)");
      ctx.fillStyle = sg; ctx.fillRect(0, sy - 90, w, 180);
      ctx.fillStyle = PAL.scanLine; ctx.fillRect(0, sy - 0.5, w, 1);

      // edges
      var i, e, a, bN, hv = hover;
      for (i = 0; i < edges.length; i++) {
        e = edges[i]; a = nodes[e[0]]; bN = nodes[e[1]];
        var hot = hv >= 0 && (e[0] === hv || e[1] === hv);
        ctx.strokeStyle = hot ? PAL.edgeHot : "rgba(" + PAL.edge + "," + (PAL.edgeA[0] + e[2] * (PAL.edgeA[1] - PAL.edgeA[0])).toFixed(3) + ")";
        ctx.lineWidth = hot ? 1.7 : 1;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(bN.x, bN.y); ctx.stroke();
      }
      // pulses (comet: short tail + glow)
      for (i = 0; i < pulses.length; i++) {
        var p = pulses[i], A = nodes[p.a], B = nodes[p.b], x = A.x + (B.x - A.x) * p.t, y = A.y + (B.y - A.y) * p.t;
        var tx = A.x + (B.x - A.x) * Math.max(0, p.t - 0.22), ty = A.y + (B.y - A.y) * Math.max(0, p.t - 0.22);
        var col = HOP[p.hop];
        var lg = ctx.createLinearGradient(tx, ty, x, y); lg.addColorStop(0, col + "00"); lg.addColorStop(1, col);
        ctx.strokeStyle = lg; ctx.lineWidth = 2.4; ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(x, y); ctx.stroke();
        ctx.save(); ctx.shadowColor = col; ctx.shadowBlur = 14; ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(x, y, 2.8, 0, 6.283); ctx.fill(); ctx.restore();
      }
      // rings (aggregation waves)
      for (i = 0; i < rings.length; i++) {
        var rg = rings[i]; ctx.strokeStyle = HOP[rg.hop] + Math.round((1 - rg.t) * 170).toString(16).padStart(2, "0");
        ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(rg.x, rg.y, 5 + rg.t * 30, 0, 6.283); ctx.stroke();
      }
      // nodes (halos are additive on the dark theme so they read as light)
      for (i = 0; i < nodes.length; i++) {
        var o = nodes[i], col2 = CLASSES[o.c][1];
        var near = Math.max(0, 1 - Math.abs(o.y - sy) / 90);                // lit by the scan band
        var isNb = hv >= 0 && adj[hv] && adj[hv].indexOf(i) >= 0;
        var lift = Math.max(o.flash, near * 0.8, i === hv ? 1 : 0, isNb ? 0.6 : 0);
        var pulse = 1 + Math.sin(ph * 1.4 + o.ph) * 0.12;
        if (lift > 0.02) {
          ctx.globalCompositeOperation = PAL.glow;
          var hg = ctx.createRadialGradient(o.x, o.y, 0, o.x, o.y, 20 + lift * 10);
          hg.addColorStop(0, col2 + "77"); hg.addColorStop(1, col2 + "00");
          ctx.fillStyle = hg; ctx.beginPath(); ctx.arc(o.x, o.y, 20 + lift * 10, 0, 6.283); ctx.fill();
          ctx.globalCompositeOperation = "source-over";
        }
        ctx.fillStyle = col2; ctx.globalAlpha = Math.min(1, PAL.nodeA + lift * 0.4);
        ctx.beginPath(); ctx.arc(o.x, o.y, o.r * pulse * (1 + lift * 0.4), 0, 6.283); ctx.fill(); ctx.globalAlpha = 1;
        ctx.fillStyle = PAL.core; ctx.beginPath(); ctx.arc(o.x, o.y, o.r * 0.36, 0, 6.283); ctx.fill();
      }
      // labels + embedding sparklines on a few nodes
      ctx.font = "11px " + MONO; ctx.textBaseline = "middle"; ctx.textAlign = "left";
      // annotations (class + embedding bars) appear only for the hovered node and its neighbours, so the
      // background stays clean behind the text unless you point at it
      ctx.font = "11px " + MONO; ctx.textBaseline = "middle"; ctx.textAlign = "left";
      if (hv >= 0 && adj[hv]) {
        var show = [hv].concat(adj[hv]);
        for (i = 0; i < show.length; i++) {
          var L = nodes[show[i]]; if (!L) continue;
          ctx.globalAlpha = i === 0 ? 1 : 0.7;
          ctx.fillStyle = PAL.label; ctx.fillText(CLASSES[L.c][0], L.x + 11, L.y - 9);
          for (var k = 0; k < 5; k++) {
            var bh = 3 + (Math.sin(ph * 1.6 + L.ph * 3 + k * 1.3) * 0.5 + 0.5) * 9;
            ctx.fillStyle = CLASSES[L.c][1] + "c0"; ctx.fillRect(L.x + 11 + k * 4, L.y - 1 + 10 - bh, 2.4, bh);
          }
        }
        ctx.globalAlpha = 1;
      }
      // faint equations (what the model actually computes), stacked bottom-left of the main area
      ctx.font = "12px " + MONO; ctx.fillStyle = PAL.eq; ctx.textBaseline = "middle"; ctx.textAlign = "left";
      var eq = [
        "h\u1d65\u207d\u02e1\u207a\u00b9\u207e = \u03c3( \u03a3\u1d64\u2208N(v) \u03b1\u1d65\u1d64 \u00b7 W h\u1d64\u207d\u02e1\u207e )",
        "\u0177\u1d64\u1d65 = \u03c3( MLP[ h\u1d64 \u2016 h\u1d65 \u2016 d\u1d64\u1d65 ] )",
        "L = \u2212\u03a3 log \u03c3( s\u207a \u2212 s\u207b )    (BPR)"
      ];
      for (i = 0; i < eq.length; i++) ctx.fillText(eq[i], sbw + 30, h - 70 + i * 20);
      // HUD legend: message-passing hops + live graph size
      var hx = w - 330, hy = h - 46;
      ctx.font = "10.5px " + MONO; ctx.fillStyle = PAL.hud;
      ctx.fillText("MESSAGE PASSING", hx, hy);
      for (i = 0; i < 3; i++) {
        ctx.fillStyle = HOP[i]; ctx.beginPath(); ctx.arc(hx + 122 + i * 62, hy, 3.4, 0, 6.283); ctx.fill();
        ctx.fillStyle = PAL.hud; ctx.fillText("hop " + (i + 1), hx + 130 + i * 62, hy);
      }
      ctx.fillText("nodes " + nodes.length + "  ·  edges " + edges.length + "  ·  msgs " + pulses.length, hx, hy + 16);
    }

    function loop(now) {
      if (stopped) return;
      var dt = Math.min(0.05, (now - last) / 1000); last = now;
      slow = slow * 0.95 + (dt * 1000) * 0.05;
      if (slow > 34 && nodes.length > 40) { nodes.length = Math.max(36, Math.floor(nodes.length * 0.9)); slow = 16; }
      step(dt, now - t0); frame(now);
      raf = W.requestAnimationFrame(loop);
    }

    function syncSidebar() {
      var sb = D.querySelector('section[data-testid="stSidebar"]');
      var r = sb ? Math.max(0, Math.round(sb.getBoundingClientRect().right)) : 0;
      sbw = r; D.documentElement.style.setProperty("--gnn-sidebar-w", r + "px");
    }
    var sbTimer = W.setInterval(syncSidebar, 400); syncSidebar();

    function onMove(ev) { mouse.x = ev.clientX; mouse.y = ev.clientY; }
    function onLeave() { mouse.x = mouse.y = -9999; }
    W.addEventListener("resize", resize);
    if (!reduce) { W.addEventListener("mousemove", onMove, { passive: true }); D.addEventListener("mouseleave", onLeave); }

    resize(); rebuildEdges();
    if (reduce) { frame(W.performance.now()); } else { raf = W.requestAnimationFrame(loop); }

    W.__gnnBg = {
      version: VERSION,
      nodes: nodes,                                   // read-only view (used by tests / debugging)
      stop: function () {
        stopped = true; W.cancelAnimationFrame(raf); W.clearInterval(sbTimer);
        W.removeEventListener("resize", resize); W.removeEventListener("mousemove", onMove); D.removeEventListener("mouseleave", onLeave);
        if (cv.parentNode) cv.parentNode.removeChild(cv);
        var s = D.getElementById("gnn-bg-script"); if (s && s.parentNode) s.parentNode.removeChild(s);
        delete W.__gnnBg;
      }
    };
  }

  var old = D.getElementById("gnn-bg-script"); if (old) old.remove();
  var s = D.createElement("script"); s.id = "gnn-bg-script";
  s.textContent = "(" + main.toString() + ")(window, document, " + JSON.stringify(VERSION) + ", " + JSON.stringify(PAL) + ");";
  D.head.appendChild(s);
})();
</script>
"""


def inject() -> None:
    """Call once per script run, right after st.set_page_config()."""
    theme = _theme()
    pal = PALETTES[theme]
    st.markdown(_css(pal["css"]), unsafe_allow_html=True)
    version = f"{VERSION}-{theme}"
    components.html(
        _INJECTOR.replace("__VERSION__", version).replace("__PAL__", json.dumps(pal["js"])),
        height=0,
    )
