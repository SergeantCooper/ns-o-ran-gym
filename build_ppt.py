#!/usr/bin/env python3
"""build_ppt.py - generate a presentation (.pptx) covering the whole energy-saving RL
project, embedding the actual result figures. Excludes the reverted masking experiment.

Design system (kept deliberately simple & cohesive):
  - one dark 'navy' for headings/title backdrop, one blue + one teal accent (both
    colour-blind-safe, Okabe-Ito), a warm amber used sparingly, plus panel/ink/muted greys.
  - depth from framed content panels and soft drop-shadows (no flat text-on-white).
  - structure from a section 'kicker' over every title, a footer rule + page numbers,
    and figures mounted on shadowed cards.
Only the styling changed vs. the earlier flat deck; the wording & numbers are unchanged.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml
from PIL import Image
import os, re

# ---- palette ----
INK    = RGBColor(0x1A, 0x22, 0x33)   # body / title ink
NAVY   = RGBColor(0x12, 0x20, 0x3A)   # title backdrop (dark)
NAVY2  = RGBColor(0x21, 0x41, 0x70)   # title backdrop gradient partner
ACC    = RGBColor(0x00, 0x72, 0xB2)   # blue accent (CVD-safe)
ACC2   = RGBColor(0x00, 0x9E, 0x73)   # teal accent (CVD-safe)
AMBER  = RGBColor(0xE6, 0x9F, 0x00)   # warm highlight (sparingly)
MUTE   = RGBColor(0x5B, 0x64, 0x72)   # muted text
PANEL  = RGBColor(0xF3, 0xF6, 0xFB)   # light content panel
PLINE  = RGBColor(0xDD, 0xE4, 0xF0)   # panel / card hairline
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
SUBTXT = RGBColor(0xC9, 0xD6, 0xE5)   # subtitle on dark

def _hex(c): return "%02X%02X%02X" % (c[0], c[1], c[2])

SW, SH = Inches(13.333), Inches(7.5)
TOTAL = 13

prs = Presentation(); prs.slide_width = SW; prs.slide_height = SH
BLANK = prs.slide_layouts[6]
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'

# ---------- low-level styling helpers ----------
def _spPr(shape):
    return shape._element.find(qn('p:spPr'))

def _no_line(shape):
    shape.line.fill.background()

def _shadow(shape, blur=100000, dist=40000, direction=5400000, alpha=66):
    """soft outer drop-shadow for depth."""
    sp = _spPr(shape)
    for el in sp.findall(qn('a:effectLst')):
        sp.remove(el)
    sp.append(parse_xml(
        f'<a:effectLst {A}>'
        f'<a:outerShdw blurRad="{blur}" dist="{dist}" dir="{direction}" rotWithShape="0">'
        f'<a:srgbClr val="000000"><a:alpha val="{alpha*1000}"/></a:srgbClr>'
        f'</a:outerShdw></a:effectLst>'))

def _gradient(shape, c1, c2, ang=5400000):
    """two-stop linear gradient (ang in 1/60000 deg; 5400000 = top->bottom)."""
    shape.fill.solid()
    sp = _spPr(shape)
    solid = sp.find(qn('a:solidFill'))
    grad = parse_xml(
        f'<a:gradFill {A}><a:gsLst>'
        f'<a:gs pos="0"><a:srgbClr val="{_hex(c1)}"/></a:gs>'
        f'<a:gs pos="100000"><a:srgbClr val="{_hex(c2)}"/></a:gs>'
        f'</a:gsLst><a:lin ang="{ang}" scaled="1"/></a:gradFill>')
    sp.replace(solid, grad)

def _solid_alpha(shape, c, alpha_pct):
    shape.fill.solid()
    sp = _spPr(shape)
    solid = sp.find(qn('a:solidFill'))
    solid.clear()
    solid.append(parse_xml(
        f'<a:srgbClr {A} val="{_hex(c)}"><a:alpha val="{int(alpha_pct*1000)}"/></a:srgbClr>'))

def _bullet_indent(p, level):
    base = 0.30 if level == 0 else 0.66
    pPr = p._p.get_or_add_pPr()
    pPr.set('marL', str(int(Inches(base))))
    pPr.set('indent', str(int(-Inches(0.30))))

def rect(slide, l, t, w, h, shape=MSO_SHAPE.RECTANGLE):
    return slide.shapes.add_shape(shape, Inches(l), Inches(t), Inches(w), Inches(h))

def _tb(slide, l, t, w, h):
    tf = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    return tf

def run(p, text, size, color, bold=False, italic=False):
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.color.rgb = color
    r.font.bold = bold; r.font.italic = italic
    return r

def _emit_runs(p, text, size, color):
    """render text, honouring a single **bold** span convention for lead-in labels."""
    for seg in re.split(r'(\*\*[^*]+\*\*)', text):
        if not seg:
            continue
        if seg.startswith('**') and seg.endswith('**'):
            run(p, seg[2:-2], size, color, bold=True)
        else:
            run(p, seg, size, color)

# ---------- slide furniture ----------
def frame(slide):
    """left accent strip — consistent frame on every content slide."""
    strip = rect(slide, 0, 0, 0.16, 7.5)
    _gradient(strip, ACC, ACC2, ang=5400000); _no_line(strip)

def title_bar(slide, title, sub=None, kicker=None):
    frame(slide)
    if kicker:
        tf = _tb(slide, 0.72, 0.36, 10.0, 0.35)
        run(tf.paragraphs[0], kicker.upper(), 12.5, ACC, bold=True)
    tf = _tb(slide, 0.70, 0.62, 12.2, 0.9)
    run(tf.paragraphs[0], title, 28, INK, bold=True)
    ul = rect(slide, 0.72, 1.46, 2.7, 0.075)
    _gradient(ul, ACC, ACC2, ang=0); _no_line(ul)
    if sub:
        tf2 = _tb(slide, 0.72, 1.56, 12.0, 0.5)
        run(tf2.paragraphs[0], sub, 14, MUTE, italic=True)

def footer(slide, n):
    ln = rect(slide, 0.70, 7.04, 11.9, 0.013); ln.fill.solid()
    ln.fill.fore_color.rgb = PLINE; _no_line(ln)
    tf = _tb(slide, 0.70, 7.06, 8.0, 0.35)
    run(tf.paragraphs[0], "Energy-Saving RL  ·  ns-O-RAN digital twin", 9, MUTE)
    tf2 = _tb(slide, 10.6, 7.06, 2.0, 0.35)
    p = tf2.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    run(p, f"{n:02d} / {TOTAL}", 9, MUTE)

def bullets(slide, items, n):
    """content mounted on a soft shadowed panel; colour-coded markers."""
    panel = rect(slide, 0.60, 1.98, 12.13, 4.88, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    panel.adjustments[0] = 0.028
    panel.fill.solid(); panel.fill.fore_color.rgb = PANEL
    panel.line.color.rgb = PLINE; panel.line.width = Pt(1)
    _shadow(panel, blur=90000, dist=32000, alpha=55)
    size = 16 if len(items) >= 7 else 18
    tf = _tb(slide, 1.02, 2.24, 11.35, 4.4)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, (txt, lvl) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        _bullet_indent(p, lvl)
        p.line_spacing = 1.12
        p.space_after = Pt(10 if lvl == 0 else 5)
        if lvl == 0:
            run(p, "▪  ", size, ACC, bold=True)
            _emit_runs(p, txt, size, INK)
        else:
            run(p, "–  ", size - 2, ACC2, bold=True)
            _emit_runs(p, txt, size - 2, MUTE)
    footer(slide, n)

def image(slide, path, n, top=2.15, max_w=11.0, max_h=4.15, caption=None):
    footer(slide, n)
    if not os.path.exists(path):
        return
    with Image.open(path) as im:
        iw, ih = im.size
    ar = iw / ih
    # fit inside the (max_w x max_h) box, preserving aspect ratio
    w_in = min(max_w, max_h * ar)
    h_in = w_in / ar
    left_in = (13.333 - w_in) / 2.0
    pad = 0.14
    card = rect(slide, left_in - pad, top - pad, w_in + 2 * pad, h_in + 2 * pad,
                shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    card.adjustments[0] = 0.02
    card.fill.solid(); card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = PLINE; card.line.width = Pt(1)
    _shadow(card, blur=110000, dist=42000, alpha=60)
    slide.shapes.add_picture(path, Inches(left_in), Inches(top), width=Inches(w_in))
    if caption:
        cy = top + h_in + 2 * pad + 0.08
        tf = _tb(slide, 0.6, cy, 12.1, 0.42)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        run(p, caption, 11, MUTE, italic=True)

def new():
    return prs.slides.add_slide(BLANK)

# ================= 1. Title =================
s = new()
bg = rect(s, 0, 0, 13.333, 7.5)
_gradient(bg, NAVY, NAVY2, ang=3000000); _no_line(bg)
# decorative translucent orbs (depth)
for (l, t, d, al) in [(9.7, -1.6, 5.6, 8), (11.0, 4.6, 3.8, 6)]:
    orb = rect(s, l, t, d, d, shape=MSO_SHAPE.OVAL)
    _solid_alpha(orb, WHITE, al); _no_line(orb)
# bottom accent band
band = rect(s, 0, 6.92, 13.333, 0.58)
_gradient(band, ACC, ACC2, ang=0); _no_line(band)
thin = rect(s, 0, 6.86, 13.333, 0.06); thin.fill.solid()
thin.fill.fore_color.rgb = AMBER; _no_line(thin)
# title text
tf = _tb(s, 0.9, 2.15, 11.6, 2.4)
run(tf.paragraphs[0], "Energy Saving in O-RAN with Reinforcement Learning", 40, WHITE, bold=True)
p2 = tf.add_paragraph(); p2.space_before = Pt(10)
run(p2, "A digital-twin study: learning to sleep 5G cells without hurting service quality", 18, SUBTXT)
# tag pills
pills = [("ns-3 mmWave O-RAN", ACC), ("imitation + PPO", ACC2), ("energy-vs-QoS tradeoff", AMBER)]
x = 0.92
for text, col in pills:
    w = 0.108 * len(text) + 0.55
    pill = rect(s, x, 4.75, w, 0.5, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    pill.adjustments[0] = 0.5
    pill.fill.solid(); pill.fill.fore_color.rgb = col; _no_line(pill)
    _shadow(pill, blur=60000, dist=22000, alpha=50)
    tf = pill.text_frame; tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    pp = tf.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER
    run(pp, text, 12, WHITE, bold=True)
    x += w + 0.24
tf = _tb(s, 0.9, 6.96, 8.0, 0.5)
run(tf.paragraphs[0], "ns-O-RAN digital-twin study  ·  imitation + reinforcement learning", 11, WHITE)

# ================= 2. Problem & goal =================
s = new(); title_bar(s, "The problem & the goal", kicker="Context")
bullets(s, [
 ("5G base stations (gNBs) draw ~600 W even when almost idle — a large, avoidable energy cost.", 0),
 ("Idea: an intelligent controller that puts unused gNBs to sleep, waking them when demand returns.", 0),
 ("The tension: sleep too aggressively and you drop calls / lose throughput (QoS).", 0),
 ("Company goal: a reinforcement-learning (RL) controller that beats the hand-coded heuristic on the "
  "energy-vs-QoS tradeoff — more energy saved without hurting throughput or reliability.", 0),
], 2)

# ================= 3. The digital twin =================
s = new(); title_bar(s, "The digital twin (what we simulate)", kicker="System")
bullets(s, [
 ("ns-3 mmWave O-RAN simulator, scenario-three: 1 LTE anchor cell + 7 mmWave gNBs (can sleep), 6 users.", 0),
 ("Traffic redesigned to 3 sharp bursts per episode (+ an always-on baseline user) — so the controller "
  "must adapt in real time (sleep in the lulls, wake for the bursts).", 0),
 ("Energy model: each ON gNB = 600 W static + 400 W x utilisation; a sleeping gNB ~ 0 W.", 0),
 ("Key constraint: the simulator runs ~15-20 s per 100 ms control step (CPU-bound) — this shapes everything.", 0),
], 3)

# ================= 4. How it works =================
s = new(); title_bar(s, "How it works — real-time closed-loop control", kicker="System")
bullets(s, [
 ("Every 100 ms, a handshake over shared memory (files + semaphores):", 0),
 ("ns-3 emits per-cell KPMs (load, throughput, dropped calls) -> a 61-number observation.", 1),
 ("the controller picks a 7-bit ON/OFF action (one bit per gNB; anchor always ON).", 1),
 ("the action is written back; ns-3 applies it and advances 100 ms.", 1),
 ("Reward the RL optimises:  throughput(Mbps) - energy(kW) - 2 x dropped-calls(RLF).", 0),
 ("The controller plays the role of an O-RAN 'xApp'; we drive the sim directly (no live RIC needed).", 0),
], 4)

# ================= 5. Controllers =================
s = new(); title_bar(s, "The controllers we compared", kicker="Method")
bullets(s, [
 ("Heuristic (TwinHeuristic) - the company baseline: load-adaptive, sleeps idle cells, wakes on neighbour load.", 0),
 ("Behaviour Cloning (BC) - a neural net trained to copy the heuristic (reached 100% match).", 0),
 ("PPO (reinforcement learning) - warm-started from BC, then fine-tuned on the reward. The one that can surpass.", 0),
 ("Pruning controller - an aggressive energy-saver with a tunable 'grace' dial (energy-vs-reliability).", 0),
 ("(ns-3's own built-in heuristic is quota-driven and can't adapt to load - hence the custom one.)", 0),
], 5)

# ================= 6. Pipeline =================
s = new(); title_bar(s, "The learning pipeline", kicker="Method")
bullets(s, [
 ("1)  Collect expert demonstrations - run the heuristic on the twin, log (observation, action) pairs.", 0),
 ("2)  Behaviour Cloning (+ critic warm-up) - clone the heuristic into the policy AND pre-train its value "
  "estimator; the warm-up fixes a classic BC->PPO collapse we hit and diagnosed.", 0),
 ("3)  PPO fine-tune (~4 h, live ns-3 rollouts) - improve on the true reward.", 0),
 ("4)  Evaluate deterministically vs the heuristic across 4 seeds; score energy / throughput / RLF.", 0),
 ("All reproducible; trained model + demos committed to the repo.", 0),
], 6)

# ================= 7. Result 1 =================
s = new(); title_bar(s, "Result 1 — the energy-vs-QoS tradeoff", kicker="Results",
                     sub="Heuristic sits near the efficient frontier; plain-reward PPO ties it byte-for-byte")
image(s, "summary_tradeoff.png", 7, top=2.05, max_w=10.0,
      caption="4-seed averages. PPO reproduces the heuristic (a tie). Aggressive pruning saves more energy but costs reliability.")

# ================= 8. Result 2 =================
s = new(); title_bar(s, "Result 2 — an improved reward moved RL off the tie", kicker="Results",
                     sub="Potential-based reward shaping -> a distinct, greener + more-reliable policy")
image(s, "figures/2_rl_vs_heuristic_by_metric.png", 8, top=2.1, max_w=11.0,
      caption="Shaped-reward RL vs balanced heuristic (4-seed avg): +6 pts energy AND ~25% fewer dropped calls, ~9% less throughput.")

# ================= 9. Why not a strict win =================
s = new(); title_bar(s, "Why RL doesn't strictly dominate (the key insight)", kicker="Analysis")
bullets(s, [
 ("The heuristic is genuinely near-optimal here - it's hard to beat because it's already good.", 0),
 ("RL is sample-starved: the slow sim affords only ~640 trial-steps, and random on/off exploration "
  "sleeps idle AND busy cells together -> can't isolate 'sleeping THIS idle cell was the good move'.", 0),
 ("The 43% 'idle-but-powered' waste is real in hindsight, but NOT free to reclaim: sleeping a "
  "momentarily-idle cell that's needed next causes a dropped call. The heuristic's caution is justified.", 0),
 ("So it's a favourable TRADEOFF (greener + more reliable, slightly less throughput), not a clean sweep.", 0),
], 9)

# ================= 10. Pruning frontier =================
s = new(); title_bar(s, "Operational value — a tunable energy/QoS frontier", kicker="Results",
                     sub="One dial lets an operator choose the operating point; the heuristic is a single fixed point")
image(s, "figures/3_pruning_frontier.png", 10, top=2.1, max_w=9.2,
      caption="Pruning 'grace' dial: from the heuristic's point up to ~57% energy saved, trading throughput as you go.")

# ================= 11. Behaviour over time =================
s = new(); title_bar(s, "What the controller does over time", kicker="Results",
                     sub="Cells sleep through the lulls and wake for the bursts")
image(s, "figures/4_controller_behavior_over_time.png", 11, top=2.1, max_w=10.2,
      caption="One shaped-RL run (seed 999, ~43% saved). RL's 4-seed AVERAGE is ~40% energy saved; "
              "per-seed it varies 25-46%. Idle cells sleep; busy cells stay on.")

# ================= 12. Bottom line =================
s = new(); title_bar(s, "Bottom line & next steps", kicker="Summary")
bullets(s, [
 ("**Delivered:** a realistic bursty twin, a working imitation+RL pipeline (BC 100% match, stable PPO), "
  "a rigorous energy-vs-QoS tradeoff, and a tunable energy-saving controller.", 0),
 ("**RL result:** with reward shaping, RL learns a greener + more-reliable operating point than the "
  "heuristic - better on energy AND dropped-calls, slightly lower throughput: a favourable, honest tradeoff.", 0),
 ("**It does not STRICTLY beat** a strong heuristic on all three axes - the real blocker is the slow "
  "simulator's tiny sample budget (~640 RL trial-steps), too little to out-tune a good heuristic.", 0),
 ("**Next steps:** (1) sample-efficient RL - offline RL on logged data, or a fast learned surrogate of the "
  "sim - to escape that budget;  (2) richer / larger network scenarios;  (3) ship the tunable controller.", 0),
], 12)

# ================= 13. Reproducibility =================
s = new(); title_bar(s, "Reproducibility & deliverables", kicker="Deliverables")
bullets(s, [
 ("Two git repos: ns-3-mmwave-oran (the twin) + ns-o-ran-gym (gym / RL / analysis).", 0),
 ("Committed: trained model + demos + obs-stats; all scripts; figures computed live from run folders.", 0),
 ("Docs (self-contained): OVERVIEW.md (how it all works), RESULTS.md (findings & numbers), "
  "REPRODUCE.md (build & run from scratch), CONTEXT.md (full-journey primer).", 0),
 ("Figures: energy-vs-QoS tradeoff, per-metric comparison, pruning frontier, controller-over-time.", 0),
], 13)

out = "energy_saving_RL_project.pptx"
prs.save(out)
print(f"saved: {out}  ({len(prs.slides._sldIdLst)} slides)")
