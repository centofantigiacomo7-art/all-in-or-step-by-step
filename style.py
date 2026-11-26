"""Visual identity, "Maison" edition: white, true black, one bronze accent, Bodoni Moda over Jost."""
import html

import plotly.graph_objects as go
import plotly.io as pio

INK, MUTED, LINE, PAPER = "#000000", "#6F6A64", "#E6E2DC", "#FFFFFF"
ACCENT = "#8C6A3F"                       # bronze: the step-by-step plan, everywhere
ALL_NOW, STEP = INK, ACCENT
GOOD, BAD = ACCENT, INK                  # history: bronze = waiting ahead, black = all now ahead
SERIF = "'Bodoni Moda', Georgia, serif"
SANS = "'Jost', 'Helvetica Neue', Helvetica, Arial, sans-serif"
FONT = SANS
FAINT = LINE

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bodoni+Moda:ital,opsz,wght@0,6..96,400;1,6..96,400&family=Jost:wght@300;400&display=swap');

html, body, .stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, label, input, textarea,
[data-testid="stExpander"] summary p, [data-testid="stCaptionContainer"] p, [data-baseweb="select"] div {{
  font-family: {SANS}; font-weight: 300; font-feature-settings: "lnum" 1;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ max-width: 1200px; padding: 2.2rem 3rem 5rem; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stSidebarCollapsedControl"],
section[data-testid="stSidebar"] {{ display: none !important; }}

.stMarkdown p:not([class]), .stMarkdown li {{ font-size: 17px; line-height: 1.6; color: {INK}; text-wrap: pretty; }}
.stMarkdown p {{ text-wrap: pretty; }}
[data-testid="stCaptionContainer"] p {{ color: {MUTED} !important; font-size: 13px !important; line-height: 1.5 !important; margin: 0 0 4px; }}
h1, h2 {{ text-wrap: balance; }}

/* widgets: hairline fields, black controls */
label p, [data-testid="stWidgetLabel"] p {{ color: {MUTED} !important; font-size: 13px !important; font-weight: 300 !important; }}
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div {{
  background: transparent !important; border: none !important; border-bottom: 1px solid {INK} !important; border-radius: 0 !important; }}
[data-baseweb="input"] input, [data-baseweb="select"] div, [data-testid="stSelectbox"] input {{ font-size: 18px !important; color: {INK}; padding-left: 0; }}
[data-testid="stNumberInputContainer"], [data-testid="stSelectbox"] > div:last-child > div {{
  border: none !important; border-bottom: 1px solid {INK} !important; border-radius: 0 !important; background: transparent !important; }}
input, [data-baseweb="select"] *, [data-testid="stSelectbox"] input, [data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"], [data-testid="stRadio"] label p, .stDownloadButton button p {{ font-family: {SANS} !important; font-weight: 300; }}
.stDownloadButton button {{ border: 1px solid {INK}; border-radius: 0; background: transparent; }}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{ display: none; }}

/* expanders read as quiet links */
[data-testid="stExpander"] details {{ border: none; border-top: 1px solid {LINE}; border-radius: 0; background: transparent; }}
[data-testid="stExpander"] summary {{ padding-left: 0; }}
[data-testid="stExpander"] summary p {{ font-size: 15px; color: {INK}; }}
[data-testid="stExpander"] summary svg {{ color: {MUTED}; }}
[data-testid="stDataFrame"] {{ border: none; }}

/* page parts */
.top {{ display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 16px; margin: 0 0 88px; }}
.top .mark {{ font-family: {SERIF}; font-style: italic; font-size: 26px; color: {INK}; }}
.top nav {{ display: flex; gap: 8px 28px; flex-wrap: wrap; font-size: 14px; font-family: {SANS}; }}
.top nav a {{ color: {MUTED}; text-decoration: none; font-family: {SANS}; font-weight: 300; }}
.top nav a:hover {{ color: {INK}; }}

.stMarkdown p.sentence {{ font-size: clamp(20px, 2.2vw, 28px) !important; line-height: 1.9 !important; max-width: 900px; margin: 0 0 28px; color: {INK}; }}
.sentence .v {{ border-bottom: 1px solid {INK}; padding-bottom: 2px; white-space: nowrap; }}
.sentence .nw {{ white-space: nowrap; }}

h1.verdict {{ font-family: {SERIF} !important; font-weight: 400; font-size: clamp(64px, 10.5vw, 150px); line-height: 0.92;
  letter-spacing: -0.025em; margin: 56px 0 0; padding: 0; color: {INK}; }}

.strip {{ margin: 88px 0 0; }}
.strip p {{ margin: 0 0 12px; font-size: 14px; }}
.strip .line {{ height: 2px; background: {INK}; }}
.strip .seg {{ display: flex; gap: 6px; }}
.strip .seg div {{ flex: 1; height: 2px; }}
.strip .ticks {{ display: flex; justify-content: space-between; margin-top: 14px; font-size: 12px; color: {MUTED}; }}

.figures {{ margin: 104px 0 0; display: flex; flex-wrap: wrap; gap: 48px 96px; align-items: flex-start; }}
.figures > div {{ flex: 1 1 380px; }}
.figures .big {{ margin: 0; font-family: {SERIF}; font-size: clamp(56px, 7vw, 96px); line-height: 1; color: {INK}; }}
.figures .say {{ margin: 16px 0 0; max-width: 440px; color: {MUTED}; }}

table.detail {{ width: 100%; border-collapse: collapse; margin: 120px 0 0; font-size: 16px; font-variant-numeric: tabular-nums; font-family: {SANS};
  border: none !important; border-top: 1px solid {INK} !important; }}
table.detail tr, table.detail thead {{ background: transparent !important; border: none !important; }}
table.detail th, table.detail td {{ padding: 16px 0 16px 16px; border: none !important; border-top: 1px solid {LINE} !important;
  text-align: right; white-space: nowrap; font-weight: 300; background: transparent !important; color: {INK};
  font-family: {SANS} !important; font-size: 16px; }}
table.detail thead th {{ border-top: none !important; font-weight: 400; }}
table.detail th:first-child, table.detail td:first-child {{ padding-left: 0; text-align: left; color: {MUTED}; white-space: normal; }}
table.detail tbody tr:last-child td {{ border-bottom: 1px solid {LINE} !important; }}

.section {{ margin: 150px 0 0; }}
#method {{ margin-bottom: 28px; }}
.section h2 {{ font-family: {SERIF} !important; font-weight: 400; font-size: clamp(36px, 4vw, 52px); line-height: 1.1;
  max-width: 780px; margin: 0; padding: 0; color: {INK}; }}
.section > p {{ margin: 24px 0 0; max-width: 640px; }}
.stMarkdown p.note {{ font-size: 14px; color: {MUTED}; line-height: 1.55; }}
.stMarkdown p.result {{ font-size: 17px; line-height: 1.6; margin: 8px 0 0; color: {INK}; }}
.stMarkdown p.result b {{ font-weight: 400; font-family: {SERIF}; font-size: 22px; }}

.scale {{ position: relative; margin: 88px 0 0; height: 120px; }}
.scale div, .scale p {{ position: absolute; margin: 0; }}
.scale p {{ font-size: 14px; }}
.scale .tick {{ font-size: 12px; transform: translateX(-50%); white-space: nowrap; top: 88px; }}

.bars {{ display: flex; height: 280px; margin: 72px 0 0; }}
.bars > div {{ flex: 1 1 0; min-width: 0; display: flex; flex-direction: column; }}
.bars .up {{ height: 140px; display: flex; flex-direction: column; justify-content: flex-end; }}
.bars .down {{ height: 140px; border-top: 1px solid {INK}; box-sizing: border-box; }}
.bars .up div, .bars .down div {{ width: max(1px, calc(100% - 2px)); }}
.years {{ display: flex; justify-content: space-between; margin-top: 14px; font-size: 12px; color: {MUTED}; }}

div.quote {{ margin: 170px auto 40px; max-width: 880px; text-align: center; border: none; padding: 0;
  font-family: {SERIF}; font-style: italic; font-size: clamp(28px, 3.2vw, 40px); line-height: 1.35; color: {INK} !important; text-wrap: balance; }}
div.quote * {{ color: {INK} !important; }}

.stMarkdown p.foot {{ line-height: 1.6; }}
.foot {{ margin: 120px 0 0; padding: 32px 0 0; border-top: 1px solid {LINE}; font-size: 14px; color: {MUTED}; }}
:focus-visible {{ outline: 2px solid {INK}; outline-offset: 2px; }}
</style>
"""


def register_plotly_template() -> None:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=SANS, size=13, color=INK),
        paper_bgcolor=PAPER, plot_bgcolor=PAPER, colorway=[INK, ACCENT, MUTED],
        xaxis=dict(showgrid=False, linecolor=LINE, ticks="", zeroline=False, tickfont=dict(color=MUTED),
                   title=dict(font=dict(size=12, color=MUTED))),
        yaxis=dict(gridcolor=LINE, zeroline=False, showline=False, tickfont=dict(color=MUTED),
                   title=dict(font=dict(size=12, color=MUTED))),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=12, color=MUTED)),
        margin=dict(l=48, r=8, t=36, b=48),
        hoverlabel=dict(bgcolor=PAPER, bordercolor=LINE, font=dict(family=SANS, color=INK)),
    )
    pio.templates["windfall"] = t
    pio.templates.default = "windfall"


e = html.escape


def top() -> str:
    return ("<div class='top'><span class='mark'>Windfall</span><nav><a href='#who'>Who should wait</a>"
            "<a href='#past'>The past</a><a href='#crash'>A crash</a><a href='#method'>Method</a></nav></div>")


def sentence(amount: str, months: str, years: str, attitude: str) -> str:
    return (f"<p class='sentence'>I received <span class='v'>{e(amount)}</span> and could spread it "
            f"<span class='nw'>over <span class='v'>{e(months)}</span>,</span> to keep "
            f"<span class='nw'>for <span class='v'>{e(years)}</span>,</span> with "
            f"<span class='nw'>a <span class='v'>{e(attitude)}</span> attitude</span> to risk.</p>")


def verdict(line1: str, line2: str, kind: str) -> str:
    return f"<h1 class='verdict verdict-{kind}'>{e(line1)}<br>{e(line2)}</h1>"


def strip(per: str, N: int, shown: int) -> str:
    segs = "".join(f"<div style='background:{ACCENT if k < N else LINE}'></div>" for k in range(shown))
    mid = shown // 3
    return (f"<figure class='strip' aria-label='All now versus {N} monthly instalments'>"
            f"<p>All of it, today</p><div class='line'></div>"
            f"<p style='margin-top:30px;color:{ACCENT}'>{e(per)} at the start of each month, for {N} months</p>"
            f"<div class='seg'>{segs}</div>"
            f"<div class='ticks'><span>Today</span><span>{mid} months</span><span>{2 * mid} months</span>"
            f"<span>{shown} months</span></div></figure>")


def figures(items: list[tuple[str, str]]) -> str:
    return "<div class='figures'>" + "".join(
        f"<div><p class='big'>{e(big)}</p><p class='say'>{e(say)}</p></div>" for big, say in items) + "</div>"


def detail(title: str, rows: list[tuple[str, str, str]], step_label: str) -> str:
    head = (f"<thead><tr><th>{e(title)}</th><th>All now</th>"
            f"<th style='color:{ACCENT};white-space:normal'>{e(step_label)}</th></tr></thead>")
    body = "".join(f"<tr><td>{l}</td><td>{e(a)}</td><td>{e(b)}</td></tr>" for l, a, b in rows)
    return f"<table class='detail'>{head}<tbody>{body}</tbody></table>"


def section_head(anchor: str, title: str, text_html: str) -> str:
    return f"<div class='section' id='{anchor}'><h2>{title}</h2><p>{text_html}</p></div>"


def scale(tip: float | None, you: float, winner_if_none: str, gmax: float = 20.0) -> str:
    """Risk-aversion line from 0 to gmax, split at the tipping point."""
    pos = lambda g: f"{max(0.0, min(g, gmax)) / gmax * 100:.1f}%"
    marks = [("Bold", 1.5, False), ("Cautious", 10.0, False)]
    dots = "".join(
        f"<div style='left:{pos(g)};top:52px;width:10px;height:10px;border-radius:50%;background:#fff;"
        f"border:1px solid {INK};box-sizing:border-box;transform:translateX(-50%)'></div>"
        f"<p class='tick' style='left:{pos(g)};color:{MUTED}'>{n}</p>" for n, g, _ in marks if abs(g - you) > 0.6)
    you_dot = (f"<div style='left:{pos(you)};top:50px;width:14px;height:14px;border-radius:50%;background:{INK};"
               f"transform:translateX(-50%)'></div><p class='tick' style='left:{pos(you)}'>You</p>")
    if tip is not None and tip < gmax:
        t = pos(tip)
        body = (f"<div style='left:0;width:{t};top:56px;height:2px;background:{INK}'></div>"
                f"<div style='left:{t};right:0;top:56px;height:2px;background:{ACCENT}'></div>"
                f"<p style='left:0;top:0'>Invest it all now</p>"
                f"<p style='right:0;top:0;color:{ACCENT};text-align:right'>Spread it out</p>"
                f"<div style='left:{t};top:30px;width:1px;height:54px;background:{INK}'></div>"
                f"<p class='tick' style='left:{t}'>{tip:.1f}</p>")
    else:
        col, label = (INK, "Invest it all now, whatever your attitude to risk") if winner_if_none == "all now" \
            else (ACCENT, "Spread it out, whatever your attitude to risk")
        body = (f"<div style='left:0;right:0;top:56px;height:2px;background:{col}'></div>"
                f"<p style='left:0;top:0;color:{col}'>{label}</p>")
    return (f"<div class='scale' role='img' aria-label='Risk aversion scale from 0 to {gmax:g}'>"
            f"{body}{dots}{you_dot}</div>")


def bars(values, years: list[str]) -> str:
    scale_px = 140 / max(0.2, max(abs(v) for v in values))
    cells = "".join(
        f"<div><div class='up'><div style='height:{max(v, 0) * scale_px:.0f}px;background:{ACCENT}'></div></div>"
        f"<div class='down'><div style='height:{max(-v, 0) * scale_px:.0f}px;background:{INK}'></div></div></div>"
        for v in values)
    ys = "".join(f"<span>{y}</span>" for y in years)
    return f"<div class='bars' role='img' aria-label='{len(values)} start months'>{cells}</div><div class='years'>{ys}</div>"


def quote(text: str) -> str:
    return f"<div class='quote' id='crash' role='note'>“{e(text)}”</div>"
