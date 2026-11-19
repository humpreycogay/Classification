"""Light/dark theme toggle shared by the generated HTML pages (model guide and reports).

The pages already define both palettes: light by default, dark under `prefers-color-scheme: dark`, and
`:root[data-theme="light"|"dark"]` to force one. The toggle only sets that attribute and remembers the
choice in localStorage. With no saved choice the page follows the system setting.

HEAD_JS goes in <head> so a saved theme applies before the first paint (no flash of the wrong theme).
BUTTON goes wherever the page wants the control; BODY_JS wires it up and must come after it.
"""

STORAGE_KEY = "classification-theme"

HEAD_JS = (
    "<script>try{var t=localStorage.getItem('" + STORAGE_KEY + "');"
    "if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t)}catch(e){}</script>"
)

_SUN = ('<svg class="theme-ico sun" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="3"/>'
        '<path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1.06 1.06M11.54 11.54l1.06 1.06'
        'M3.4 12.6l1.06-1.06M11.54 4.46l1.06-1.06"/></svg>')
_MOON = ('<svg class="theme-ico moon" viewBox="0 0 16 16" aria-hidden="true">'
         '<path d="M13.5 9.6A5.6 5.6 0 0 1 6.4 2.5a5.6 5.6 0 1 0 7.1 7.1z"/></svg>')

BUTTON = (f'<button type="button" class="theme-toggle" id="theme-toggle" aria-label="Switch to dark mode" '
          f'title="Switch to dark mode">{_SUN}{_MOON}</button>')

BODY_JS = (
    "<script>(function(){var b=document.getElementById('theme-toggle');if(!b)return;"
    "var r=document.documentElement,m=window.matchMedia?matchMedia('(prefers-color-scheme: dark)'):null;"
    "function cur(){var t=r.getAttribute('data-theme');return t==='light'||t==='dark'?t:(m&&m.matches?'dark':'light')}"
    "function sync(){var d=cur()==='dark',l=d?'Switch to light mode':'Switch to dark mode';"
    "b.setAttribute('data-mode',d?'dark':'light');b.setAttribute('aria-label',l);b.title=l}"
    "b.addEventListener('click',function(){var n=cur()==='dark'?'light':'dark';r.setAttribute('data-theme',n);"
    "try{localStorage.setItem('" + STORAGE_KEY + "',n)}catch(e){}sync()});"
    "if(m&&m.addEventListener)m.addEventListener('change',sync);sync()})()</script>"
)

# Icon visibility. Shows the theme you would switch TO: a moon in light mode, a sun in dark mode.
# Before BODY_JS runs (no data-mode yet) both icons are hidden, so no wrong icon flashes.
ICON_CSS = """
.theme-toggle .theme-ico { width: 15px; height: 15px; fill: none; stroke: currentColor; stroke-width: 1.6;
  stroke-linecap: round; stroke-linejoin: round; display: none; }
.theme-toggle[data-mode="light"] .moon, .theme-toggle[data-mode="dark"] .sun { display: block; }
"""

# Stand-alone look for the reports, which have no top bar: a small fixed button in the top-right corner.
# Colours use each report's own tokens (--surface, --line/--border, --ink/--text).
FLOATING_CSS = ICON_CSS + """
.theme-toggle { position: fixed; top: 12px; right: 12px; z-index: 20; display: inline-flex; align-items: center;
  justify-content: center; width: 34px; height: 34px; padding: 0; border-radius: 8px; cursor: pointer;
  background: var(--surface); color: var(--ink, var(--text)); border: 1px solid var(--line-strong, var(--border));
  box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.theme-toggle:hover { border-color: currentColor; }
.theme-toggle:focus-visible { outline: 2px solid var(--mark, var(--series)); outline-offset: 2px; }
@media print { .theme-toggle { display: none; } }
"""
