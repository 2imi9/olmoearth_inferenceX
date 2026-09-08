You are producing a publication-quality TikZ figure for the repository you are running in (olmoearth_inferenceX, a label-free audit of OlmoEarth inference results). Work only in docs/figures/tikz/. Do not touch any other file.

GOAL
Re-create the repository's overview figure in TikZ, one to one in content, diagram-first, as a standalone LaTeX document that compiles with pdflatex.

SOURCE OF TRUTH
Read docs/figures/inferencex_overview.py. It is the matplotlib script that draws the current figure (docs/figures/inferencex_overview.png shows the result). Every string in that script is the figure's content: the title and subtitle, the five panels A to E with their titles and colors, the boxes and arrows of panel A, the six belief rows of panel B (claim, one-line evidence, experiment ids), the three boxes of panel C, the text of panels D and E, the legend line under panel B, and the footer. Reproduce that content exactly; do not add, drop, or rephrase claims. The palette (INK, MUTED, EMERALD, AMBER, VIOLET, the light blue and light green fills) is defined at the top of the script; define the same colors with \definecolor{...}{HTML}{...}.

LAYOUT
Same arrangement as the PNG: panel A (what is audited) at the left with a vertical flow of boxes joined by arrows; panel B (what we believe is true) in the middle, the widest, six rows each with a green check mark, a bold claim line, a muted evidence line and a muted experiment-id line; panel C (the test every claim passed) at the right with three stacked rounded boxes; panels D (open question) and E (where it stands) under A and C. Arrows from A to B and from B to C at mid height. Target size 26 cm wide by 12 cm tall, to be used at full page width; all text must stay inside its panel (use text width and align=left on every multi-line node; check the compiled PDF for overflow and fix it rather than shrinking fonts below \scriptsize).

TIKZ CONVENTIONS
- \documentclass[tikz,border=4pt]{standalone}; \usetikzlibrary{positioning,fit,calc,arrows.meta,shapes.misc,backgrounds}.
- Define styles once in the tikzpicture options: panel (rounded rectangle, thin ink border, light fill), box, flowbox, claim, evidence, ids, check (a small filled circle with a white tick), arrow (-{Stealth[length=2mm]}, thick).
- Place nodes with the positioning library (below=of, right=of, node distance) and with fit for the panels; use \coordinate and calc for the arrows between panels. No absolute coordinates except for the panel anchors.
- Fonts: title \Large\bfseries, panel titles \normalsize\bfseries in the panel color, claims \small\bfseries, evidence and ids \scriptsize in MUTED.
- Keep the code readable: one node per line, comments per panel, no macros beyond styles.

DELIVERABLES
1. docs/figures/tikz/inferencex_overview.tex, standalone, compiles with `pdflatex -interaction=nonstopmode inferencex_overview.tex` inside docs/figures/tikz/ with zero errors and no overfull boxes that clip text.
2. docs/figures/tikz/inferencex_overview.pdf, the compiled result (run pdflatex yourself; the machine has pdflatex from TinyTeX; pgfplots is not installed and is not needed).
3. docs/figures/tikz/README.md, five lines: what the file is, the compile command, and that the content is generated from inferencex_overview.py and must be changed there first.

CHECKS BEFORE YOU FINISH
- Compile; read the log; fix every error and every "Overfull \hbox" that clips text.
- Compare against the PNG: same panels, same rows, same numbers, same experiment ids. Any string you cannot fit must be wrapped, not cut.
- Report at the end: the compile command, the page size of the PDF, and any string you had to wrap.
