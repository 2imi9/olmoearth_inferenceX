"""Generate the API reference pages for oe_inferencex at build time (mkdocs-gen-files), one page per module with
a mkdocstrings directive, plus a SUMMARY.md that mkdocs-literate-nav turns into the API Reference navigation
(the arrangement rslearn uses). griffe parses the sources statically, so torch is not needed to build the pages."""
from pathlib import Path

import mkdocs_gen_files

PACKAGE = "oe_inferencex"
root = Path(__file__).resolve().parents[1]
src = root / PACKAGE
nav = mkdocs_gen_files.Nav()

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(root).with_suffix("")
    parts = tuple(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = module_path.parent / "index.md"
    elif parts[-1].startswith("_"):
        continue
    else:
        doc_path = module_path.with_suffix(".md")
    identifier = ".".join(parts)
    nav_parts = parts[1:] if len(parts) > 1 else (PACKAGE,)
    full_doc_path = Path("reference", *doc_path.parts[1:])
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# `{identifier}`\n\n::: {identifier}\n")
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))
    nav[nav_parts] = Path(*doc_path.parts[1:]).as_posix()

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
