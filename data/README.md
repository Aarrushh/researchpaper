<!--
PURPOSE: Map of the data folder and its rules.
WHAT GOES HERE: Nothing but this README and the subfolder READMEs are committed.
FILLED IN AT STEP: W1.4 (downloads), W2.3 onward (interim pages).
DO NOT: Commit any data file. Do not edit anything in raw/ after download.
-->

# data/

Adapted from the Cookiecutter Data Science layout.

| Folder | What belongs here | Comes from | Committed |
|---|---|---|---|
| `external/fonts/` | Font files and their licence files | Font publishers (W1.4) | no |
| `raw/texts/` | Public-domain source texts | Project Gutenberg (W1.4) | no |
| `interim/pages/` | Rendered page images | `mrr.render` (regenerable) | no |
| `processed/` | Cleaned, paginated text ready for rendering | `mrr.text_source` | no |

**Licence:** each file keeps its source's licence. See each subfolder README.

**Rule:** `raw/` is immutable once downloaded. Everything else must be regenerable by a script.
