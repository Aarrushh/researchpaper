<!--
PURPOSE: Record of every font used, where it came from and its licence.
WHAT GOES HERE: Font files, each with its licence file next to it. This table, filled in.
FILLED IN AT STEP: W1.4.
DO NOT: Commit a font without its licence file beside it. Do not use a font whose licence you have not read.
-->

# data/external/fonts/

- **What belongs here:** TrueType or OpenType font files, each with its licence file beside it.
- **Where it comes from:** the font's official publisher. Record the exact URL you downloaded from and the version.
- **Licence:** per font; see table. Rendered pages inherit the font's licence.
- **Committed:** yes, for the fonts listed under "Fonts in use". SIL OFL 1.1 allows redistribution as long as the licence ships with the font. Font versions change glyph shapes, so the exact files are pinned in git.
- **Rule:** keep each font's licence file next to it. Do not rename font files after download. If you replace a font file, update its version, date and SHA-256 below in the same commit.

## Fonts in use

| Font | File | Version | Licence | Licence file | Downloaded from | Date |
|---|---|---|---|---|---|---|
| Source Sans 3 (Adobe) | `source-sans-3/SourceSans3-Regular.ttf` | 3.052 | SIL OFL 1.1 | `source-sans-3/LICENSE.md` | https://github.com/adobe-fonts/source-sans | 2026-10-03 |
| Source Serif 4 Display (Adobe) | `source-serif-4d/SourceSerif4Display-Regular.ttf` | 4.005 | SIL OFL 1.1 | `source-serif-4d/LICENSE.md` | https://github.com/adobe-fonts/source-serif | 2026-10-03 |
| Fira Mono (Mozilla) | `fira-mono/FiraMono-Regular.ttf` | 3.206 | SIL OFL 1.1 | `fira-mono/OFL.txt` | https://raw.githubusercontent.com/google/fonts/22d9fb782d20f88fc9952807408d0e56739cc43a/ofl/firamono/FiraMono-Regular.ttf (licence: same repository at 49fbebd3dc75d42fe72c4a3eef6524f8fcc335fd, `ofl/firamono/OFL.txt`) | 2026-10-18 |

Notes:

- **Version** is read from the font's own `name` table (ID 5, "Version 3.052" and "Version 4.005"). Check with `fc-query <file>` or fontTools.
- **Downloaded from** is the publisher's repository. The exact release tag or file URL was not recorded. TODO[W1.4]: replace with the exact release asset URL if known.
- **Date** is the date the files were added to this repo. The original download date was not recorded. TODO[W1.4]: correct if it differs.
- Both are static (non-variable) TTFs. Source Serif 4 Display is the display optical size, not the text cut.
- Licence copyright lines: Source Sans 3 says "Copyright 2010-2024 Adobe". Source Serif 4 Display says "Copyright 2014 - 2023 Adobe". Both name 'Source' as a Reserved Font Name, so any modified version must be renamed.
- **Fira Mono** is Study 2's monospace face, chosen on 18 October 2026 by the rule of `PREREGISTRATION_2.md` section 7.1 (screening of eight candidates, then the 10-page pre-test on Qwen; counts in `reports/tables/study2_prerun/`). Static TTF; version from its `name` table ("Version 3.206"). The URLs are pinned to the google/fonts commits that last changed each file; the downloaded files' SHA-256 equal those of the files at those commits. Licence copyright line: "Copyright (c) 2012-2013, The Mozilla Corporation and Telefonica S.A."

### SHA-256 checksums

```
4644c81b86ec9caaa76b634889968ed3c4f4f52f054855933acc7c2b21e53b0f  source-sans-3/SourceSans3-Regular.ttf
56af9b9c6715597e458284a474dc118a50a4150e9d547c70f7b4a33c3e6a9328  source-sans-3/LICENSE.md
a1788d674713c99e61d4541c85e5cfbc3145c4e746b0185c674a0b15f7313295  source-serif-4d/SourceSerif4Display-Regular.ttf
c21d7293d87b6d7ab1d0229a2f55b77f33a7613a6a4e66f6693d68d7d8d09464  source-serif-4d/LICENSE.md
2e00b0cf3106a3d792f35711b7722740b76c02ee7300ea8c60f940584b7a8ac9  fira-mono/FiraMono-Regular.ttf
d1829ff3455bbf3b9d13dfb0a57758a863107c9108aab48558f26f37f1094180  fira-mono/OFL.txt
```

Verify from this folder with `sha256sum -c` on the block above.

## Other candidate fonts (not downloaded)

| Font | Licence | Notes | Downloaded from | Version | Date |
|---|---|---|---|---|---|
| Noto Sans | SIL OFL 1.1 | | TODO[W1.4]: source URL | TODO[W1.4]: version | TODO[W1.4]: date |
| Liberation Sans | SIL OFL 1.1 for version 2.0 and later only | Check the version; earlier versions used a different licence | TODO[W1.4]: source URL | TODO[W1.4]: version | TODO[W1.4]: date |
| DejaVu Sans | Bitstream Vera licence, NOT OFL | Read its licence separately | TODO[W1.4]: source URL | TODO[W1.4]: version | TODO[W1.4]: date |
