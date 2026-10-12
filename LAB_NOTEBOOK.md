<!--
PURPOSE: Append-only log of daily work by both authors.
WHAT GOES HERE: One dated entry per working session: what you did, found, broke and will do next.
FILLED IN AT STEP: Every step, starting at W1.1.
DO NOT: Edit or delete past entries. Do not paste secrets, tokens or personal data.
-->

# Lab notebook

**Rule: append only.** Add new entries at the bottom. Never edit or delete an old entry. If an old entry is wrong, add a new entry that corrects it and says which entry it corrects.

## Entry template (copy this, do not fill it in place)

```
## YYYY-MM-DD, <name>
Did:
Found:
Broke:
Next:
```

<!-- Entries start below this line. -->

## 2026-10-12, Claude (D22)
Did:
- Re-planned the calendar for one laptop (D20, D21). G1 Wed 14 Oct 12:00; `prereg-v1` Wed 14 Oct 18:00; G2 Fri 16 Oct 12:00; full run Fri 16 Oct to Wed 21 Oct 06:00; freeze and D16 Wed 21 Oct; G3 Sat 24 Oct; submission Wed 28 Oct; deadline Fri 30 Oct, 19:00 UTC. Slip rule: if G1 has not passed by Wed 14 Oct 12:00, every later date moves by one day, D16 included, and the cut order applies (stretch analyses, then H2, then SmolVLM-500M, then the anti-aliasing factor).
- Logged D7 to D19 as open and D20 to D23 as decided in `docs/05_decisions.md`. Appended steps W1.8 to W1.10 and W3.6 to W3.8 to `docs/03_pipeline.md`, moved the gate dates and added the freeze gate.
- Created the environment: uv-managed CPython 3.11.15 at `%APPDATA%\uv\python\cpython-3.11.15-windows-x86_64-none`, with a virtual environment `.venv` in the working tree (ignored by git). torch 2.14.1+cu130 and torchvision 0.29.1+cu130 come from `https://download.pytorch.org/whl/cu130`. Also installed: transformers 5.19.0, accelerate 1.15.0, numpy 2.4.6, scipy 1.17.1, jiwer 4.0.0, pillow 12.3.0, freetype-py 2.5.1, matplotlib 3.11.2, pandas 3.0.6, pytest 9.1.1, PyYAML 6.0.3, huggingface-hub 2.2.0, tokenizers 0.23.3.
- Fetched each reader's weights at its pinned revision into `%USERPROFILE%\.cache\huggingface\hub`: `Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3`, `HuggingFaceTB/SmolVLM-Instruct@81cd9a775a4d644f2faf4e7becff4559b46b14c7`, `HuggingFaceTB/SmolVLM-500M-Instruct@a7da5b986cb59b408707209984f360a5f4ad7e47`.
Found:
- Machine: Windows 11 Home; 31.7 GB RAM; NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB, driver 610.74; long paths enabled (`LongPathsEnabled` = 1).
- The torch check printed `2.14.1+cu130 13.0 True NVIDIA GeForce RTX 5060 Laptop GPU ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`, then `transformers 5.19.0 numpy 2.4.6 freetype (2, 13, 2)`.
- GNU make is not installed. Until it is, each `make` target is run as its recipe.
Broke:
- Nothing.
Next:
- W1.3: `requirements.txt`, `requirements.lock.txt`, the `model` marker (D17), the Makefile, CI and `scripts/00_check_env.py`.
