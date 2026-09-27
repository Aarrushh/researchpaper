<!--
PURPOSE: Architecture-decision-record style log of every design and gate decision.
WHAT GOES HERE: One entry per decision with evidence and consequences.
FILLED IN AT STEP: G1, G2, G3, and any time a plan changes.
DO NOT: Delete or rewrite old entries. Supersede them with a new entry that references the old ID.
-->

# Decision log

## Template

```
### D<nn>. <title>
- ID: D<nn>
- Status: open | decided | superseded by D<nn>
- Date: YYYY-MM-DD
- Decision:
- Options considered:
- Evidence:
- Consequences:
```

---

### D1. Two readers with different grids
- ID: D1
- Status: decided
- Date: 2026-09-27 (recorded at scaffold time; decided before it)
- Decision: Use two VLM readers whose encoders have different token cell sizes (Qwen2.5-VL-3B-Instruct and SmolVLM-2.2B).
- Options considered: one reader; two readers with the same grid; two readers with different grids.
- Evidence: If error periodicity tracks each reader's own cell size, the effect can be attributed to the grid rather than to the font or text.
- Consequences: SmolVLM cell size must be confirmed at W1.6. Doubles compute for the full sweep.

### D2. Transcription CER as primary metric rather than QA
- ID: D2
- Status: decided
- Date: 2026-09-27 (recorded at scaffold time; decided before it)
- Decision: Primary metric is character error rate of a full transcription, computed with jiwer. WER is secondary.
- Options considered: question answering accuracy; transcription CER; transcription WER.
- Evidence: QA scores overstate text fidelity (ZeroSense, arXiv 2608.01848). TODO[W1.2]: verify this claim and log it in references/citations.md.
- Consequences: Outputs are long; `max_new_tokens` must cover a full page. Text normalisation must be fixed in the preregistration.

### D3. Render at processor-native sizes
- ID: D3
- Status: decided
- Date: 2026-09-27 (recorded at scaffold time; decided before it)
- Decision: Render every page at a size the processor will not resize, so the phase we set is the phase the encoder sees.
- Options considered: render at arbitrary size and let the processor resize; render at native size.
- Evidence: Qwen2.5-VL rounds image sides to multiples of 28, which would move glyphs relative to the grid.
- Consequences: Page sizes depend on the reader's cell size. `mrr.grid.native_size` must be verified at W1.6.

### D4. G1 outcome
- ID: D4
- Status: open
- Date: TODO[W1.7]: date of G1 decision
- Decision: TODO[W1.7]: go or no-go at G1
- Options considered: go; no-go.
- Evidence: TODO[W1.7]: token count check, overlay images, transcription test
- Consequences: TODO[W1.7]: what happens next

### D5. G2 outcome
- ID: D5
- Status: open
- Date: TODO[W2.5]: date of G2 decision
- Decision: TODO[W2.5]: periodic or not
- Options considered: go (periodic); no-go (negative-result framing).
- Evidence: TODO[W2.5]: Figure 1 and the preregistered periodicity test
- Consequences: TODO[W2.5]: framing for W3 onward

### D6. G3 outcome
- ID: D6
- Status: open
- Date: TODO[W4.4]: date of G3 decision
- Decision: TODO[W4.4]: kernel matches or not
- Options considered: go (matches); no-go (documented failure).
- Evidence: TODO[W4.4]: judge table from h2/judge/compare.py
- Consequences: TODO[W4.4]: how H2 is reported
