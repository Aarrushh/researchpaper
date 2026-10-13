<!--
PURPOSE: Record of every source text used for pages.
WHAT GOES HERE: Public-domain books as downloaded (plain text or EPUB), with the table below filled in.
FILLED IN AT STEP: W1.4.
DO NOT: Edit a file after download (strip boilerplate in code, not by hand). Release any text or rendered page that still contains the Project Gutenberg header or footer.
-->

# data/raw/texts/

- **What belongs here:** Project Gutenberg public-domain texts, exactly as downloaded. Plain UTF-8 `.txt` or EPUB 3 `.epub`.
- **Where it comes from:** Project Gutenberg. Record the book ID and title for each file.
- **Licence:** the texts are public domain in the US. "Project Gutenberg" is a trademark, and the header and footer carry its licence terms. Strip both in code (`mrr.text_source.strip_gutenberg_boilerplate`), never by editing the raw file. Anything released (processed text, rendered pages) must come from stripped text only.
- **Committed:** yes, for the files listed below. The repo is private. The raw files keep the Gutenberg header and footer unedited, which the Project Gutenberg licence allows.
- **Rule:** immutable once downloaded. If you replace a file, update its row and checksum in the same commit.

## How a file becomes text

`mrr.text_source.load_book` reads `.txt` as is. For `.epub` it walks the spine in reading order and turns each paragraph or heading into a block separated by one blank line; `<br/>` becomes a line break. The header and footer are kept at this stage. `strip_gutenberg_boilerplate` then cuts everything outside the `*** START OF ...` and `*** END OF ...` lines, and raises an error if either is missing.

Next, `skip_front_matter` drops the front matter (table below), and `normalise_text` applies D13 to what is left: NFC, curly quotes to straight ones, the em dash to a hyphen, whitespace collapsed.

## Front matter

Every stripped book still opens with its title and a contents list, and pg1661 also with an editorial note. `skip_front_matter(text, marker, occurrence)` drops everything up to and including the given occurrence of a chapter heading:

| Book | Marker | Occurrence | Text after the skip begins |
|---|---|---|---|
| pg105, Persuasion | `CHAPTER I.` | 2nd | "Sir Walter Elliot, of Kellynch Hall, in Somersetshire, was a man who" |
| pg84, Frankenstein | `Letter 1` | 2nd | "To Mrs. Saville, England. St. Petersburgh, Dec. 11th, 17—." |
| pg1661, Sherlock Holmes | `I. A SCANDAL IN BOHEMIA` | 1st | "I. To Sherlock Holmes she is always the woman." |

- In pg105 and pg84 the first occurrence is the contents list. The contents list of pg1661 spells the title in mixed case ("I. A Scandal in Bohemia"), and matching is case-sensitive, so its first match is the heading itself.
- The heading of pg1661 breaks over two lines, "I." and "A SCANDAL IN BOHEMIA". The marker's words therefore match across any whitespace, and only as whole words.
- The last column is the text after whitespace is collapsed, before `normalise_text` (so the em dash is still there). `tests/test_text_source.py` checks all three rows on the committed files.
- Later chapter headings ("CHAPTER II." and so on) stay in the text. TODO[W2.2]: decide whether to keep them, and record it here. Proposed: keep them; they are text on the page and in the reference alike, so they cost nothing.

## Books

| File | Book ID | Title | Author | Why | Downloaded from | PG last updated | Date |
|---|---|---|---|---|---|---|---|
| `pg105-images-3.epub` | 105 | Persuasion | Jane Austen | Long prose paragraphs | https://www.gutenberg.org/ebooks/105 | 2024-10-29 | 2026-10-03 |
| `pg84-images-3.epub` | 84 | Frankenstein; or, the Modern Prometheus | Mary Wollstonecraft Shelley | Varied vocabulary, letters and narrative | https://www.gutenberg.org/ebooks/84 | 2026-02-10 | 2026-10-03 |
| `pg1661-images-3.epub` | 1661 | The Adventures of Sherlock Holmes | Arthur Conan Doyle | Dialogue-heavy, punctuation-dense | https://www.gutenberg.org/ebooks/1661 | 2023-10-10 | 2026-10-03 |

Notes:

- **Book ID, title and author** are read from each EPUB's own metadata (`dc:identifier`, `dc:title`, `dc:creator`) and match the IDs above.
- **PG last updated** is the "Most recently updated" line in each book's Gutenberg header. Gutenberg revises texts, so a later download can differ. The checksums below pin the exact files.
- **Downloaded from** is the book's Gutenberg page. The exact file URL was not recorded. TODO[W1.4]: replace if known.
- **Date** is the date the files were added to this repo. The original download date was not recorded. TODO[W1.4]: correct if it differs.
- The text between the markers still has front matter: title, a contents list, and for 1661 a Gutenberg editorial note ("THERE IS AN ILLUSTRATED EDITION OF THIS TITLE ..."). It does not contain the words "Project Gutenberg". `skip_front_matter` removes it (see Front matter above).

### SHA-256 checksums

```
0f9df1bf6f2c922111703df41e1bd5813d1ae2328b456fed1d84518e2e2c91a4  pg105-images-3.epub
b82b1ad5cdd15439ab99e864741a0343f107adc796ce4326e44235a3f6f558a7  pg84-images-3.epub
310ddf29f2b98b870b0e970061d236c9ce7c65163e0e172c254a1db3d070d3f4  pg1661-images-3.epub
```

Verify from this folder with `sha256sum -c` on the block above.
