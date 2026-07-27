# Report

Task-by-task write-up: decisions made, why, and what was observed. See `README.md` for
setup/run instructions.

## Task 1 — Law Selection & PDF Acquisition

**Law selected:** The Contract Act, 1872 (Act No. IX of 1872), enacted 25 April 1872.
Still in force in Pakistan as the foundational law governing contracts (offer, acceptance,
consideration, capacity, free consent, void agreements, performance, indemnity, guarantee,
bailment, agency). Two of its original chapters — Sale of Goods and Partnership — were later
repealed and split off into their own Acts (1930 and 1932 respectively), which is visible in
the PDF itself (Chapter XI is just a "Repealed" notice).

PDF saved at `data/contract_act_1872.pdf` (64 pages).

**Structure:** Preamble, then a "Preliminary" part (short title, extent, definitions), then
11 numbered Chapters (I–XI), each grouping related numbered Sections (1 through 238) under
chapter and sub-headings — e.g. Chapter VI "Of the Performance of Contracts" groups Sections
37–67 under sub-headings like "Performance of Reciprocal Promises" and "Appropriation of
Payments". Each section has a short marginal title (e.g. "47. Time and place for performance
of promise") followed by the operative text, and many sections include worked "Illustrations"
(lettered (a), (b), (c) examples) directly under the section. This numbered-section structure
is what Task 2's chunking is built around.

**Text quality:** Not clean. The PDF's metadata shows `Creator: PDF24 Tools - OCR` — it's a
scanned reprint that was OCR'd, not a born-digital document. Extracted text has real OCR
noise: misrecognized words (e.g. "Indian Contract Act" reads as "fadieay Contract Act" in one
spot), broken line/column order in places, and marginal notes/footnotes bleeding into the
body text. This is expected for a scan of an 1872-era Act and is documented here rather than
treated as a blocker. Task 2 adds a light cleanup pass before chunking to reduce the damage.
