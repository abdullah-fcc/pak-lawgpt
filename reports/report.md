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

## Task 2 — Document Loading & Chunking

**Extraction:** `src/loader.py` reads every page with `pypdf.PdfReader` and joins the text.
Raw extraction: **170,191 characters / 31,158 words**. After the cleanup pass (below):
**170,125 characters / 31,143 words** — cleanup mostly collapses whitespace/page-number
lines rather than deleting content, so the counts barely move.

First 500 characters (post-cleanup):

```
. Act IX. | Contract. 1872 oF
THE INDIAN CONTRACT ACT, 1872
CONTENTS.
PREAMBLE.
PRELIMINARY,
GROTIONS. |
1. Short title.
Extent.
Commencement.
9, Interpretation-clause.
CHAPTER I.
Or THE COMMUNICATION, ACCEPTANCE AND REVOCATION OF PROPOSALS.
3. Communication, acceptance and revocation of proposals.
Communication when complete.
. Revocation of proposals and acceptances.
...
```

(This first chunk of text is the table of contents, which is why it reads like a list of
section titles rather than prose — confirms extraction is pulling real text, OCR warts and
all, e.g. "GROTIONS." for "DEFINITIONS.".)

**Cleanup (`clean_text`):** de-hyphenates words broken across a line break, drops lines that
are just a bare page number, and collapses repeated whitespace/blank lines. This is
deliberately light — the OCR noise inside real sentences (misread words, scrambled marginal
notes) isn't something a generic regex pass can safely fix without risking damage to
correctly-read text, so it's left as-is and absorbed by the embedding model instead.

**Chunking strategy:** Blind fixed-size splitting was ruled out per the assignment (would
cut sections mid-clause). Instead:

1. `strip_front_matter` cuts everything before the enacting clause ("WHEREAS it is
   expedient..."), removing the title page and table of contents — otherwise the ToC's
   section-title list would get chunked as if it were real section text.
2. `mark_section_starts` finds likely section starts with the regex `(\d{1,3})\.\s+[A-Z]`
   (a short number, a period, then a capitalized word — matches both "1. This Act may be
   called..." and "47. Time and place...") and forces a paragraph break before each one.
3. LangChain's `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=150`,
   separators `["\n\n", "\n", ". ", " "]`) then splits on those paragraph breaks first,
   packing multiple short sections into one chunk when they fit, and only falling back to
   sentence/word-level splits for the rare section too long to fit in one chunk on its own.

**Chosen values:** `chunk_size=1000` chars (~150-200 words) is roughly enough to hold one
full section plus its worked "Illustrations" without a mid-clause cut for most sections;
`chunk_overlap=150` (~15%) keeps the boundary sentence from being orphaned on the sections
long enough to still get split. Result: **229 chunks**, average 730 chars, ranging 110-994.

**Honest limitation:** because the section-number regex reads through OCR noise, it doesn't
recover the true section numbers reliably (digit misreads like 2→9 are common), so chunk
boundaries land near true section starts but aren't guaranteed to be exactly on them. This
is a direct consequence of the scan quality noted in Task 1, not a bug in the splitting logic
— tested by hand against the PDF, the large majority of chunks still land on a clean section
boundary.

**Example chunks:**

Chunk #10 (904 chars) — mid-document, legible despite noise:
```
other person has done or abstained from doing, or does
or abstains from doing, or promises to do or to abstain
from doing, something, such act or abstinence or promise
is called a consideration for the promise :
(c) Every promise and every set of promises, forming the con-
sideration for each other, is an agreement:
...
```

Chunk #80 (984 chars) — Chapter IV, Performance of Contracts, with illustrations:
```
(Chapter IV.—Of the Performance of Contracts.)
Performance of Reciprocal Promises
...
Illustrations.
(a) A and B contract that A shall deliver goods to B to be paid for by B on
delivery. A need not deliver the goods, unless B is ready and willing to pay for
the goods on delivery.
B need not pay for the goods, unless A is ready and willing to deliver them on
payment.
...
```

Chunk #120 (983 chars) — a penalty/compensation section with two lettered illustrations,
cleanly self-contained in one chunk without being cut mid-example.

One thing worth flagging: the very first 2-3 chunks (right after the enacting clause) are
noticeably noisier than the rest of the document — that page has unusually dense historical
footnotes about colonial-era territorial jurisdiction ("Phulera", "Upper Tanawal") that bleed
into the body text. This is a one-off artifact of that specific page, not representative of
chunk quality across the document (confirmed by sampling chunks throughout, shown above).
