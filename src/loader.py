# Task 2 - loads the Contract Act PDF, cleans up OCR noise, and splits it into chunks
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import settings

# a probable section start: a 1-3 digit number, a period, then a capitalized word
# (this is what "1. This Act may be called..." and "47. Time and place..." both match)
SECTION_START = re.compile(r"(?<=[\s\n])(\d{1,3})\.\s+(?=[A-Z])")


# pulls raw text out of every page and joins it into one string
def extract_text(pdf_path: Path = settings.PDF_PATH) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages)


# undoes OCR noise that would otherwise pollute every chunk
def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # rejoin a word broken across a line by a hyphen
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)  # drop lines that are just a page number
    text = re.sub(r"[ \t]{2,}", " ", text)  # collapse repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse repeated blank lines
    return text.strip()


# cuts off the title page and table of contents, keeping only the actual Act text
# (the ToC repeats section titles in the same "N. Capitalized" shape, which would
# otherwise get chunked as if it were real section content)
def strip_front_matter(text: str) -> str:
    idx = text.find("WHEREAS it is expedient")
    return text[idx:] if idx != -1 else text


# forces a paragraph break before every likely section start, so the splitter below
# prefers to break there instead of mid-clause
def mark_section_starts(text: str) -> str:
    text = SECTION_START.sub(lambda m: "\n\n" + m.group(0), text)
    return re.sub(r"\n{3,}", "\n\n", text)


# splits into chunks, preferring the section-start breaks inserted above, and only
# falling back to sentence/word splits for a single section too long to fit one chunk
def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


# runs the full Task 2 pipeline: extract -> clean -> cut front matter -> chunk
def load_and_chunk(pdf_path: Path = settings.PDF_PATH) -> tuple[str, list[str]]:
    raw = extract_text(pdf_path)
    cleaned = clean_text(raw)
    body = strip_front_matter(cleaned)
    marked = mark_section_starts(body)
    chunks = chunk_text(marked)
    return cleaned, chunks
