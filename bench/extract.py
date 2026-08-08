"""Document text extraction. Pure functions over bytes.

The bench composes what it extracts into the prompt, identically for
every model in a comparison, so this module decides what the models
actually read. Two consequences follow and both are load-bearing.

IT IS PART OF THE MEASUREMENT, not a convenience. A comparison over an
attached document is a comparison over THIS module's reading of that
document; if the extraction is poor, every model is being asked a poorer
question, and they are all being asked the same poorer question, which
is the property that keeps the comparison fair even when the extraction
is bad. That is the inline estimand, and it is why extraction is the
default rather than native document parts: the bench's own parsing is
the one thing every model can be held to equally.

SO THE PARSER'S IDENTITY IS PROVENANCE. Every attachment row records the
extractor and its version, because a pypdf upgrade changes the text a
model reads and a record that could not say which parser produced it
would be a record of a prompt nobody can reconstruct. extract() returns
those two strings beside the text for exactly that reason; it does not
let a caller record only the result.

No I/O and no clock: bytes in, text out. The caller owns the file, the
database and the decision to store anything.
"""

import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO

import pypdf

# Where a document stops being a document the bench will read. Bounded
# for the reason every subject in this codebase is bounded: extraction
# runs inside a request, and a file whose text is larger than any prompt
# could carry is a file that will be refused later anyway, so refusing
# it here is refusing it in the cheaper place.
#
# Generous on purpose. 400k characters is roughly a 150-page report,
# well past what fits any current context window once a prompt is added,
# so a document refused here was never going to be answerable whole. The
# composed-prompt ceiling is a separate and smaller bound applied at
# composition, because that one is about what reaches a provider.
MAX_EXTRACTED_CHARS = 400_000

# What this module can read, by lowercased filename suffix. Keyed on the
# suffix rather than the browser-supplied MIME type deliberately: the
# MIME arrives from the client and is a claim, while the extension is
# also a claim but is the one the person typed. Neither is trusted for
# CONTENT, which is why each extractor validates the bytes it gets and
# fails rather than guessing.
SUPPORTED_SUFFIXES = (".txt", ".md", ".pdf", ".docx")

# The docx paragraph-text element, in the WordprocessingML namespace.
# Pinned against the OOXML shape rather than matched loosely: w:t holds
# run text, and a bare local-name match would also collect w:tab, w:tbl
# and anything else beginning with t.
_DOCX_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DOCX_TEXT = f"{_DOCX_NS}t"
_DOCX_PARAGRAPH = f"{_DOCX_NS}p"
_DOCX_DOCUMENT = "word/document.xml"


class ExtractionError(RuntimeError):
    """A document the bench will not read.

    RuntimeError rather than a custom hierarchy, matching DatasetError
    and the boot-time configuration errors: the caller's only sensible
    response is to show the message, and the message is written to be
    shown.
    """


def suffix_of(filename: str) -> str:
    """The lowercased final suffix, or "" when there is none.

    Its own function because two callers need the same answer and a
    second spelling of "what kind of file is this" is one too many.
    """
    _, dot, tail = filename.rpartition(".")
    return f".{tail.lower()}" if dot else ""


def _decoded(content: bytes) -> str:
    """Plain text bytes as a string, utf-8 first and cp1252 after.

    THE FALLBACK IS DECLARED rather than silent, which is the whole
    reason it is spelled out here. utf-8 is the answer for anything
    written this decade; the file that is not utf-8 is almost always a
    Windows export, and cp1252 decodes every byte sequence without
    error, so it is a fallback that cannot itself fail.

    That last property is why it is also the LAST resort: because it
    never raises, a wrong guess here becomes mojibake in the prompt
    rather than a refusal. Every model in the comparison reads the same
    mojibake, so the comparison stays fair, and the alternative (refuse
    anything not utf-8) would reject readable files over an encoding the
    person cannot see.
    """
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("cp1252")


def _extract_text(content: bytes) -> str:
    return _decoded(content)


def _extract_pdf(content: bytes) -> str:
    """Page text through pypdf, joined with blank lines between pages.

    Pages joined rather than concatenated, because a page boundary is
    real structure in the source and running the last line of one page
    into the first of the next invents a sentence nobody wrote.

    A PDF whose pages are images carries no text and extracts to
    nothing. That is not an error here; it is caught by the empty check
    in extract(), which can say the useful thing (this looks like a
    scan) rather than this function saying the useless one.
    """
    try:
        reader = pypdf.PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        # pypdf raises a family of its own errors plus whatever the
        # underlying parse hits, and the caller's response to every one
        # of them is identical: refuse the upload and say why. Catching
        # the family by name would be a list to keep in step with a
        # dependency, which is the same defect as a stated field order.
        raise ExtractionError(
            f"this PDF could not be read ({type(exc).__name__}: {exc}). "
            "It may be encrypted, damaged, or not a PDF. Attach a text "
            "or Word version, or paste the text into the prompt."
        ) from exc
    return "\n\n".join(page for page in pages if page.strip())


def _extract_docx(content: bytes) -> str:
    """Paragraph text from WordprocessingML, standard library only.

    A docx is a zip holding word/document.xml, and the text a reader
    sees is the w:t runs inside w:p paragraphs. Walking those two
    elements is the whole extraction, which is why this needs no
    dependency: the format's own structure does the work.

    PARAGRAPHS ARE PRESERVED and runs inside one are joined without a
    separator, because Word splits a single sentence into several runs
    whenever formatting changes mid-line. Joining runs with a space
    would insert spaces into the middle of words wherever somebody
    bolded half of one.

    What this does not read: tables, headers, footers, footnotes and
    comments, which live in other parts of the archive. Stated rather
    than discovered, and a limit worth knowing before attaching a
    document whose content is mostly tabular.
    """
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            document = archive.read(_DOCX_DOCUMENT)
    except KeyError as exc:
        raise ExtractionError(
            "this .docx has no word/document.xml inside it, so it is not "
            "a Word document however it is named. If it is an older .doc, "
            "save it as .docx first."
        ) from exc
    except zipfile.BadZipFile as exc:
        raise ExtractionError(
            "this .docx is not a readable zip archive, so it is damaged or "
            "is not a Word document. Try re-saving it from Word."
        ) from exc
    try:
        root = ET.fromstring(document)
    except ET.ParseError as exc:
        raise ExtractionError(
            f"this .docx contains unreadable XML ({exc}). The file is "
            "damaged; try re-saving it from Word."
        ) from exc
    paragraphs = [
        "".join(node.text or "" for node in paragraph.iter(_DOCX_TEXT))
        for paragraph in root.iter(_DOCX_PARAGRAPH)
    ]
    return "\n".join(text for text in paragraphs if text.strip())


# suffix -> (reader, extractor name). The name is what lands on the row
# beside the version, so it says which code read the file rather than
# which library was installed: "stdlib-zipfile-xml" is the honest label
# for an extractor this repo wrote out of the standard library.
_READERS = {
    ".txt": (_extract_text, "text"),
    ".md": (_extract_text, "text"),
    ".pdf": (_extract_pdf, "pypdf"),
    ".docx": (_extract_docx, "stdlib-zipfile-xml"),
}

# The version recorded beside each extractor name. pypdf reports its
# own; the two written here are versioned by the app build, and app_sha
# is already on every run, so "1" is a real answer rather than a
# placeholder: it changes when the reading rule changes, deliberately
# and by hand.
_VERSIONS = {
    "text": "1",
    "pypdf": pypdf.__version__,
    "stdlib-zipfile-xml": "1",
}


def extract(filename: str, content: bytes) -> dict[str, str]:
    """Read one document, or refuse it with a message worth showing.

    Returns the text and the parser's identity together, because the
    identity is provenance: the same bytes read by a later pypdf may
    produce different text, and a record that cannot say which parser
    ran cannot reconstruct the prompt a model actually saw.

    REFUSAL IS THE POINT OF DOING THIS AT UPLOAD. A document that cannot
    be read is a comparison that would run, cost money, and compare
    models on a prompt containing nothing from the file. Failing here is
    failing before anything is attached and before anything is spent,
    which is the same argument the dataset loader makes for reading the
    whole file at creation rather than at trial 300.

    An empty extraction is a refusal for the same reason. A scanned PDF
    is the ordinary case: it parses perfectly and yields no text, so
    every model would be asked about a document none of them received.
    """
    suffix = suffix_of(filename)
    reader = _READERS.get(suffix)
    if reader is None:
        raise ExtractionError(
            f"{filename!r} is a {suffix or 'suffixless'} file, and the "
            f"bench reads {', '.join(SUPPORTED_SUFFIXES)}. Convert it, or "
            "paste its text into the prompt."
        )
    read, extractor = reader
    text = read(content).strip()
    if not text:
        raise ExtractionError(
            f"no text could be read from {filename!r}. A PDF of scanned "
            "pages looks like this: it parses correctly and contains no "
            "text at all. The bench does not do OCR, so attach a text "
            "version or paste the text into the prompt."
        )
    if len(text) > MAX_EXTRACTED_CHARS:
        raise ExtractionError(
            f"{filename!r} extracts to {len(text)} characters, over the "
            f"{MAX_EXTRACTED_CHARS} limit. That is past what any current "
            "context window holds once a prompt is added, so the "
            "comparison could not have run whole. Attach the relevant "
            "section instead."
        )
    return {
        "text": text,
        "extractor": extractor,
        "extractor_version": _VERSIONS[extractor],
    }
