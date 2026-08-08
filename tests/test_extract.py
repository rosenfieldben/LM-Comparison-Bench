"""Document extraction: bytes in, text out, or a refusal worth showing.

Pure functions, so every case is a value in and a value out. What is
being asserted is not "a parser was called" but WHAT THE MODELS WILL
READ, because the composed prompt is built from exactly this text and a
comparison over an attached document is a comparison over this module's
reading of it.

The adversarial matrix per format lands with K2, which owns composition;
these are the cases the upload boundary needs to be able to stand on.
"""

import io
import zipfile

import pytest

from bench.extract import (
    MAX_EXTRACTED_CHARS,
    ExtractionError,
    extract,
    suffix_of,
)


def docx_bytes(*paragraphs, runs=None):
    """A minimal but real .docx: a zip holding word/document.xml.

    Built rather than fixtured, because the point of the stdlib
    extractor is that the format's own structure does the work, and a
    binary blob checked into the repo would hide which part of the
    structure the reader depends on. runs splits one paragraph into
    several w:t elements, which is what Word does whenever formatting
    changes mid-sentence.
    """
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(
        "<w:p>" + "".join(f"<w:t>{run}</w:t>" for run in (runs or [text])) + "</w:p>"
        if text is None
        else f"<w:p><w:t>{text}</w:t></w:p>"
        for text in paragraphs
    )
    document = f'<?xml version="1.0"?><w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def pdf_bytes(text="hello from a pdf"):
    """A one-page PDF carrying extractable text, written out by hand.

    Hand-built rather than fixtured, and rather than produced by pypdf's
    writer, for one reason: the text has to be EXTRACTABLE, which needs
    a font resource the page's /Resources actually names. A page whose
    content stream draws with an undeclared font parses perfectly and
    extracts to nothing, which is indistinguishable from a scan.

    That is the same shape as the scanned-page case below, and it is
    why an empty extraction is a refusal rather than an empty string:
    the first version of this helper produced exactly such a file and
    the tests caught it as a scan.
    """
    stream = f"BT /F1 24 Tf 20 100 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    start_xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{start_xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def scanned_pdf_bytes():
    """A PDF whose page carries no text at all: the scan shape.

    One page, no content stream, which is what a page of pure image data
    looks like to a text extractor.
    """
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    start_xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{start_xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


# ---- What kind of file is this.


def test_the_suffix_is_lowercased_and_absence_is_empty():
    assert suffix_of("Report.PDF") == ".pdf"
    assert suffix_of("notes.txt") == ".txt"
    assert suffix_of("Makefile") == ""


# ---- Plain text and markdown.


def test_utf8_text_reads_as_itself():
    out = extract("notes.txt", "hello, wörld".encode())

    assert out["text"] == "hello, wörld"
    assert out["extractor"] == "text"


def test_markdown_is_read_as_text_rather_than_rendered():
    """The marks stay. A model reading markdown source is reading what
    the file says; stripping the syntax would be the bench editing the
    document before every model saw it, which is a change to the
    question rather than a courtesy."""
    out = extract("notes.md", b"# Title\n\n- one\n- two\n")

    assert out["text"] == "# Title\n\n- one\n- two"


def test_review_repro_a_non_utf8_file_falls_back_rather_than_refusing():
    """THE DECLARED FALLBACK. cp1252 decodes every byte sequence without
    raising, so a Windows export lands as text rather than as a refusal
    over an encoding the person cannot see.

    A wrong guess becomes mojibake in the prompt, and every model in the
    comparison reads the same mojibake, so the comparison stays fair.
    That is the trade, and it is why the fallback is last rather than
    first.
    """
    # 0x93 and 0x94 are cp1252 smart quotes and are not valid utf-8.
    out = extract("legacy.txt", b"he said \x93hello\x94 to her")

    assert out["text"] == "he said “hello” to her"


# ---- PDF.


def test_a_pdf_with_text_extracts_it():
    out = extract("paper.pdf", pdf_bytes("the quick brown fox"))

    assert "quick brown fox" in out["text"]
    assert out["extractor"] == "pypdf"


def test_the_pypdf_version_is_recorded_rather_than_assumed():
    """A pypdf upgrade changes the text a model reads, so the record has
    to say which parser produced it. Read from the library rather than
    written down, because a hand-copied version is a number that goes
    stale silently."""
    import pypdf

    out = extract("paper.pdf", pdf_bytes())

    assert out["extractor_version"] == pypdf.__version__
    assert out["extractor_version"] != ""


def test_a_file_that_is_not_a_pdf_is_refused_with_a_remedy():
    with pytest.raises(ExtractionError) as caught:
        extract("broken.pdf", b"this is not a PDF at all")

    assert "could not be read" in str(caught.value)
    assert "paste the text" in str(caught.value)


def test_review_repro_a_pdf_of_scanned_pages_is_refused_not_accepted_empty():
    """The ordinary case, and the one that would be worst to accept. A
    scan parses PERFECTLY and yields no text, so a comparison would run,
    cost money, and ask every model about a document none of them
    received. The refusal names the shape so the person knows what
    happened to their file."""
    with pytest.raises(ExtractionError) as caught:
        extract("scan.pdf", scanned_pdf_bytes())

    assert "no text could be read" in str(caught.value)
    assert "scanned" in str(caught.value)
    assert "OCR" in str(caught.value)


# ---- docx.


def test_a_docx_extracts_its_paragraphs():
    out = extract("memo.docx", docx_bytes("first para", "second para"))

    assert out["text"] == "first para\nsecond para"
    assert out["extractor"] == "stdlib-zipfile-xml"
    assert out["extractor_version"] == "1"


def test_review_repro_runs_inside_a_paragraph_join_without_a_separator():
    """Word splits one sentence into several w:t runs wherever formatting
    changes, so joining runs with a space inserts spaces into the middle
    of words wherever somebody bolded half of one. "unbelievable" must
    not arrive as "un believ able"."""
    out = extract("memo.docx", docx_bytes(None, runs=["un", "believ", "able"]))

    assert out["text"] == "unbelievable"


def test_a_docx_that_is_not_a_zip_is_refused():
    with pytest.raises(ExtractionError) as caught:
        extract("memo.docx", b"not a zip")

    assert "readable zip archive" in str(caught.value)


def test_a_zip_without_the_word_document_is_refused_as_not_a_docx():
    """A .doc renamed to .docx is the case: it is a real file and it is
    not this format, and the message says which."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("something/else.xml", "<a/>")

    with pytest.raises(ExtractionError) as caught:
        extract("memo.docx", buffer.getvalue())

    assert "no word/document.xml" in str(caught.value)
    assert "older .doc" in str(caught.value)


def test_a_docx_with_broken_xml_is_refused():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", "<w:document><unclosed>")

    with pytest.raises(ExtractionError) as caught:
        extract("memo.docx", buffer.getvalue())

    assert "unreadable XML" in str(caught.value)


# ---- The refusals every format shares.


def test_an_unsupported_suffix_names_what_the_bench_reads():
    with pytest.raises(ExtractionError) as caught:
        extract("sheet.xlsx", b"anything")

    message = str(caught.value)
    assert ".xlsx file" in message
    for supported in (".txt", ".md", ".pdf", ".docx"):
        assert supported in message


def test_a_suffixless_file_is_refused_by_name():
    with pytest.raises(ExtractionError) as caught:
        extract("Makefile", b"all:\n\techo hi\n")

    assert "suffixless" in str(caught.value)


def test_a_whitespace_only_document_is_empty_rather_than_present():
    """Stripped before the emptiness check, so a file of newlines is the
    same refusal as a file of nothing. A prompt with a delimited block
    containing three blank lines tells every model there was a document
    and shows them none of it."""
    with pytest.raises(ExtractionError) as caught:
        extract("blank.txt", b"   \n\n\t  \n")

    assert "no text could be read" in str(caught.value)


def test_an_enormous_extraction_is_refused_with_the_arithmetic():
    """Bounded like the scorers' subject, and for the same reason: the
    work happens inside a request. The message gives both numbers,
    because a bare "too large" leaves the person guessing whether they
    are over by a line or by a factor of ten."""
    oversized = b"x" * (MAX_EXTRACTED_CHARS + 1)

    with pytest.raises(ExtractionError) as caught:
        extract("huge.txt", oversized)

    message = str(caught.value)
    assert str(MAX_EXTRACTED_CHARS + 1) in message
    assert str(MAX_EXTRACTED_CHARS) in message


def test_a_document_exactly_at_the_limit_is_accepted():
    """The boundary, from the legal side. An off-by-one here refuses a
    file the message says is allowed."""
    out = extract("big.txt", b"x" * MAX_EXTRACTED_CHARS)

    assert len(out["text"]) == MAX_EXTRACTED_CHARS
