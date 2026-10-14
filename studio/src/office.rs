//! The text of the documents people attach to a chat: PDF (its text layer), Word (.docx), PowerPoint (.pptx), EPUB books and web pages (.html), and plain text and code (`util::read_text_file`). A scan
//! without a text layer has nothing to read and says so.

use std::io::Read;
use std::path::Path;

use crate::util::{human_bytes, read_text_file};

/// The largest PDF or Office file that is opened (the text that comes out is limited by the caller's `limit`).
const MAX_FILE: u64 = 256 << 20;

/// Is this a file `read_document` reads with a parser of its own (slow enough to be done off the window's thread)?
pub fn is_office(path: &Path) -> bool {
    matches!(extension(path).as_str(), "pdf" | "docx" | "pptx" | "epub" | "html" | "htm" | "xhtml")
}

fn extension(path: &Path) -> String {
    path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase()
}

/// The text of a file to attach: PDF, .docx and .pptx through their parsers, anything else as UTF-8 text (up to `limit` bytes, or characters for the documents).
pub fn read_document(path: &Path, limit: u64) -> Result<String, String> {
    let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let kind = extension(path);
    // formats that are not plain text and that no reader here understands (yet): say what to do instead of "not a text file"
    if matches!(kind.as_str(), "doc" | "xls" | "xlsx" | "ppt" | "odt" | "ods" | "odp" | "rtf" | "pages" | "key" | "numbers") {
        return Err(format!("{name}: .{kind} files are not read yet: export or save it as a PDF or a .docx file (or as plain text) and attach that."));
    }
    if !is_office(path) {
        return read_text_file(path, limit);
    }
    let meta = std::fs::metadata(path).map_err(|e| format!("{name}: {e}"))?;
    if meta.len() > MAX_FILE {
        return Err(format!("{name} is {}: too large to open (the limit is {}).", human_bytes(meta.len()), human_bytes(MAX_FILE)));
    }
    let text = match kind.as_str() {
        "pdf" => pdf_text(path).map_err(|e| format!("{name}: {e}"))?,
        "docx" => docx_text(path).map_err(|e| format!("{name}: {e}"))?,
        "pptx" => pptx_text(path).map_err(|e| format!("{name}: {e}"))?,
        "epub" => epub_text(path).map_err(|e| format!("{name}: {e}"))?,
        _ => html_text(&String::from_utf8_lossy(&std::fs::read(path).map_err(|e| format!("{name}: {e}"))?)),
    };
    let text = tidy(&text);
    if text.chars().all(|c| c.is_whitespace()) {
        return Err(if kind == "pdf" { format!("{name} has no text layer (a scan?): it would need character recognition (OCR) first.") } else { format!("{name} holds no text.") });
    }
    if text.len() as u64 > limit {
        return Err(format!("the text of {name} is {} and the limit is {}: attach a smaller file.", human_bytes(text.len() as u64), human_bytes(limit)));
    }
    Ok(text)
}

/// The ligature characters that PDF fonts carry (one glyph for "fi", "ffi" ...) as the letters they stand for: a search for "office" finds "o\u{fb03}ce".
fn unligate(text: &str) -> String {
    text.replace('\u{fb00}', "ff").replace('\u{fb01}', "fi").replace('\u{fb02}', "fl").replace('\u{fb03}', "ffi").replace('\u{fb04}', "ffl").replace('\u{fb05}', "st").replace('\u{fb06}', "st")
}

/// Page breaks and runs of blank lines and spaces tidied up: at most one blank line in a row, no trailing spaces, ligatures spelled out, a word that a line break cut with a hyphen put together again
/// ("neu-" at the end of a line and "ral" at the start of the next).
fn tidy(text: &str) -> String {
    // the pages of a PDF are tidied one by one and keep their form feed between them (an empty page stays: the numbers must not move)
    if text.contains('\u{c}') {
        return text.split('\u{c}').map(tidy_run).collect::<Vec<_>>().join("\u{c}");
    }
    tidy_run(text)
}

fn tidy_run(text: &str) -> String {
    let text = unligate(text);
    let mut out = String::with_capacity(text.len());
    let mut blank = 0usize;
    for line in text.replace('\r', "").lines() {
        let line = line.trim_end();
        if line.trim().is_empty() {
            blank += 1;
            continue;
        }
        // a hyphen that ends a line right after a letter, and a lower-case letter that starts the next one: one word
        let hyphenated = blank == 0 && out.ends_with('-') && out.chars().rev().nth(1).is_some_and(|c| c.is_alphabetic()) && line.chars().next().is_some_and(|c| c.is_lowercase());
        if hyphenated {
            out.pop();
        } else if !out.is_empty() {
            out.push('\n');
            if blank > 0 {
                out.push('\n');
            }
        }
        blank = 0;
        out.push_str(line);
    }
    out
}

/// The text layer of a PDF. A malformed file can make the parser panic: that is an error here, not a crash.
fn pdf_text(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    pdf_text_from_bytes(&bytes)
}

fn pdf_text_from_bytes(bytes: &[u8]) -> Result<String, String> {
    // (the panic message of the parser goes to the standard error as usual: the panic hook is process-wide, and two files read at once must not fight over it)
    let r = std::panic::catch_unwind(|| pdf_extract::extract_text_from_mem_by_pages(bytes));
    match r {
        // the pages one after the other, a form feed between two of them (the chat reads them back as page numbers)
        Ok(Ok(pages)) => Ok(pages.join("\u{c}")),
        Ok(Err(e)) => Err(format!("not a PDF that can be read ({e})")),
        Err(_) => Err("this PDF could not be read (its structure is not one the reader understands)".to_string()),
    }
}

fn open_zip(path: &Path) -> Result<zip::ZipArchive<std::fs::File>, String> {
    let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    zip::ZipArchive::new(f).map_err(|e| format!("not an Office file ({e})"))
}

fn zip_entry(z: &mut zip::ZipArchive<std::fs::File>, name: &str) -> Result<String, String> {
    let f = z.by_name(name).map_err(|_| format!("{name} is missing: not a complete Office file"))?;
    // a part that unpacks to more than 200 MB is not a document (a zip bomb would otherwise fill the memory)
    let mut s = String::new();
    f.take(200 << 20).read_to_string(&mut s).map_err(|e| format!("{name}: {e}"))?;
    Ok(s)
}

/// The text of a Word document: the paragraphs of `word/document.xml`.
fn docx_text(path: &Path) -> Result<String, String> {
    let mut z = open_zip(path)?;
    Ok(xml_text(&zip_entry(&mut z, "word/document.xml")?, "w:t", "w:p"))
}

/// The text of a PowerPoint file: every slide, in order, under a "Slide N" line.
fn pptx_text(path: &Path) -> Result<String, String> {
    let mut z = open_zip(path)?;
    let mut slides: Vec<(usize, String)> = z
        .file_names()
        .filter_map(|n| {
            let n = n.ok()?;
            let num = n.strip_prefix("ppt/slides/slide")?.strip_suffix(".xml")?;
            Some((num.parse::<usize>().ok()?, n.to_string()))
        })
        .collect();
    slides.sort();
    let mut out = String::new();
    for (n, name) in slides {
        let text = xml_text(&zip_entry(&mut z, &name)?, "a:t", "a:p");
        if !text.trim().is_empty() {
            out.push_str(&format!("Slide {n}\n{}\n\n", text.trim()));
        }
    }
    Ok(out)
}

/// The text of a web page: what is between the tags, without scripts, styles and comments, with a line break where a block ends (paragraph, heading, list item, row) and the entities decoded.
pub fn html_text(html: &str) -> String {
    fn push_text(out: &mut String, s: &str) {
        for c in decode_entities(s).chars() {
            if c.is_whitespace() {
                if !out.is_empty() && !out.ends_with(char::is_whitespace) {
                    out.push(' ');
                }
            } else {
                out.push(c);
            }
        }
    }
    // the position of `needle` in `hay`, upper and lower case alike (needle in lower case)
    fn find_ci(hay: &str, needle: &str) -> Option<usize> {
        let (h, n) = (hay.as_bytes(), needle.as_bytes());
        (0..=h.len().checked_sub(n.len())?).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
    }
    let mut out = String::new();
    let mut rest = html;
    while let Some(lt) = rest.find('<') {
        push_text(&mut out, &rest[..lt]);
        rest = &rest[lt + 1..];
        if let Some(r) = rest.strip_prefix("!--") {
            rest = r.find("-->").map_or("", |i| &r[i + 3..]);
            continue;
        }
        let Some(gt) = rest.find('>') else { break };
        let tag = &rest[..gt];
        rest = &rest[gt + 1..];
        let closing = tag.starts_with('/');
        let name = tag.trim_start_matches('/').chars().take_while(|c| c.is_ascii_alphanumeric()).collect::<String>().to_lowercase();
        match name.as_str() {
            "script" | "style" if !closing => {
                rest = find_ci(rest, &format!("</{name}")).map_or("", |i| &rest[i..]);
            }
            "td" | "th" => {
                if closing {
                    out.push('\t');
                }
            }
            "br" | "hr" | "p" | "div" | "li" | "tr" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "section" | "article" | "header" | "footer" | "blockquote" | "pre" | "table" | "ul" | "ol" | "dd" | "dt" | "title" => {
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push('\n');
                }
            }
            _ => {}
        }
    }
    push_text(&mut out, rest);
    out
}

/// The attribute `name` of a tag (`id="x"` or `id='x'`).
fn attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    for q in ['"', '\''] {
        let key = format!("{name}={q}");
        let mut from = 0;
        while let Some(i) = tag[from..].find(&key) {
            let at = from + i;
            // the attribute starts the tag text or follows a space
            if at == 0 || tag.as_bytes()[at - 1].is_ascii_whitespace() {
                let v = &tag[at + key.len()..];
                return v.find(q).map(|e| &v[..e]);
            }
            from = at + key.len();
        }
    }
    None
}

/// The text of an EPUB book: its chapters (XHTML files) in the order of the book's spine, or in the order of their names when the spine cannot be read.
fn epub_text(path: &Path) -> Result<String, String> {
    let mut z = open_zip(path)?;
    let names: Vec<String> = z.file_names().filter_map(|n| n.ok().map(|n| n.to_string())).collect();
    let is_page = |n: &str| {
        let l = n.to_lowercase();
        l.ends_with(".xhtml") || l.ends_with(".html") || l.ends_with(".htm")
    };
    // META-INF/container.xml names the package file, whose manifest maps ids to files and whose spine gives the order
    let mut order: Vec<String> = Vec::new();
    if let Ok(container) = zip_entry(&mut z, "META-INF/container.xml") {
        if let Some(opf_path) = container.split('<').find(|t| t.starts_with("rootfile ")).and_then(|t| attr(t, "full-path")).map(str::to_string) {
            if let Ok(opf) = zip_entry(&mut z, &opf_path) {
                let dir = opf_path.rsplit_once('/').map(|d| format!("{}/", d.0)).unwrap_or_default();
                let tags: Vec<&str> = opf.split('<').collect();
                let href_of = |id: &str| tags.iter().filter(|t| t.starts_with("item ")).find(|t| attr(t, "id") == Some(id)).and_then(|t| attr(t, "href"));
                for t in tags.iter().filter(|t| t.starts_with("itemref")) {
                    if let Some(href) = attr(t, "idref").and_then(href_of) {
                        order.push(format!("{dir}{}", href.replace("%20", " ")));
                    }
                }
            }
        }
    }
    order.retain(|n| names.contains(n) && is_page(n));
    if order.is_empty() {
        order = names.iter().filter(|n| is_page(n)).cloned().collect();
        // chapter2 before chapter10: compare the digit runs as numbers
        let key = |n: &String| -> Vec<(bool, u64, String)> {
            let mut parts: Vec<(bool, u64, String)> = Vec::new();
            let mut cur = String::new();
            for c in n.chars().chain(std::iter::once('\0')) {
                let digit = c.is_ascii_digit();
                if !cur.is_empty() && cur.chars().next().is_some_and(|f| f.is_ascii_digit()) != digit {
                    parts.push((cur.chars().next().is_some_and(|f| f.is_ascii_digit()), cur.parse().unwrap_or(0), std::mem::take(&mut cur)));
                }
                cur.push(c);
            }
            parts
        };
        order.sort_by_key(key);
    }
    if order.is_empty() {
        return Err("no chapters found: not an EPUB book".to_string());
    }
    let mut out = String::new();
    for n in order {
        let page = zip_entry(&mut z, &n)?;
        out.push_str(&html_text(&page));
        out.push_str("\n\n");
    }
    Ok(out)
}

/// The text of an Office XML part: what sits between `<text_tag>` and `</text_tag>`, a line break at the end of every `<para_tag>` paragraph, tabs and line breaks where the file has them, the XML
/// entities decoded. Everything else (formatting, drawings, field codes) is skipped.
fn xml_text(xml: &str, text_tag: &str, para_tag: &str) -> String {
    let mut out = String::new();
    let mut rest = xml;
    while let Some(lt) = rest.find('<') {
        rest = &rest[lt + 1..];
        let Some(gt) = rest.find('>') else { break };
        let tag = &rest[..gt];
        rest = &rest[gt + 1..];
        let name = tag.trim_end_matches('/').split_whitespace().next().unwrap_or("");
        if name == text_tag && !tag.ends_with('/') {
            let end = rest.find('<').unwrap_or(rest.len());
            out.push_str(&decode_entities(&rest[..end]));
            rest = &rest[end..];
        } else if tag.strip_prefix('/') == Some(para_tag) {
            out.push('\n');
        } else if name == "w:tab" || name == "a:tab" {
            out.push('\t');
        } else if matches!(name, "w:br" | "w:cr" | "a:br") {
            out.push('\n');
        }
    }
    out
}

fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let Some(semi) = rest.find(';').filter(|&i| i <= 10) else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let ent = &rest[1..semi];
        let ch = match ent {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            // the few named characters that web pages and books use most
            "nbsp" => Some(' '),
            "mdash" => Some('\u{2014}'),
            "ndash" => Some('\u{2013}'),
            "hellip" => Some('\u{2026}'),
            "lsquo" => Some('\u{2018}'),
            "rsquo" => Some('\u{2019}'),
            "ldquo" => Some('\u{201c}'),
            "rdquo" => Some('\u{201d}'),
            "laquo" => Some('\u{ab}'),
            "raquo" => Some('\u{bb}'),
            "copy" => Some('\u{a9}'),
            "reg" => Some('\u{ae}'),
            "trade" => Some('\u{2122}'),
            "middot" => Some('\u{b7}'),
            "bull" => Some('\u{2022}'),
            _ => ent
                .strip_prefix("#x")
                .and_then(|h| u32::from_str_radix(h, 16).ok())
                .or_else(|| ent.strip_prefix('#').and_then(|d| d.parse::<u32>().ok()))
                .and_then(char::from_u32),
        };
        match ch {
            Some(c) => {
                out.push(c);
                rest = &rest[semi + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("forge-studio-office-{name}-{}", crate::util::new_id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_zip(path: &Path, entries: &[(&str, &str)]) {
        let f = std::fs::File::create(path).unwrap();
        let mut z = zip::ZipWriter::new(f);
        let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        for (name, body) in entries {
            z.start_file(*name, opts).unwrap();
            z.write_all(body.as_bytes()).unwrap();
        }
        z.finish().unwrap();
    }

    /// A one-page PDF with the given lines of text (standard Helvetica, xref table computed here).
    fn tiny_pdf(lines: &[&str]) -> Vec<u8> {
        let mut content = String::from("BT /F1 12 Tf 14 TL 72 720 Td\n");
        for l in lines {
            content.push_str(&format!("({}) Tj T*\n", l.replace('(', "\\(").replace(')', "\\)")));
        }
        content.push_str("ET");
        let objs = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_string(),
            format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len()),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        ];
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, o) in objs.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.extend(format!("{} 0 obj\n{o}\nendobj\n", i + 1).bytes());
        }
        let xref = pdf.len();
        pdf.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).bytes());
        for off in offsets {
            pdf.extend(format!("{off:010} 00000 n \n").bytes());
        }
        pdf.extend(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objs.len() + 1).bytes());
        pdf
    }

    #[test]
    fn xml_paragraphs_tabs_and_entities() {
        let xml = r#"<w:document><w:body><w:p><w:r><w:t>Caf&#233; &amp; bar</w:t></w:r><w:r><w:tab/><w:t xml:space="preserve"> 5 &lt; 6</w:t></w:r></w:p><w:p><w:r><w:t>Second</w:t><w:br/><w:t>line</w:t></w:r></w:p><w:p/></w:body></w:document>"#;
        assert_eq!(xml_text(xml, "w:t", "w:p"), "Caf\u{e9} & bar\t 5 < 6\nSecond\nline\n");
        assert_eq!(decode_entities("a &unknown; &#x41;&#66; & b"), "a &unknown; AB & b");
        let slide = r#"<p:sld><a:p><a:r><a:t>Title</a:t></a:r></a:p><a:p><a:r><a:t>Point one</a:t></a:r></a:p></p:sld>"#;
        assert_eq!(xml_text(slide, "a:t", "a:p"), "Title\nPoint one\n");
    }

    #[test]
    fn page_breaks_and_blank_runs_are_tidied() {
        assert_eq!(tidy("one  \r\ntwo\n\n\n\nthree\u{c}four\n"), "one\ntwo\n\nthree\u{c}four");
        assert_eq!(tidy("a\u{c}\u{c}c"), "a\u{c}\u{c}c");
        assert_eq!(tidy("  \n\n"), "");
        // ligatures and words cut by a hyphen at the end of a line; a hyphen before a capital, a digit or a blank line stays
        assert_eq!(tidy("the o\u{fb03}ce of neu-\nral networks and \u{fb01}ne state-\nOf the art\n\nend-\n\nnew 5-\n6"), "the office of neural networks and fine state-\nOf the art\n\nend-\n\nnew 5-\n6");
    }

    #[test]
    fn a_word_file_gives_its_paragraphs() {
        let dir = temp_dir("docx");
        let f = dir.join("notes.docx");
        write_zip(&f, &[("[Content_Types].xml", "<Types/>"), ("word/document.xml", "<w:document><w:body><w:p><w:r><w:t>The lighthouse keeper's cat is named Biscuit.</w:t></w:r></w:p><w:p><w:r><w:t>Second paragraph.</w:t></w:r></w:p></w:body></w:document>")]);
        assert!(is_office(&f));
        assert_eq!(read_document(&f, 1 << 20).unwrap(), "The lighthouse keeper's cat is named Biscuit.\nSecond paragraph.");
        // not an Office file, and one that lacks the main part
        let bad = dir.join("bad.docx");
        std::fs::write(&bad, b"plain text").unwrap();
        assert!(read_document(&bad, 1 << 20).unwrap_err().contains("not an Office file"));
        let empty = dir.join("empty.docx");
        write_zip(&empty, &[("other.xml", "<x/>")]);
        assert!(read_document(&empty, 1 << 20).unwrap_err().contains("word/document.xml is missing"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_powerpoint_file_gives_its_slides_in_order() {
        let dir = temp_dir("pptx");
        let f = dir.join("deck.pptx");
        let slide = |t: &str| format!("<p:sld><a:p><a:r><a:t>{t}</a:t></a:r></a:p></p:sld>");
        write_zip(&f, &[("ppt/slides/slide10.xml", &slide("Tenth")), ("ppt/slides/slide2.xml", &slide("Second")), ("ppt/slides/slide1.xml", &slide("First"))]);
        assert_eq!(read_document(&f, 1 << 20).unwrap(), "Slide 1\nFirst\n\nSlide 2\nSecond\n\nSlide 10\nTenth");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_pdf_gives_its_text_and_a_scan_or_a_broken_file_says_why() {
        let dir = temp_dir("pdf");
        let f = dir.join("report.pdf");
        std::fs::write(&f, tiny_pdf(&["The ship docks at six.", "Cargo: tea and rope."])).unwrap();
        let t = read_document(&f, 1 << 20).unwrap();
        assert!(t.contains("The ship docks at six.") && t.contains("Cargo: tea and rope."), "{t:?}");
        // a page without any text: a scan
        let scan = dir.join("scan.pdf");
        std::fs::write(&scan, tiny_pdf(&[])).unwrap();
        assert!(read_document(&scan, 1 << 20).unwrap_err().contains("no text layer"));
        // not a PDF at all, and a PDF cut in half
        let junk = dir.join("junk.pdf");
        std::fs::write(&junk, b"this is not a pdf").unwrap();
        assert!(read_document(&junk, 1 << 20).is_err());
        let full = tiny_pdf(&["Some text."]);
        let cut = dir.join("cut.pdf");
        std::fs::write(&cut, &full[..full.len() / 2]).unwrap();
        let _ = read_document(&cut, 1 << 20); // an error or the text that survived, never a panic
        // the limit applies to the text
        assert!(read_document(&f, 10).unwrap_err().contains("limit"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Real files: FORGE_TEST_DOCS=path;path cargo test --lib real_files -- --ignored --nocapture
    #[test]
    #[ignore]
    fn real_files() {
        for p in std::env::var("FORGE_TEST_DOCS").unwrap_or_default().split(';').filter(|p| !p.is_empty()) {
            let t0 = std::time::Instant::now();
            if p.ends_with(".pdf") {
                if let Ok(raw) = pdf_text(Path::new(p)) {
                    println!("{p}: the parser's text has {} page breaks", raw.matches('\u{c}').count());
                }
            }
            match read_document(Path::new(p), 24 << 20) {
                Ok(t) => println!("{p}: {} characters, {} lines in {:.2} s; starts: {:?}", t.chars().count(), t.lines().count(), t0.elapsed().as_secs_f64(), t.chars().take(160).collect::<String>()),
                Err(e) => println!("{p}: ERROR {e}"),
            }
        }
    }

    #[test]
    fn a_web_page_gives_its_text() {
        let html = "<!doctype html><html><head><title>Harbour</title><style>p { color: red }</style><script>var x = '<p>no</p>';</script></head><body><!-- a comment --><h1>The Harbour</h1><p>Ships dock at <b>six</b> &amp; leave\n   at nine&nbsp;o&rsquo;clock.</p><ul><li>tea</li><li>rope</li></ul><table><tr><td>a</td><td>b</td></tr></table><SCRIPT>alert(1)</SCRIPT><p>End.</p></body></html>";
        let t = tidy(&html_text(html));
        assert_eq!(t, "Harbour\nThe Harbour\nShips dock at six & leave at nine o\u{2019}clock.\ntea\nrope\na\tb\nEnd.", "{t:?}");
        assert!(!t.contains("color") && !t.contains("alert") && !t.contains("comment"));
        assert_eq!(html_text("no tags at all"), "no tags at all");
        assert_eq!(html_text("broken <p>tag"), "broken 
tag");
    }

    #[test]
    fn an_epub_gives_its_chapters_in_the_order_of_the_spine() {
        let dir = temp_dir("epub");
        let f = dir.join("book.epub");
        let container = r#"<?xml version="1.0"?><container><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"#;
        let opf = r#"<package><manifest><item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/><item id="c2" href="ch%202.xhtml" media-type="application/xhtml+xml"/><item id="css" href="style.css" media-type="text/css"/></manifest><spine><itemref idref="c2"/><itemref idref="c1"/></spine></package>"#;
        write_zip(&f, &[("META-INF/container.xml", container), ("OEBPS/content.opf", opf), ("OEBPS/ch1.xhtml", "<html><body><p>First chapter.</p></body></html>"), ("OEBPS/ch 2.xhtml", "<html><body><p>Second chapter.</p></body></html>")]);
        assert!(is_office(&f));
        assert_eq!(read_document(&f, 1 << 20).unwrap(), "Second chapter.\n\nFirst chapter.");
        // no readable spine: the pages in the order of their names, numbers compared as numbers
        let g = dir.join("loose.epub");
        write_zip(&g, &[("ch10.xhtml", "<p>Ten.</p>"), ("ch2.xhtml", "<p>Two.</p>"), ("ch1.xhtml", "<p>One.</p>")]);
        assert_eq!(read_document(&g, 1 << 20).unwrap(), "One.\n\nTwo.\n\nTen.");
        let none = dir.join("none.epub");
        write_zip(&none, &[("mimetype", "application/epub+zip")]);
        assert!(read_document(&none, 1 << 20).unwrap_err().contains("not an EPUB"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn formats_that_are_not_read_say_what_to_do() {
        let dir = temp_dir("unsupported");
        let f = dir.join("sheet.xlsx");
        std::fs::write(&f, b"PK\x03\x04 not really").unwrap();
        let e = read_document(&f, 1 << 20).unwrap_err();
        assert!(e.contains(".xlsx files are not read yet") && e.contains("PDF"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn other_files_are_read_as_text() {
        let dir = temp_dir("txt");
        let f = dir.join("a.rs");
        std::fs::write(&f, "fn main() {}\n").unwrap();
        assert!(!is_office(&f));
        assert_eq!(read_document(&f, 1 << 20).unwrap(), "fn main() {}\n");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
