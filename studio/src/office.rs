//! The text of the documents people attach to a chat: PDF (its text layer), Word (.docx), Excel (.xlsx), PowerPoint (.pptx), OpenDocument (.odt, .ods, .odp), RTF, EPUB books and web pages (.html), and plain
//! text and code (`util::read_text_file`). A page of a PDF that is only a picture (a scan) is read by the text recognition of the system where there is one (`ocr`: Windows), else it says so.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::Path;

use crate::util::{human_bytes, read_text_file};

/// The largest PDF or Office file that is opened (the text that comes out is limited by the caller's `limit`).
const MAX_FILE: u64 = 256 << 20;

/// Is this a file `read_document` reads with a parser of its own (slow enough to be done off the window's thread)?
pub fn is_office(path: &Path) -> bool {
    matches!(extension(path).as_str(), "pdf" | "docx" | "pptx" | "xlsx" | "xlsm" | "odt" | "ods" | "odp" | "rtf" | "epub" | "html" | "htm" | "xhtml") || is_picture(path)
}

/// A picture file (a screenshot, a photo of a page): its text is read by the text recognition of the system where there is one.
fn is_picture(path: &Path) -> bool {
    matches!(extension(path).as_str(), "png" | "jpg" | "jpeg" | "bmp" | "gif" | "tif" | "tiff" | "webp")
}

fn extension(path: &Path) -> String {
    path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase()
}

/// The text of a file to attach: PDF, Office and OpenDocument files, EPUB and web pages through their parsers, anything else as UTF-8 text (up to `limit` bytes, or characters for the documents).
pub fn read_document(path: &Path, limit: u64) -> Result<String, String> {
    let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let kind = extension(path);
    // formats that are not plain text and that no reader here understands (yet): say what to do instead of "not a text file"
    if matches!(kind.as_str(), "doc" | "dot" | "xls" | "xlsb" | "ppt" | "pages" | "key" | "numbers") {
        let to = match kind.as_str() {
            "doc" | "dot" => "a .docx file",
            "xls" | "xlsb" | "numbers" => "an .xlsx file",
            "ppt" | "key" => "a .pptx file",
            _ => "a PDF or a .docx file",
        };
        return Err(format!("{name}: .{kind} files are not read yet: export or save it as {to} (or as plain text) and attach that."));
    }
    if !is_office(path) {
        return read_text_file(path, limit);
    }
    let meta = std::fs::metadata(path).map_err(|e| format!("{name}: {e}"))?;
    if meta.len() > MAX_FILE {
        return Err(format!("{name} is {}: too large to open (the limit is {}).", human_bytes(meta.len()), human_bytes(MAX_FILE)));
    }
    // why a scan could not be read as pictures either, when it had no text layer
    let scan_failure: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
    let parse = || -> Result<String, String> {
        Ok(match kind.as_str() {
            "pdf" => {
                let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
                let mut pages = pdf_pages(&bytes)?;
                // pages without a text layer (a scan is all of them, a report with a scanned appendix some) are pictures: read by the text recognition that comes with the system where there is one
                if let Err(why) = crate::ocr::read_blank_pages(&bytes, &mut pages) {
                    *scan_failure.borrow_mut() = Some(why);
                }
                pages.join("\u{c}")
            }
            "docx" => docx_text(path)?,
            "pptx" => pptx_text(path)?,
            "xlsx" | "xlsm" => xlsx_text(path)?,
            "odt" => odf_file_text(path, Odf::Text)?,
            "ods" => odf_file_text(path, Odf::Sheet)?,
            "odp" => odf_file_text(path, Odf::Slides)?,
            "rtf" => rtf_text(&std::fs::read(path).map_err(|e| e.to_string())?),
            _ if is_picture(path) => crate::ocr::recognize(&std::fs::read(path).map_err(|e| e.to_string())?)?,
            "epub" => epub_text(path)?,
            _ => html_text(&decode_html(&std::fs::read(path).map_err(|e| e.to_string())?)),
        })
    };
    // a file that no reader copes with is an error on screen, never a thread that dies and leaves "Reading ..." behind
    let text = std::panic::catch_unwind(std::panic::AssertUnwindSafe(parse))
        .unwrap_or_else(|_| Err("this file could not be read (its structure is not one the reader understands)".to_string()))
        .map_err(|e| format!("{name}: {e}"))?;
    // prose is tidied (hyphens at line ends rejoined ...); the rows of a sheet are not prose: a row that ends with a hyphen must not swallow the next one
    let text = if matches!(kind.as_str(), "xlsx" | "xlsm" | "ods") { text.trim_matches(['\n', '\r', ' ']).to_string() } else { tidy(&text) };
    if text.chars().all(|c| c.is_whitespace()) {
        return Err(if kind == "pdf" {
            match scan_failure.borrow().as_deref() {
                Some(why) => format!("{name} has no text layer (a scan?) and reading its pages as pictures failed: {why}."),
                None => format!("{name} has no text layer (a scan?) and no text could be read from its pictures."),
            }
        } else {
            format!("{name} holds no text.")
        });
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
        // (not in a web address or an e-mail address: "solutions/hybrid-" and "cloud" are two parts of a path, not one word)
        let address = |t: &str| t.contains('/') || t.contains('@');
        let in_address = out.rsplit(char::is_whitespace).next().is_some_and(address) || line.split_whitespace().next().is_some_and(address);
        let hyphenated = blank == 0 && out.ends_with('-') && !in_address && out.chars().rev().nth(1).is_some_and(|c| c.is_alphabetic()) && line.chars().next().is_some_and(|c| c.is_lowercase());
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
    // the pages one after the other, a form feed between two of them (the chat reads them back as page numbers)
    pdf_pages(bytes).map(|pages| pages.join("\u{c}"))
}

/// The text layer of each page of a PDF.
fn pdf_pages(bytes: &[u8]) -> Result<Vec<String>, String> {
    // (the panic message of the parser goes to the standard error as usual: the panic hook is process-wide, and two files read at once must not fight over it)
    let r = std::panic::catch_unwind(|| pdf_extract::extract_text_from_mem_by_pages(bytes));
    match r {
        Ok(Ok(pages)) => Ok(pages),
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

/// The text of a Word document: the paragraphs of `word/document.xml`, then its footnotes and endnotes (each in a part of its own).
fn docx_text(path: &Path) -> Result<String, String> {
    let mut z = open_zip(path)?;
    let mut text = xml_text(&zip_entry(&mut z, "word/document.xml")?, "w:t", "w:p");
    for (part, tag, title) in [("word/footnotes.xml", "w:footnote", "Footnotes"), ("word/endnotes.xml", "w:endnote", "Endnotes")] {
        if let Ok(xml) = zip_entry(&mut z, part) {
            let notes = notes_text(&xml, tag);
            if !notes.is_empty() {
                text.push_str(&format!("\n{title}:\n{notes}"));
            }
        }
    }
    Ok(text)
}

/// The notes of `word/footnotes.xml` or `word/endnotes.xml` (`tag` is "w:footnote" or "w:endnote"), one line each; the separators that Word keeps there are left out.
fn notes_text(xml: &str, tag: &str) -> String {
    let (open, close) = (format!("<{tag} "), format!("</{tag}>"));
    let mut out = String::new();
    for chunk in xml.split(open.as_str()).skip(1) {
        let (attrs, rest) = chunk.split_once('>').unwrap_or((chunk, ""));
        if attrs.contains("w:type=") {
            continue;
        }
        let note = xml_text(rest.split(close.as_str()).next().unwrap_or(""), "w:t", "w:p");
        let note = note.split_whitespace().collect::<Vec<_>>().join(" ");
        if !note.is_empty() {
            out.push_str(&note);
            out.push('\n');
        }
    }
    out
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
        let mut text = xml_text(&zip_entry(&mut z, &name)?, "a:t", "a:p").trim().to_string();
        let (diagrams, notes) = slide_extras(&mut z, &name);
        if !diagrams.is_empty() {
            text.push('\n');
            text.push_str(&diagrams);
        }
        if !text.trim().is_empty() || notes.is_some() {
            out.push_str(&format!("Slide {n}\n{}\n", text.trim()));
            if let Some(notes) = notes {
                out.push_str(&format!("Notes:\n{notes}\n"));
            }
            out.push('\n');
        }
    }
    Ok(out)
}

/// The parts that a slide points to through its relationships file: the kind of each ("notesSlide", "diagramData" ...) and its path inside the file.
fn slide_parts(z: &mut zip::ZipArchive<std::fs::File>, slide: &str) -> Vec<(String, String)> {
    let Some(file) = slide.rsplit('/').next() else { return Vec::new() };
    let rels = zip_entry(z, &format!("ppt/slides/_rels/{file}.rels")).unwrap_or_default();
    rels.split('<')
        .filter(|t| t.starts_with("Relationship "))
        .filter_map(|t| {
            let kind = attr(t, "Type")?.rsplit('/').next()?.to_string();
            // the target is relative to the folder of the slide ("../notesSlides/notesSlide2.xml")
            let target = attr(t, "Target")?;
            let mut parts: Vec<&str> = if target.starts_with('/') { Vec::new() } else { vec!["ppt", "slides"] };
            for p in target.split('/') {
                match p {
                    "" | "." => {}
                    ".." => {
                        parts.pop();
                    }
                    p => parts.push(p),
                }
            }
            Some((kind, parts.join("/")))
        })
        .collect()
}

/// What a slide has outside its own file: the text of its SmartArt diagrams (it is in a part of its own) and its speaker notes (without the number of the notes page).
fn slide_extras(z: &mut zip::ZipArchive<std::fs::File>, slide: &str) -> (String, Option<String>) {
    let (mut diagrams, mut notes) = (String::new(), None);
    for (kind, path) in slide_parts(z, slide) {
        let Ok(xml) = zip_entry(z, &path) else { continue };
        let text = xml_text(&xml, "a:t", "a:p");
        match kind.as_str() {
            "diagramData" => {
                diagrams.push_str(text.trim());
                diagrams.push('\n');
            }
            "notesSlide" => {
                let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.chars().all(|c| c.is_ascii_digit()) && *l != "\u{2039}#\u{203a}").collect();
                if !lines.is_empty() {
                    notes = Some(lines.join("\n"));
                }
            }
            _ => {}
        }
    }
    (diagrams.trim().to_string(), notes)
}

/// The text of an Excel workbook: every sheet under a "Sheet: name" line, one line per row, the cells separated by tabs (a gap between two cells keeps its tabs, so the columns stay in line). Text as
/// typed, numbers as stored (with the floating-point noise of 0.1 + 0.2 rounded away), cells that have a date format as dates, formulas as the value Excel last computed.
fn xlsx_text(path: &Path) -> Result<String, String> {
    let mut z = open_zip(path)?;
    let workbook = zip_entry(&mut z, "xl/workbook.xml")?;
    let rels = zip_entry(&mut z, "xl/_rels/workbook.xml.rels").unwrap_or_default();
    let strings = zip_entry(&mut z, "xl/sharedStrings.xml").map(|x| shared_strings(&x)).unwrap_or_default();
    let dates = zip_entry(&mut z, "xl/styles.xml").map(|x| cell_styles(&x)).unwrap_or_default();
    let mut out = String::new();
    for (i, tag) in workbook.split('<').filter(|t| t.starts_with("sheet ")).enumerate() {
        let name = attr(tag, "name").map(decode_entities).unwrap_or_else(|| format!("Sheet{}", i + 1));
        // the workbook names a relationship, and the relationships file says which part it is
        let target = attr(tag, "r:id").and_then(|rid| rels.split('<').find(|t| t.starts_with("Relationship ") && attr(t, "Id") == Some(rid))).and_then(|t| attr(t, "Target"));
        let part = match target {
            Some(t) => t.strip_prefix('/').map_or_else(|| format!("xl/{t}"), str::to_string),
            None => format!("xl/worksheets/sheet{}.xml", i + 1),
        };
        // a chart sheet has no rows, a missing part is skipped
        let Ok(xml) = zip_entry(&mut z, &part) else { continue };
        let rows = sheet_rows(&xml, &strings, &dates);
        if !rows.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&format!("Sheet: {name}\n{rows}"));
        }
    }
    Ok(out)
}

/// The strings of `xl/sharedStrings.xml` in order: the text runs of each item, without the phonetic guides (furigana) that Japanese files carry.
fn shared_strings(xml: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let (mut in_item, mut phonetic) = (false, 0usize);
    let mut rest = xml;
    while let Some(lt) = rest.find('<') {
        rest = &rest[lt + 1..];
        let Some(gt) = rest.find('>') else { break };
        let tag = &rest[..gt];
        rest = &rest[gt + 1..];
        let (closing, empty) = (tag.starts_with('/'), tag.ends_with('/'));
        let name = tag.trim_start_matches('/').trim_end_matches('/').split_whitespace().next().unwrap_or("");
        match name {
            "si" if empty => out.push(String::new()),
            "si" if closing => {
                out.push(std::mem::take(&mut cur));
                in_item = false;
            }
            "si" => {
                cur.clear();
                in_item = true;
            }
            "rPh" if !empty => phonetic = if closing { phonetic.saturating_sub(1) } else { phonetic + 1 },
            "t" if in_item && phonetic == 0 && !closing && !empty => {
                let end = rest.find('<').unwrap_or(rest.len());
                cur.push_str(&decode_entities(&rest[..end]));
                rest = &rest[end..];
            }
            _ => {}
        }
    }
    out
}

/// The column of a cell reference ("B7" is 1, "AA1" is 26), counted from 0; None for something that is not a column of a worksheet (at most "XFD").
fn column_of(cell_ref: &str) -> Option<usize> {
    let letters: String = cell_ref.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
    if letters.is_empty() || letters.len() > 3 {
        return None;
    }
    let n = letters.to_ascii_uppercase().bytes().fold(0usize, |n, b| n * 26 + (b - b'A' + 1) as usize);
    (n <= 16384).then(|| n - 1)
}

/// The rows of one worksheet part, one text line each (an empty row leaves no line).
fn sheet_rows(xml: &str, strings: &[String], dates: &CellStyles) -> String {
    let mut out = String::new();
    let mut row: Vec<String> = Vec::new();
    let mut next_col = 0usize;
    // the cell being read: its column, its type and its style; then its value and, for an inline string, its text
    let mut cell: Option<(usize, String, usize)> = None;
    let (mut value, mut inline) = (String::new(), String::new());
    let (mut in_inline, mut phonetic) = (false, 0usize);
    let mut rest = xml;
    while let Some(lt) = rest.find('<') {
        rest = &rest[lt + 1..];
        let Some(gt) = rest.find('>') else { break };
        let tag = &rest[..gt];
        rest = &rest[gt + 1..];
        let (closing, empty) = (tag.starts_with('/'), tag.ends_with('/'));
        let name = tag.trim_start_matches('/').trim_end_matches('/').split_whitespace().next().unwrap_or("");
        match name {
            "row" if !closing => {
                row.clear();
                next_col = 0;
            }
            "row" => {
                while row.last().is_some_and(|c| c.is_empty()) {
                    row.pop();
                }
                if !row.is_empty() {
                    out.push_str(&row.join("\t"));
                    out.push('\n');
                }
                row.clear();
            }
            "c" if !closing => {
                let col = attr(tag, "r").and_then(column_of).unwrap_or(next_col);
                next_col = col + 1;
                if !empty {
                    cell = Some((col, attr(tag, "t").unwrap_or("n").to_string(), attr(tag, "s").and_then(|s| s.parse().ok()).unwrap_or(0)));
                    value.clear();
                    inline.clear();
                }
            }
            "c" => {
                if let Some((col, kind, style)) = cell.take() {
                    let text = cell_text(&kind, &value, &inline, style, strings, dates).replace(['\t', '\r', '\n'], " ");
                    if row.len() <= col {
                        row.resize(col + 1, String::new());
                    }
                    row[col] = text;
                }
            }
            "v" if cell.is_some() && !closing && !empty => {
                let end = rest.find('<').unwrap_or(rest.len());
                value = decode_entities(&rest[..end]);
                rest = &rest[end..];
            }
            "is" => in_inline = !closing && !empty,
            "rPh" if !empty => phonetic = if closing { phonetic.saturating_sub(1) } else { phonetic + 1 },
            "t" if in_inline && phonetic == 0 && !closing && !empty => {
                let end = rest.find('<').unwrap_or(rest.len());
                inline.push_str(&decode_entities(&rest[..end]));
                rest = &rest[end..];
            }
            _ => {}
        }
    }
    out
}

/// What a cell shows, by its type: "s" an entry of the shared strings, "inlineStr" its own text, "b" a truth value, "str" (the result of a formula), "e" (an error such as #DIV/0!) and "d" as written,
/// anything else a number.
fn cell_text(kind: &str, value: &str, inline: &str, style: usize, strings: &[String], dates: &CellStyles) -> String {
    match kind {
        "s" => value.trim().parse::<usize>().ok().and_then(|i| strings.get(i)).cloned().unwrap_or_default(),
        "inlineStr" => inline.to_string(),
        "b" => {
            if value.trim() == "1" {
                "TRUE".to_string()
            } else {
                "FALSE".to_string()
            }
        }
        "str" | "e" | "d" => value.to_string(),
        _ => number_text(value.trim(), style, dates),
    }
}

/// A number cell as it is shown: a date or time by its format, a number by the format code of its style when `format_number` understands it, else the number as stored.
fn number_text(v: &str, style: usize, styles: &CellStyles) -> String {
    if let (Ok(x), false) = (v.parse::<f64>(), styles.dates.contains(&style)) {
        if let Some(shown) = styles.numbers.get(&style).and_then(|code| format_number(x, code)) {
            return shown;
        }
    }
    raw_number_text(v, styles.dates.contains(&style))
}

fn raw_number_text(v: &str, date: bool) -> String {
    let Ok(x) = v.parse::<f64>() else { return v.to_string() };
    // serial 2958465 is the last day Excel knows (31 December 9999)
    if date && (0.0..2_958_466.0).contains(&x) {
        return excel_date(x);
    }
    // 0.1 + 0.2 is stored as 0.30000000000000004: nobody typed that
    match v.find('.') {
        Some(dot) if v.len() - dot > 14 && !v.contains(['e', 'E']) => format!("{x:.10}").trim_end_matches('0').trim_end_matches('.').to_string(),
        _ => v.to_string(),
    }
}

/// An Excel serial date (days since 1899-12-30; the fraction is the time of day) as "2023-03-15", "2023-03-15 12:30", or, below one day, as the time "18:00".
fn excel_date(serial: f64) -> String {
    let secs = (serial * 86400.0).round() as i64;
    let (days, tod) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    let (h, m, s) = (tod / 3600, tod % 3600 / 60, tod % 60);
    let time = if s != 0 { format!("{h:02}:{m:02}:{s:02}") } else { format!("{h:02}:{m:02}") };
    if days < 1 {
        return time;
    }
    // days since 1970-01-01 (serial 25569). Excel believes 1900 was a leap year: before its 60th day the serials are one day off the real calendar
    let unix = days - 25569 + i64::from(days < 60);
    let (y, mo, d) = civil_from_days(unix);
    if tod == 0 {
        format!("{y:04}-{mo:02}-{d:02}")
    } else {
        format!("{y:04}-{mo:02}-{d:02} {time}")
    }
}

/// The calendar date of a day count since 1970-01-01 (the proleptic Gregorian calendar).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// How the cells of a workbook show their numbers: the styles (positions in `cellXfs`) whose number format is a date or a time, and the format code of the styles that have one of their own.
#[derive(Default)]
struct CellStyles {
    dates: HashSet<usize>,
    numbers: HashMap<usize, String>,
}

/// The format codes that Excel has built in (the numbers it does not list are General, dates and times, fractions, accounting and text).
fn builtin_format(id: u32) -> Option<&'static str> {
    Some(match id {
        1 => "0",
        2 => "0.00",
        3 => "#,##0",
        4 => "#,##0.00",
        9 => "0%",
        10 => "0.00%",
        11 => "0.00E+00",
        37 => "#,##0 ;(#,##0)",
        38 => "#,##0 ;[Red](#,##0)",
        39 => "#,##0.00;(#,##0.00)",
        40 => "#,##0.00;[Red](#,##0.00)",
        _ => return None,
    })
}

/// The number as the format code of a cell shows it (the first section for positive numbers, the second for negative ones, the third for zero): digit placeholders 0 # ?, a decimal point, thousands
/// separators, percent, scientific notation, quoted text, backslash escapes, currency blocks [$EUR-407], padding with _x. None for what is not understood (fractions, conditions, scaling commas,
/// General and text formats): the number is then shown as stored.
fn format_number(x: f64, code: &str) -> Option<String> {
    if !x.is_finite() {
        return None;
    }
    // the sections, split at the semicolons that are not quoted, bracketed or escaped
    let mut sections: Vec<String> = vec![String::new()];
    let (mut quoted, mut bracketed, mut escaped) = (false, false, false);
    for c in code.chars() {
        if escaped {
            escaped = false;
        } else if quoted {
            quoted = c != '"';
        } else if bracketed {
            bracketed = c != ']';
        } else {
            match c {
                '"' => quoted = true,
                '[' => bracketed = true,
                '\\' | '_' | '*' => escaped = true,
                ';' => {
                    sections.push(String::new());
                    continue;
                }
                _ => {}
            }
        }
        sections.last_mut()?.push(c);
    }
    if sections.iter().any(|s| s.trim().is_empty() && sections.len() == 1) || code.trim().eq_ignore_ascii_case("general") || code.trim() == "@" {
        return None;
    }
    // the section that applies: a second one is for negative numbers (and then shows no minus sign of its own), a third one for zero
    let (section, own_sign) = if x < 0.0 && sections.len() >= 2 {
        (&sections[1], true)
    } else if x == 0.0 && sections.len() >= 3 {
        (&sections[2], false)
    } else {
        (&sections[0], false)
    };
    // literal text before the digits, the digit pattern, literal text after them
    let (mut prefix, mut suffix) = (String::new(), String::new());
    let (mut int_pat, mut frac_pat) = (String::new(), String::new());
    let (mut in_fraction, mut seen_digits, mut digits_over) = (false, false, false);
    let (mut percent, mut exponent) = (0i32, None::<usize>);
    let chars: Vec<char> = section.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        i += 1;
        let literal = if seen_digits { &mut suffix } else { &mut prefix };
        match c {
            '"' => {
                while i < chars.len() && chars[i] != '"' {
                    literal.push(chars[i]);
                    i += 1;
                }
                i += 1;
            }
            '\\' => {
                if let Some(&n) = chars.get(i) {
                    literal.push(n);
                    i += 1;
                }
            }
            '_' => {
                i += 1;
                literal.push(' ');
            }
            '*' => i += 1,
            '[' => {
                let end = chars[i..].iter().position(|&c| c == ']').map_or(chars.len(), |p| i + p);
                let inner: String = chars[i..end].iter().collect();
                i = end + 1;
                if let Some(cur) = inner.strip_prefix('$') {
                    // [$EUR-407]: the currency symbol, then the locale
                    literal.push_str(cur.split('-').next().unwrap_or(""));
                } else if inner.contains(['<', '>', '=']) {
                    return None;
                }
            }
            '0' | '#' | '?' if !digits_over => {
                seen_digits = true;
                if in_fraction {
                    frac_pat.push(c);
                } else {
                    int_pat.push(c);
                }
            }
            ',' if seen_digits && !in_fraction && !digits_over => {
                // a comma between digit placeholders groups thousands; one after the last digit would scale the number
                if !chars[i..].iter().take_while(|&&n| n != '.' && !n.is_alphabetic()).any(|&n| matches!(n, '0' | '#' | '?')) {
                    return None;
                }
                int_pat.push(',');
            }
            '.' if seen_digits && !in_fraction && !digits_over => in_fraction = true,
            'E' | 'e' if seen_digits && matches!(chars.get(i), Some('+' | '-')) => {
                i += 1;
                let zeros = chars[i..].iter().take_while(|&&n| n == '0').count();
                i += zeros;
                exponent = Some(zeros.max(1));
                digits_over = true;
            }
            '0' | '#' | '?' | ',' | '.' => return None,
            '%' => {
                percent += 1;
                literal.push('%');
            }
            '/' if seen_digits => return None,
            _ => {
                if seen_digits {
                    digits_over = true;
                }
                literal.push(c);
            }
        }
    }
    if !seen_digits {
        // a section with text only ("zero")
        return Some(format!("{prefix}{suffix}"));
    }
    let decimals = frac_pat.len();
    let mandatory = frac_pat.chars().filter(|&c| c == '0').count();
    let mut v = x.abs() * 100f64.powi(percent);
    let body = if let Some(exp_digits) = exponent {
        let sci = format!("{v:.decimals$e}");
        let (mantissa, exp) = sci.split_once('e')?;
        let exp: i32 = exp.parse().ok()?;
        format!("{mantissa}E{}{:0width$}", if exp < 0 { '-' } else { '+' }, exp.abs(), width = exp_digits.max(2))
    } else {
        // round half away from zero as Excel does (Rust's formatting rounds ties to even)
        let scale = 10f64.powi(decimals as i32);
        v = (v * scale).round() / scale;
        let mut s = format!("{v:.decimals$}");
        if decimals > mandatory {
            // optional decimals (#) drop their trailing zeros
            let keep = s.find('.').map_or(s.len(), |d| d + 1 + mandatory);
            while s.len() > keep && s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        let (int_s, frac_s) = s.split_once('.').map_or((s.as_str(), ""), |p| p);
        let min_int = int_pat.chars().filter(|&c| c == '0').count();
        let mut int_digits = if int_s == "0" && min_int == 0 && !frac_s.is_empty() { String::new() } else { int_s.to_string() };
        while int_digits.len() < min_int {
            int_digits.insert(0, '0');
        }
        if int_pat.contains(',') {
            let mut grouped = String::new();
            for (n, c) in int_digits.chars().enumerate() {
                if n > 0 && (int_digits.len() - n) % 3 == 0 {
                    grouped.push(',');
                }
                grouped.push(c);
            }
            int_digits = grouped;
        }
        if frac_s.is_empty() { int_digits } else { format!("{int_digits}.{frac_s}") }
    };
    let sign = if x < 0.0 && !own_sign { "-" } else { "" };
    Some(format!("{sign}{prefix}{body}{suffix}"))
}

/// The cell styles of `xl/styles.xml`: the dates and times, and the number formats.
fn cell_styles(xml: &str) -> CellStyles {
    let mut formats: HashMap<u32, String> = HashMap::new();
    let mut styles = CellStyles::default();
    let (mut in_xfs, mut index) = (false, 0usize);
    for tag in xml.split('<') {
        if tag.starts_with("numFmt ") {
            if let (Some(id), Some(code)) = (attr(tag, "numFmtId").and_then(|v| v.parse().ok()), attr(tag, "formatCode")) {
                formats.insert(id, decode_entities(code));
            }
        } else if tag.starts_with("cellXfs") {
            in_xfs = true;
            index = 0;
        } else if tag.starts_with("/cellXfs") {
            in_xfs = false;
        } else if in_xfs && (tag.starts_with("xf ") || tag.starts_with("xf/") || tag.starts_with("xf>")) {
            let id: u32 = attr(tag, "numFmtId").and_then(|v| v.parse().ok()).unwrap_or(0);
            if is_date_format(id, formats.get(&id).map(String::as_str)) {
                styles.dates.insert(index);
            } else if let Some(code) = formats.get(&id).map(String::as_str).or_else(|| builtin_format(id)) {
                styles.numbers.insert(index, code.to_string());
            }
            index += 1;
        }
    }
    styles
}

/// Is a number format a date or a time? The built-in ones by number (14 to 22, the East-Asian ones, 45 to 47), a custom one by its code: a y, m, d, h or s that is not inside quotes or brackets.
fn is_date_format(id: u32, code: Option<&str>) -> bool {
    if matches!(id, 14..=22 | 27..=36 | 45..=47 | 50..=58) {
        return true;
    }
    let Some(code) = code else { return false };
    let (mut quoted, mut bracketed, mut skip) = (false, false, false);
    for c in code.chars() {
        if skip {
            skip = false;
        } else if quoted {
            quoted = c != '"';
        } else if bracketed {
            bracketed = c != ']';
        } else {
            match c {
                '"' => quoted = true,
                '[' => bracketed = true,
                // a backslash, an underscore (a space as wide as the next character) and an asterisk (a fill) take the character after them
                '\\' | '_' | '*' => skip = true,
                'y' | 'Y' | 'm' | 'M' | 'd' | 'D' | 'h' | 'H' | 's' | 'S' => return true,
                _ => {}
            }
        }
    }
    false
}

/// A character of Windows-1252, the code page of the `\'hh` escapes of RTF files (the bytes 0x80 to 0x9F are the ones that differ from Latin-1).
fn cp1252(b: u8) -> char {
    match b {
        0x80 => '\u{20ac}',
        0x82 => '\u{201a}',
        0x83 => '\u{192}',
        0x84 => '\u{201e}',
        0x85 => '\u{2026}',
        0x86 => '\u{2020}',
        0x87 => '\u{2021}',
        0x88 => '\u{2c6}',
        0x89 => '\u{2030}',
        0x8a => '\u{160}',
        0x8b => '\u{2039}',
        0x8c => '\u{152}',
        0x8e => '\u{17d}',
        0x91 => '\u{2018}',
        0x92 => '\u{2019}',
        0x93 => '\u{201c}',
        0x94 => '\u{201d}',
        0x95 => '\u{2022}',
        0x96 => '\u{2013}',
        0x97 => '\u{2014}',
        0x98 => '\u{2dc}',
        0x99 => '\u{2122}',
        0x9a => '\u{161}',
        0x9b => '\u{203a}',
        0x9c => '\u{153}',
        0x9e => '\u{17e}',
        0x9f => '\u{178}',
        _ => b as char,
    }
}

/// The groups of an RTF file that hold no text of the document (tables of fonts and colors, styles, document properties, pictures, headers and footers, the instruction half of a field ...).
const RTF_SKIPPED: [&str; 31] = [
    "fonttbl", "colortbl", "stylesheet", "info", "pict", "object", "header", "footer", "headerl", "headerr", "headerf", "footerl", "footerr", "footerf", "themedata", "colorschememapping", "latentstyles", "datastore", "listtable",
    "listoverridetable", "rsidtbl", "generator", "xmlnstbl", "filetbl", "revtbl", "fldinst", "private", "protusertbl", "userprops", "pgdsctbl", "mmathpr",
];

/// The text of an RTF document (what WordPad and "Save as RTF" write): the text of the groups that are text, `\par` and `\line` as line breaks, `\tab` and `\cell` as tabs, `\'hh` bytes as Windows-1252,
/// `\uN` as the Unicode character (a pair of surrogates as one), the escaped `\\`, `\{` and `\}`. Everything else (formatting words, tables, pictures, fields' instructions) is dropped.
fn rtf_text(b: &[u8]) -> String {
    // a character of the text, unless the group is skipped or it is the fallback that follows a \uN
    fn emit(out: &mut String, skip: bool, fallback: &mut usize, c: char) {
        if skip {
            return;
        }
        if *fallback > 0 {
            *fallback -= 1;
            return;
        }
        out.push(c);
    }
    let mut out = String::new();
    let mut stack: Vec<(bool, usize)> = Vec::new();
    let (mut skip, mut uc, mut fallback) = (false, 1usize, 0usize);
    let mut high: Option<u32> = None;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        i += 1;
        match c {
            b'{' => stack.push((skip, uc)),
            b'}' => {
                if let Some((s, u)) = stack.pop() {
                    skip = s;
                    uc = u;
                }
            }
            // line breaks in the file are not part of the text
            b'\r' | b'\n' => {}
            b'\\' => {
                let Some(&n) = b.get(i) else { break };
                if !n.is_ascii_alphabetic() {
                    i += 1;
                    match n {
                        b'\\' | b'{' | b'}' => emit(&mut out, skip, &mut fallback, n as char),
                        b'\'' => {
                            let byte = b.get(i..i + 2).and_then(|h| std::str::from_utf8(h).ok()).and_then(|h| u8::from_str_radix(h, 16).ok());
                            if let Some(byte) = byte {
                                i += 2;
                                emit(&mut out, skip, &mut fallback, cp1252(byte));
                            }
                        }
                        // an ignorable destination: {\*\word ...} is dropped as a whole
                        b'*' => skip = true,
                        b'~' => emit(&mut out, skip, &mut fallback, ' '),
                        b'_' => emit(&mut out, skip, &mut fallback, '-'),
                        // a backslash in front of a line break is a paragraph break
                        b'\n' | b'\r' => emit(&mut out, skip, &mut fallback, '\n'),
                        _ => {}
                    }
                    continue;
                }
                // a control word: letters, an optional number, one space that ends it
                let start = i;
                while i < b.len() && b[i].is_ascii_alphabetic() {
                    i += 1;
                }
                let word = String::from_utf8_lossy(&b[start..i]).to_ascii_lowercase();
                let (neg, mut j) = if b.get(i) == Some(&b'-') { (true, i + 1) } else { (false, i) };
                let digits = j;
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
                let param: Option<i64> = if j > digits { std::str::from_utf8(&b[digits..j]).ok().and_then(|d| d.parse::<i64>().ok()).map(|v| if neg { -v } else { v }) } else { None };
                if j > digits {
                    i = j;
                }
                if b.get(i) == Some(&b' ') {
                    i += 1;
                }
                match word.as_str() {
                    "par" | "line" | "sect" | "page" | "row" => emit(&mut out, skip, &mut fallback, '\n'),
                    "tab" | "cell" | "nestcell" => emit(&mut out, skip, &mut fallback, '\t'),
                    "emdash" => emit(&mut out, skip, &mut fallback, '\u{2014}'),
                    "endash" => emit(&mut out, skip, &mut fallback, '\u{2013}'),
                    "bullet" => emit(&mut out, skip, &mut fallback, '\u{2022}'),
                    "lquote" => emit(&mut out, skip, &mut fallback, '\u{2018}'),
                    "rquote" => emit(&mut out, skip, &mut fallback, '\u{2019}'),
                    "ldblquote" => emit(&mut out, skip, &mut fallback, '\u{201c}'),
                    "rdblquote" => emit(&mut out, skip, &mut fallback, '\u{201d}'),
                    "emspace" | "enspace" | "qmspace" => emit(&mut out, skip, &mut fallback, ' '),
                    "uc" => uc = param.unwrap_or(1).clamp(0, 8) as usize,
                    "u" => {
                        if let Some(p) = param {
                            let code = p.rem_euclid(65536) as u32;
                            let ch = match (high.take(), code) {
                                (Some(h), 0xdc00..=0xdfff) => char::from_u32(0x10000 + ((h - 0xd800) << 10) + (code - 0xdc00)),
                                (_, 0xd800..=0xdbff) => {
                                    high = Some(code);
                                    None
                                }
                                _ => char::from_u32(code),
                            };
                            if let Some(ch) = ch {
                                emit(&mut out, skip, &mut fallback, ch);
                            }
                            // the fallback characters that an old reader would show instead
                            if !skip {
                                fallback = uc;
                            }
                        }
                    }
                    // binary data of a picture: skip its bytes
                    "bin" => i = (i + param.unwrap_or(0).max(0) as usize).min(b.len()),
                    w if RTF_SKIPPED.contains(&w) => skip = true,
                    _ => {}
                }
            }
            _ => emit(&mut out, skip, &mut fallback, cp1252(c)),
        }
    }
    out
}

/// Which part of an OpenDocument file is wanted: prose, a spreadsheet (a heading for every sheet, rows of tab-separated cells) or a presentation (a "Slide N" line for every page).
#[derive(Clone, Copy, PartialEq)]
enum Odf {
    Text,
    Sheet,
    Slides,
}

/// The text of an OpenDocument file (.odt, .ods, .odp): its `content.xml`.
fn odf_file_text(path: &Path, kind: Odf) -> Result<String, String> {
    let mut z = open_zip(path)?;
    Ok(odf_text(&zip_entry(&mut z, "content.xml")?, kind))
}

/// The text of an OpenDocument `content.xml`: what is inside paragraphs and headings (`text:p`, `text:h`), a line break at the end of each; in a table the cells of a row are separated by tabs and the
/// paragraphs of one cell are joined with a space. Explicit tabs, line breaks and runs of spaces are kept, the rest of the white space is collapsed as the format says.
fn odf_text(xml: &str, kind: Odf) -> String {
    let mut out = String::new();
    let (mut para, mut cell, mut slide) = (0usize, 0usize, 0usize);
    let mut rows: Vec<usize> = Vec::new();
    // the name of the sheet whose first row with text has not come yet: its heading goes in front of that row
    let mut sheet: Option<String> = None;
    let mut rest = xml;
    while let Some(lt) = rest.find('<') {
        if para > 0 {
            for c in decode_entities(&rest[..lt]).chars() {
                if c.is_whitespace() {
                    if !out.is_empty() && !out.ends_with(char::is_whitespace) {
                        out.push(' ');
                    }
                } else {
                    out.push(c);
                }
            }
        }
        rest = &rest[lt + 1..];
        let Some(gt) = rest.find('>') else { break };
        let tag = &rest[..gt];
        rest = &rest[gt + 1..];
        let (closing, empty) = (tag.starts_with('/'), tag.ends_with('/'));
        let name = tag.trim_start_matches('/').trim_end_matches('/').split_whitespace().next().unwrap_or("");
        match name {
            // comments and the records of tracked changes (the words that were deleted) are not part of the text
            "office:annotation" | "text:tracked-changes" if !closing && !empty => {
                let end = format!("</{name}>");
                rest = rest.find(&end).map_or("", |i| &rest[i + end.len()..]);
            }
            "text:p" | "text:h" if !empty => {
                if !closing {
                    para += 1;
                } else if para > 0 {
                    para -= 1;
                    if para == 0 {
                        // a paragraph inside a table cell stays in the row
                        if cell > 0 {
                            if !out.ends_with(char::is_whitespace) {
                                out.push(' ');
                            }
                        } else {
                            out.push('\n');
                        }
                    }
                }
            }
            "text:tab" if para > 0 => out.push(if cell > 0 { ' ' } else { '\t' }),
            "text:line-break" if para > 0 => out.push(if cell > 0 { ' ' } else { '\n' }),
            "text:s" if para > 0 => {
                let n = attr(tag, "text:c").and_then(|v| v.parse::<usize>().ok()).unwrap_or(1);
                out.push_str(&" ".repeat(n.min(200)));
            }
            "table:table" if !closing && !empty => {
                if kind == Odf::Sheet {
                    sheet = Some(attr(tag, "table:name").map(decode_entities).unwrap_or_default());
                } else if !out.is_empty() && !out.ends_with('\n') {
                    out.push('\n');
                }
            }
            "table:table-row" if !closing && !empty => rows.push(out.len()),
            "table:table-row" if closing => {
                let start = rows.pop().unwrap_or(0).min(out.len());
                let kept = out[start..].trim_end_matches(['\t', ' ', '\n']).len();
                out.truncate(start + kept);
                if kept > 0 {
                    if let Some(n) = sheet.take() {
                        out.insert_str(start, &format!("{}Sheet: {n}\n", if start > 0 { "\n" } else { "" }));
                    }
                    out.push('\n');
                }
            }
            "table:table-cell" | "table:covered-table-cell" => {
                if empty {
                    // empty cells come as one element with a repeat count: a gap between two cells keeps its tabs (the ones at the end of the row are cut off with the row)
                    let n = attr(tag, "table:number-columns-repeated").and_then(|v| v.parse::<usize>().ok()).unwrap_or(1);
                    out.push_str(&"\t".repeat(n.clamp(1, 64)));
                } else if closing {
                    cell = cell.saturating_sub(1);
                    // (the space that the last paragraph of the cell left)
                    out.truncate(out.trim_end_matches(' ').len());
                    out.push('\t');
                } else {
                    cell += 1;
                }
            }
            "draw:page" if kind == Odf::Slides && !closing && !empty => {
                slide += 1;
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str(&format!("{}Slide {slide}\n", if out.is_empty() { "" } else { "\n" }));
            }
            _ => {}
        }
    }
    out
}

/// The characters of a saved web page: UTF-8 (with or without its mark) or UTF-16 when the file says so, Windows-1252 when the page declares that or when the bytes are not valid UTF-8 (the HTML that
/// Word and other Office programs save is Windows-1252: its curly quotes are single bytes). Other legacy encodings (Shift-JIS, GBK, KOI8-R ...) are not known here and come out as replacement characters.
fn decode_html(bytes: &[u8]) -> String {
    if let Some(rest) = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]) {
        return String::from_utf8_lossy(rest).into_owned();
    }
    if bytes.starts_with(&[0xff, 0xfe]) || bytes.starts_with(&[0xfe, 0xff]) {
        let little = bytes[0] == 0xff;
        let units: Vec<u16> = bytes[2..].chunks_exact(2).map(|c| if little { u16::from_le_bytes([c[0], c[1]]) } else { u16::from_be_bytes([c[0], c[1]]) }).collect();
        return String::from_utf16_lossy(&units);
    }
    // the charset the page declares near its start (<meta charset=...> or the http-equiv form)
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(4096)]).to_lowercase();
    let declared = head.find("charset=").map(|i| head[i + 8..].trim_start_matches(['"', '\'', ' ']).chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').collect::<String>()).unwrap_or_default();
    let legacy = matches!(declared.as_str(), "windows-1252" | "cp1252" | "iso-8859-1" | "latin1" | "latin-1" | "iso-8859-15" | "us-ascii" | "ascii");
    match std::str::from_utf8(bytes) {
        Ok(s) if !legacy => s.to_string(),
        _ => bytes.iter().map(|&b| cp1252(b)).collect(),
    }
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
        // a web address or an e-mail address cut at a hyphen keeps the hyphen
        assert_eq!(tidy("see https://www.ibm.com/solutions/hybrid-\ncloud for more and neu-\nral"), "see https://www.ibm.com/solutions/hybrid-\ncloud for more and neu-\nral".replace("neu-\nral", "neural").as_str());
        assert_eq!(tidy("write to first.last-\nname@example.org"), "write to first.last-\nname@example.org");
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
    fn the_footnotes_and_endnotes_of_a_word_file_follow_its_text() {
        let dir = temp_dir("docx-notes");
        let f = dir.join("notes.docx");
        let body = "<w:document><w:body><w:p><w:r><w:t>Body with a note.</w:t></w:r><w:r><w:footnoteReference w:id=\"1\"/></w:r></w:p></w:body></w:document>";
        // the separators (ids -1 and 0) have a type and no text of their own; a note can have several paragraphs
        let footnotes = "<w:footnotes><w:footnote w:type=\"separator\" w:id=\"-1\"><w:p><w:r><w:separator/></w:r></w:p></w:footnote><w:footnote w:id=\"1\"><w:p><w:r><w:footnoteRef/></w:r><w:r><w:t xml:space=\"preserve\"> First note,</w:t></w:r></w:p><w:p><w:r><w:t>second paragraph.</w:t></w:r></w:p></w:footnote><w:footnote w:id=\"2\"><w:p><w:r><w:t>Another one &amp; more.</w:t></w:r></w:p></w:footnote></w:footnotes>";
        let endnotes = "<w:endnotes><w:endnote w:type=\"separator\" w:id=\"-1\"><w:p/></w:endnote><w:endnote w:id=\"1\"><w:p><w:r><w:t>An endnote.</w:t></w:r></w:p></w:endnote></w:endnotes>";
        write_zip(&f, &[("word/document.xml", body), ("word/footnotes.xml", footnotes), ("word/endnotes.xml", endnotes)]);
        assert_eq!(read_document(&f, 1 << 20).unwrap(), "Body with a note.\n\nFootnotes:\nFirst note, second paragraph.\nAnother one & more.\n\nEndnotes:\nAn endnote.");
        // a file with notes parts that hold only separators gets no heading
        let g = dir.join("plain.docx");
        write_zip(&g, &[("word/document.xml", body), ("word/footnotes.xml", "<w:footnotes><w:footnote w:type=\"separator\" w:id=\"-1\"><w:p/></w:footnote></w:footnotes>")]);
        assert_eq!(read_document(&g, 1 << 20).unwrap(), "Body with a note.");
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
    fn a_slides_speaker_notes_follow_the_slide() {
        let dir = temp_dir("pptx-notes");
        let f = dir.join("deck.pptx");
        let slide = |t: &str| format!("<p:sld><a:p><a:r><a:t>{t}</a:t></a:r></a:p></p:sld>");
        let rels = |target: &str| format!(r#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="{target}"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData" Target="../diagrams/data1.xml"/></Relationships>"#);
        // the text of a SmartArt diagram is in a part of its own
        let diagram = "<dgm:dataModel><dgm:pt><dgm:t><a:p><a:r><a:t>SmartArt text</a:t></a:r></a:p></dgm:t></dgm:pt></dgm:dataModel>";
        // the notes page has the notes and the number of the page in a field
        let notes = "<p:notes><a:p><a:r><a:t>Talk about the harbour.</a:t></a:r></a:p><a:p><a:r><a:t>Second point &amp; a figure.</a:t></a:r></a:p><a:p><a:fld><a:t>1</a:t></a:fld></a:p></p:notes>";
        write_zip(
            &f,
            &[
                ("ppt/slides/slide1.xml", &slide("Harbour")),
                ("ppt/slides/_rels/slide1.xml.rels", &rels("../notesSlides/notesSlide7.xml")),
                ("ppt/notesSlides/notesSlide7.xml", notes),
                ("ppt/diagrams/data1.xml", diagram),
                ("ppt/slides/slide2.xml", &slide("No notes here")),
                ("ppt/slides/slide3.xml", "<p:sld/>"),
                ("ppt/slides/_rels/slide3.xml.rels", &rels("/ppt/notesSlides/notesSlide7.xml")),
            ],
        );
        // (slides 1 and 3 point to the same diagram and notes: the part is found by its path, a part that is missing is skipped)
        assert_eq!(
            read_document(&f, 1 << 20).unwrap(),
            "Slide 1\nHarbour\nSmartArt text\nNotes:\nTalk about the harbour.\nSecond point & a figure.\n\nSlide 2\nNo notes here\n\nSlide 3\nSmartArt text\nNotes:\nTalk about the harbour.\nSecond point & a figure."
        );
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
                Ok(t) => {
                    println!("{p}: {} characters, {} lines in {:.2} s; starts: {:?}", t.chars().count(), t.lines().count(), t0.elapsed().as_secs_f64(), t.chars().take(160).collect::<String>());
                    // FORGE_TEST_DUMP=folder writes the text of every file there (name.txt)
                    if let Ok(dir) = std::env::var("FORGE_TEST_DUMP") {
                        let _ = std::fs::create_dir_all(&dir);
                        let _ = std::fs::write(Path::new(&dir).join(format!("{}.txt", Path::new(p).file_name().unwrap().to_string_lossy())), &t);
                    }
                    // FORGE_TEST_SHOW=2500 prints that many characters of the text itself
                    if let Some(n) = std::env::var("FORGE_TEST_SHOW").ok().and_then(|v| v.parse::<usize>().ok()) {
                        println!("{}", t.chars().take(n).collect::<String>());
                    }
                }
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
        assert_eq!(html_text("broken <p>tag"), "broken \ntag");
    }

    #[test]
    fn a_saved_page_is_decoded_as_utf8_utf16_or_windows_1252() {
        // Word's HTML: curly quotes are the single bytes 0x93 and 0x94, the page declares windows-1252
        let word = b"<html><head><meta http-equiv=Content-Type content=\"text/html; charset=windows-1252\"></head><body><p>\x93Machine learning\x94 isn\x92t magic.</p></body></html>";
        assert_eq!(decode_html(word), "<html><head><meta http-equiv=Content-Type content=\"text/html; charset=windows-1252\"></head><body><p>\u{201c}Machine learning\u{201d} isn\u{2019}t magic.</p></body></html>");
        // no declaration, bytes that are not UTF-8: the same guess
        assert_eq!(decode_html(b"caf\xe9 \x93x\x94"), "caf\u{e9} \u{201c}x\u{201d}");
        // UTF-8 stays UTF-8 (also when it has its mark, and when the declaration says utf-8)
        assert_eq!(decode_html("<meta charset=\"utf-8\">caf\u{e9} \u{201c}x\u{201d}".as_bytes()), "<meta charset=\"utf-8\">caf\u{e9} \u{201c}x\u{201d}");
        assert_eq!(decode_html(b"\xef\xbb\xbfplain"), "plain");
        // UTF-16 with its mark, both byte orders
        assert_eq!(decode_html(&[0xff, 0xfe, b'h', 0, b'i', 0, 0xe9, 0]), "hi\u{e9}");
        assert_eq!(decode_html(&[0xfe, 0xff, 0, b'h', 0, b'i']), "hi");
        assert_eq!(decode_html(b""), "");
        // the file is read through it
        let dir = temp_dir("html-1252");
        let f = dir.join("page.html");
        std::fs::write(&f, word).unwrap();
        assert_eq!(read_document(&f, 1 << 20).unwrap(), "\u{201c}Machine learning\u{201d} isn\u{2019}t magic.");
        let _ = std::fs::remove_dir_all(&dir);
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
        for (file, wanted) in [("sheet.xls", "an .xlsx file"), ("old.doc", "a .docx file"), ("deck.ppt", "a .pptx file"), ("notes.pages", "a PDF or a .docx file")] {
            let f = dir.join(file);
            std::fs::write(&f, b"not really").unwrap();
            let e = read_document(&f, 1 << 20).unwrap_err();
            assert!(e.contains("files are not read yet") && e.contains(wanted), "{file}: {e}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_excel_workbook_gives_its_sheets_as_rows_of_tab_separated_cells() {
        let dir = temp_dir("xlsx");
        let f = dir.join("book.xlsx");
        let workbook = r#"<workbook><sheets><sheet name="Sales &amp; Costs" sheetId="1" r:id="rId1"/><sheet name="Empty" sheetId="2" r:id="rId2"/><sheet name="Notes" sheetId="3" r:id="rId3"/></sheets></workbook>"#;
        let rels = r#"<Relationships><Relationship Id="rId1" Type="x" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="x" Target="worksheets/sheet2.xml"/><Relationship Id="rId3" Type="x" Target="/xl/worksheets/sheet3.xml"/></Relationships>"#;
        // item 1 has a phonetic guide that must not show, item 2 is empty
        let strings = r#"<sst><si><t>Item</t></si><si><r><t>Ap</t></r><r><t xml:space="preserve">ples</t></r><rPh sb="0" eb="1"><t>XX</t></rPh></si><si/><si><t>Total &amp; tax</t></si></sst>"#;
        // style 1 is a built-in date, 2 a custom one, 3 a number with a word in quotes (not a date: shown as Excel shows it, "5.00 days"); the cellStyleXfs entry must not count
        let styles = r#"<styleSheet><numFmts count="2"><numFmt numFmtId="164" formatCode="yyyy\-mm\-dd"/><numFmt numFmtId="165" formatCode="0.00&quot; days&quot;"/></numFmts><cellStyleXfs count="1"><xf numFmtId="14"/></cellStyleXfs><cellXfs count="4"><xf numFmtId="0"/><xf numFmtId="14"/><xf numFmtId="164"/><xf numFmtId="165"/></cellXfs></styleSheet>"#;
        let sheet1 = r#"<worksheet><sheetData>
<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>Qty</t></is></c><c r="D1" t="s"><v>3</v></c></row>
<row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><v>3</v></c><c r="C2" s="1"><v>45000</v></c><c r="D2" s="2"><v>45000.5</v></c><c r="E2" s="3"><v>5</v></c></row>
<row r="3"><c r="A3"/><c r="B3"/></row>
<row r="4"><c r="B4" t="b"><v>1</v></c><c r="C4"><v>0.30000000000000004</v></c><c r="D4" t="str"><f>A1&amp;"x"</f><v>x &lt; y</v></c><c r="F4" t="e"><v>#DIV/0!</v></c></row>
</sheetData></worksheet>"#;
        let sheet3 = r#"<worksheet><sheetData><row r="1"><c r="A1" t="s"><v>2</v></c><c r="B1" t="s"><v>1</v></c></row><row r="2"><c r="A2" t="inlineStr"><is><r><t>multi</t></r><r><t>part</t></r></is></c></row></sheetData></worksheet>"#;
        write_zip(&f, &[("xl/workbook.xml", workbook), ("xl/_rels/workbook.xml.rels", rels), ("xl/sharedStrings.xml", strings), ("xl/styles.xml", styles), ("xl/worksheets/sheet1.xml", sheet1), ("xl/worksheets/sheet2.xml", "<worksheet><sheetData/></worksheet>"), ("xl/worksheets/sheet3.xml", sheet3)]);
        assert!(is_office(&f));
        let t = read_document(&f, 1 << 20).unwrap();
        assert_eq!(t, "Sheet: Sales & Costs\nItem\tQty\t\tTotal & tax\nApples\t3\t2023-03-15\t2023-03-15 12:00\t5.00 days\n\tTRUE\t0.3\tx < y\t\t#DIV/0!\n\nSheet: Notes\n\tApples\nmultipart", "{t:?}");
        // without the relationships file the sheets are the parts sheet1.xml, sheet2.xml ... in the order of the workbook
        let g = dir.join("plain.xlsx");
        write_zip(&g, &[("xl/workbook.xml", r#"<workbook><sheets><sheet name="Only"/></sheets></workbook>"#), ("xl/worksheets/sheet1.xml", r#"<worksheet><sheetData><row><c t="inlineStr"><is><t>a</t></is></c><c><v>2</v></c></row></sheetData></worksheet>"#)]);
        assert_eq!(read_document(&g, 1 << 20).unwrap(), "Sheet: Only\na\t2");
        // not a workbook, and a workbook without a cell of text
        let bad = dir.join("bad.xlsx");
        write_zip(&bad, &[("other.xml", "<x/>")]);
        assert!(read_document(&bad, 1 << 20).unwrap_err().contains("xl/workbook.xml is missing"));
        let empty = dir.join("empty.xlsx");
        write_zip(&empty, &[("xl/workbook.xml", r#"<workbook><sheets><sheet name="A"/></sheets></workbook>"#), ("xl/worksheets/sheet1.xml", "<worksheet><sheetData/></worksheet>")]);
        assert!(read_document(&empty, 1 << 20).unwrap_err().contains("holds no text"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn excel_dates_numbers_and_columns() {
        assert_eq!(excel_date(45000.0), "2023-03-15");
        assert_eq!(excel_date(45000.5), "2023-03-15 12:00");
        assert_eq!(excel_date(45000.0 + 45296.0 / 86400.0), "2023-03-15 12:34:56");
        assert_eq!(excel_date(25569.0), "1970-01-01");
        assert_eq!(excel_date(36526.0), "2000-01-01");
        assert_eq!(excel_date(2958465.0), "9999-12-31");
        // Excel's calendar has a 29 February 1900: before March the serials are one day later than the real calendar
        assert_eq!((excel_date(1.0), excel_date(59.0), excel_date(61.0)), ("1900-01-01".to_string(), "1900-02-28".to_string(), "1900-03-01".to_string()));
        assert_eq!(excel_date(0.75), "18:00");
        assert!(is_date_format(14, None) && is_date_format(22, None) && !is_date_format(0, None) && !is_date_format(9, None));
        assert!(is_date_format(164, Some("yyyy\\-mm\\-dd")) && is_date_format(166, Some("[$-409]h:mm AM/PM")) && is_date_format(171, Some("[h]:mm:ss")));
        for not in ["General", "0.00\" days\"", "#,##0.00_);[Red](#,##0.00)", "0.0%", "@", "0.00E+00"] {
            assert!(!is_date_format(170, Some(not)), "{not}");
        }
        assert_eq!((raw_number_text("0.30000000000000004", false), raw_number_text("2.9999999999999996", false), raw_number_text("-0.30000000000000004", false)), ("0.3".to_string(), "3".to_string(), "-0.3".to_string()));
        assert_eq!((raw_number_text("12.5", false), raw_number_text("1E-3", false), raw_number_text("100", false), raw_number_text("abc", true)), ("12.5".to_string(), "1E-3".to_string(), "100".to_string(), "abc".to_string()));
        assert_eq!(raw_number_text("45000", true), "2023-03-15");
        assert_eq!(raw_number_text("-5", true), "-5");
        assert_eq!((column_of("A1"), column_of("B7"), column_of("AA3"), column_of("XFD9"), column_of("XFE1"), column_of("12"), column_of("ABCD1")), (Some(0), Some(1), Some(26), Some(16383), None, None, None));
    }

    /// What Excel itself shows (its CSV export of a workbook with these formats) for each value and format code.
    #[test]
    fn numbers_are_shown_as_excel_shows_them() {
        let cases: [(f64, &str, Option<&str>); 31] = [
            (1234.5678, "General", None),
            (1234.5678, "0", Some("1235")),
            (1234.5678, "0.00", Some("1234.57")),
            (1234.5678, "#,##0", Some("1,235")),
            (1234.5678, "#,##0.00", Some("1,234.57")),
            (0.256, "0%", Some("26%")),
            (0.256, "0.00%", Some("25.60%")),
            (1234.5, "$#,##0.00", Some("$1,234.50")),
            (1234.5, "\"$\"#,##0", Some("$1,235")),
            (-1234.5, "#,##0.00", Some("-1,234.50")),
            (-1234.5, "$#,##0.00;($#,##0.00)", Some("($1,234.50)")),
            (0.5, "0.0%", Some("50.0%")),
            (1234567.0, "#,##0", Some("1,234,567")),
            (5.0, "00000", Some("00005")),
            (3.14159, "0.000", Some("3.142")),
            (1234.5, "[$\u{20ac}-407] #,##0.00", Some("\u{20ac} 1,234.50")),
            (1234.5, "#,##0.00 \"USD\"", Some("1,234.50 USD")),
            (12.0, "0 \"items\"", Some("12 items")),
            (0.0, "0.00;-0.00;\"zero\"", Some("zero")),
            (1234.5, "@", None),
            (0.000123, "0.00E+00", Some("1.23E-04")),
            (1234.5, "#,##0.00_);(#,##0.00)", Some("1,234.50 ")),
            (2.5, "0.0", Some("2.5")),
            // not in the workbook above: optional decimals, a number without its leading zero, the built-in accounting-like code, a fraction (not understood), a scaling comma
            (2.25, "0.0#", Some("2.25")),
            (3.0, "0.0#", Some("3.0")),
            (0.5, "#.##", Some(".5")),
            (-5.0, "#,##0 ;(#,##0)", Some("(5)")),
            (0.5, "# ?/?", None),
            (1234567.0, "#,##0,", None),
            (-0.5, "0%", Some("-50%")),
            (1e21, "0", Some("1000000000000000000000")),
        ];
        for (x, code, want) in cases {
            assert_eq!(format_number(x, code).as_deref(), want, "{x} in {code:?}");
        }
        // a cell uses the code of its style; a date style wins; unknown styles show the number as stored
        let mut styles = CellStyles::default();
        styles.numbers.insert(1, "0.0%".to_string());
        styles.numbers.insert(2, "# ?/?".to_string());
        styles.dates.insert(3);
        assert_eq!((number_text("0.256", 1, &styles), number_text("0.5", 2, &styles), number_text("45000", 3, &styles), number_text("0.30000000000000004", 9, &styles)), ("25.6%".to_string(), "0.5".to_string(), "2023-03-15".to_string(), "0.3".to_string()));
        // the built-in codes and a custom one through a workbook's styles
        let xml = r##"<styleSheet><numFmts count="1"><numFmt numFmtId="170" formatCode="&quot;$&quot;#,##0.00"/></numFmts><cellXfs count="4"><xf numFmtId="0"/><xf numFmtId="9"/><xf numFmtId="4"/><xf numFmtId="170"/></cellXfs></styleSheet>"##;
        let s = cell_styles(xml);
        assert_eq!((s.numbers.get(&0), s.numbers.get(&1).map(String::as_str), s.numbers.get(&2).map(String::as_str), s.numbers.get(&3).map(String::as_str)), (None, Some("0%"), Some("#,##0.00"), Some("\"$\"#,##0.00")));
    }

    #[test]
    fn an_opendocument_spreadsheet_gives_its_sheets() {
        let dir = temp_dir("ods");
        let f = dir.join("budget.ods");
        let content = r#"<office:document-content><office:body><office:spreadsheet>
<table:table table:name="Budget &amp; Plan">
  <table:table-row><table:table-cell office:value-type="string"><text:p>Item</text:p></table:table-cell><table:table-cell office:value-type="float" office:value="3"><text:p>3</text:p></table:table-cell><table:table-cell table:number-columns-repeated="2"/><table:table-cell office:value-type="date" office:date-value="2023-03-15"><text:p>03/15/23</text:p></table:table-cell></table:table-row>
  <table:table-row table:number-rows-repeated="1048000"><table:table-cell table:number-columns-repeated="1024"/></table:table-row>
  <table:table-row><table:table-cell><text:p>pre-</text:p></table:table-cell></table:table-row>
  <table:table-row><table:table-cell><text:p>next</text:p></table:table-cell></table:table-row>
</table:table>
<table:table table:name="Blank"><table:table-row><table:table-cell/></table:table-row></table:table>
<table:table table:name="Second"><table:table-row><table:table-cell><text:p>a</text:p><text:p>b</text:p></table:table-cell><table:table-cell><text:p>x<text:s text:c="2"/>y<text:tab/>z</text:p></table:table-cell></table:table-row></table:table>
</office:spreadsheet></office:body></office:document-content>"#;
        write_zip(&f, &[("mimetype", "application/vnd.oasis.opendocument.spreadsheet"), ("content.xml", content)]);
        assert!(is_office(&f));
        let t = read_document(&f, 1 << 20).unwrap();
        // (a row that ends with a hyphen does not swallow the next row; a sheet without text has no heading)
        assert_eq!(t, "Sheet: Budget & Plan\nItem\t3\t\t\t03/15/23\npre-\nnext\n\nSheet: Second\na b\tx  y z", "{t:?}");
        let bad = dir.join("bad.ods");
        write_zip(&bad, &[("other.xml", "<x/>")]);
        assert!(read_document(&bad, 1 << 20).unwrap_err().contains("content.xml is missing"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_opendocument_text_gives_its_paragraphs_lists_and_tables() {
        let dir = temp_dir("odt");
        let f = dir.join("harbour.odt");
        let content = r#"<office:document-content><office:body><office:text>
<text:tracked-changes><text:changed-region><text:deletion><text:p>deleted words</text:p></text:deletion></text:changed-region></text:tracked-changes>
<text:h text:outline-level="1">The &quot;Harbour&quot;</text:h>
<text:p text:style-name="P1">Ships <text:span text:style-name="T1">dock</text:span> at six<text:tab/>sharp,<text:line-break/>every   day &amp; night.<office:annotation><dc:creator>Sam</dc:creator><text:p>check this</text:p></office:annotation></text:p>
<text:list><text:list-item><text:p>tea</text:p></text:list-item><text:list-item><text:p>rope</text:p></text:list-item></text:list>
<text:p/>
<table:table table:name="Table1"><table:table-row><table:table-cell><text:p>a</text:p></table:table-cell><table:table-cell><text:p>b</text:p></table:table-cell></table:table-row></table:table>
<text:p>End.</text:p>
</office:text></office:body></office:document-content>"#;
        write_zip(&f, &[("content.xml", content)]);
        assert_eq!(read_document(&f, 1 << 20).unwrap(), "The \"Harbour\"\nShips dock at six\tsharp,\nevery day & night.\ntea\nrope\na\tb\nEnd.");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_opendocument_presentation_gives_its_slides() {
        let dir = temp_dir("odp");
        let f = dir.join("deck.odp");
        let content = r#"<office:document-content><office:body><office:presentation>
<draw:page draw:name="page1"><draw:frame><draw:text-box><text:p>Title</text:p><text:p>Point one</text:p></draw:text-box></draw:frame></draw:page>
<draw:page draw:name="page2"><draw:frame><draw:text-box><text:p>Second</text:p></draw:text-box></draw:frame></draw:page>
</office:presentation></office:body></office:document-content>"#;
        write_zip(&f, &[("content.xml", content)]);
        assert_eq!(read_document(&f, 1 << 20).unwrap(), "Slide 1\nTitle\nPoint one\n\nSlide 2\nSecond");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_rtf_file_gives_its_text() {
        let dir = temp_dir("rtf");
        let f = dir.join("letter.rtf");
        let rtf = r#"{\rtf1\ansi\ansicpg1252\deff0{\fonttbl{\f0\fswiss Arial;}}{\colortbl;\red255\green0\blue0;}{\*\generator Riched20 10.0;}\viewkind4\uc1\pard\f0\fs20 The \b harbour\b0  at \'e9t\'e9: caf@u233?, @u8212? and @u-10179?@u-8704? (emoji).\par
Second\tab line\line broken \{braces\} \\ end.\par
{\pict\wmetafile8 0102030405}{\field{\*\fldinst HYPERLINK "http://x"}{\fldrslt link text}}\par
{\header ignored header}{\info{\title Not text}}\trowd\intbl one\cell two\cell\row
}"#;
        // (the \uN words are written with an @ here: a backslash and four digits in a source text can be taken for an escape by the tools that write it)
        std::fs::write(&f, rtf.replace("@u", "\\u")).unwrap();
        assert!(is_office(&f));
        assert_eq!(read_document(&f, 1 << 20).unwrap(), "The harbour at \u{e9}t\u{e9}: caf\u{e9}, \u{2014} and \u{1f600} (emoji).\nSecond\tline\nbroken {braces} \\ end.\nlink text\none\ttwo");
        // a file with nothing but formatting, and a cut one: no panic
        let empty = dir.join("empty.rtf");
        std::fs::write(&empty, r"{\rtf1\ansi{\fonttbl{\f0 Arial;}}\pard\par}").unwrap();
        assert!(read_document(&empty, 1 << 20).unwrap_err().contains("holds no text"));
        let cut = dir.join("cut.rtf");
        std::fs::write(&cut, r"{\rtf1\ansi Some text \u82").unwrap();
        assert!(read_document(&cut, 1 << 20).unwrap().contains("Some text"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Pieces of markup and escapes in random order: no reader may panic or hang on them.
    #[test]
    fn random_markup_never_panics_a_reader() {
        let pieces = [
            "<", ">", "/", "/>", "text:p", "text:h", "text:s text:c=\"5\"", "text:tab", "table:table", "table:table-row", "table:table-cell", " table:number-columns-repeated=\"9\"", " table:name=\"S\"", "draw:page", "office:annotation",
            "</office:annotation>", "text:tracked-changes", "c", " r=\"B7\"", " r=\"ZZZZ1\"", " t=\"s\"", " t=\"inlineStr\"", " s=\"3\"", "v", "is", "t", "rPh", "row", "si", "xf ", "numFmt ", " numFmtId=\"14\"", " formatCode=\"yy\"", "cellXfs",
            "&amp;", "&#x41;", "&#99999999;", "&", ";", "\\", "{", "}", "'", "u", "-", "9", "bin", "par", "*", "fonttbl", "uc", " ", "\u{e9}", "\u{1f600}", "\n", "a", "\"",
        ];
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // (FORGE_FUZZ=400000 runs it much longer)
        let rounds = std::env::var("FORGE_FUZZ").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(6000);
        for _ in 0..rounds {
            let mut s = String::new();
            for _ in 0..(next() % 40) {
                s.push_str(pieces[(next() as usize) % pieces.len()]);
            }
            for kind in [Odf::Text, Odf::Sheet, Odf::Slides] {
                let _ = odf_text(&s, kind);
            }
            let _ = rtf_text(s.as_bytes());
            let _ = sheet_rows(&s, &["a".to_string()], &CellStyles { dates: HashSet::from([3usize]), numbers: HashMap::from([(2usize, "0.00%".to_string())]) });
            let _ = shared_strings(&s);
            let _ = cell_styles(&s);
            let _ = html_text(&s);
            let _ = xml_text(&s, "w:t", "w:p");
        }
    }

    #[test]
    fn a_picture_is_a_document_whose_text_the_system_reads() {
        let dir = temp_dir("picture");
        for name in ["shot.png", "photo.JPG", "scan.tiff"] {
            let f = dir.join(name);
            std::fs::write(&f, b"not really a picture").unwrap();
            assert!(is_office(&f), "{name}");
            // a file that is not a picture, or a system with no text recognition: an error that says why, never a panic
            let e = read_document(&f, 1 << 20).unwrap_err();
            assert!(e.starts_with(name), "{e}");
        }
        assert!(!is_office(&dir.join("notes.txt")));
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
