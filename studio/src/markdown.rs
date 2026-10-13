//! A small Markdown parser and its egui renderer, for chat messages: headings, emphasis, inline code, fenced code, lists, quotes, tables,
//! links and rules. The parser never fails: text it does not understand stays text, and a fence that is still open (a streaming reply) is a code block up to the end.

use eframe::egui::{self, text::LayoutJob, Color32, CornerRadius, FontId, Frame, Margin, RichText, Sense, Stroke, TextFormat, Ui};
use std::cell::Cell;

#[derive(Clone, Debug, PartialEq)]
pub enum Span {
    Text(String),
    Bold(Vec<Span>),
    Italic(Vec<Span>),
    Strike(Vec<Span>),
    Code(String),
    Link { text: Vec<Span>, url: String },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Block {
    Heading(u8, Vec<Span>),
    Paragraph(Vec<Span>),
    Code { lang: String, text: String },
    List { ordered: bool, start: u64, items: Vec<Vec<Block>> },
    Quote(Vec<Block>),
    Rule,
    Table { header: Vec<Vec<Span>>, rows: Vec<Vec<Vec<Span>>> },
}

// ------------------------------------------------------------------------------------------------------------------ blocks

pub fn parse(src: &str) -> Vec<Block> {
    let text = src.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<&str> = text.split('\n').collect();
    parse_lines(&lines)
}

fn indent_of(line: &str) -> usize {
    let mut n = 0;
    for c in line.chars() {
        match c {
            ' ' => n += 1,
            '\t' => n += 4,
            _ => break,
        }
    }
    n
}

/// Drop up to `n` columns of leading whitespace.
fn strip_indent(line: &str, n: usize) -> &str {
    let mut cols = 0;
    for (i, c) in line.char_indices() {
        if cols >= n {
            return &line[i..];
        }
        match c {
            ' ' => cols += 1,
            '\t' => cols += 4,
            _ => return &line[i..],
        }
    }
    ""
}

fn fence_start(trimmed: &str) -> Option<(char, usize, String)> {
    let c = trimmed.chars().next()?;
    if c != '`' && c != '~' {
        return None;
    }
    let n = trimmed.chars().take_while(|&x| x == c).count();
    if n < 3 {
        return None;
    }
    let info = trimmed[n..].trim();
    if c == '`' && info.contains('`') {
        return None;
    }
    let lang = info.split_whitespace().next().unwrap_or("").to_string();
    Some((c, n, lang))
}

fn is_fence_end(trimmed: &str, c: char, n: usize) -> bool {
    let k = trimmed.chars().take_while(|&x| x == c).count();
    k >= n && trimmed[k..].trim().is_empty()
}

fn heading(trimmed: &str) -> Option<(u8, &str)> {
    let n = trimmed.chars().take_while(|&c| c == '#').count();
    if n == 0 || n > 6 {
        return None;
    }
    let rest = &trimmed[n..];
    if !rest.is_empty() && !rest.starts_with(' ') && !rest.starts_with('\t') {
        return None;
    }
    let text = rest.trim().trim_end_matches('#').trim_end();
    Some((n as u8, text))
}

fn is_rule(trimmed: &str) -> bool {
    let t: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
    t.len() >= 3 && (t.chars().all(|c| c == '-') || t.chars().all(|c| c == '*') || t.chars().all(|c| c == '_'))
}

struct Marker {
    indent: usize,
    ordered: bool,
    number: u64,
    /// Column where the item's text starts.
    content: usize,
    /// Byte offset in the line where the item's text starts.
    text: usize,
}

fn list_marker(line: &str) -> Option<Marker> {
    let indent = indent_of(line);
    let rest = line.trim_start();
    let start = line.len() - rest.len();
    let mut chars = rest.chars();
    let first = chars.next()?;
    if matches!(first, '-' | '*' | '+') {
        let after = &rest[1..];
        if after.is_empty() || after.starts_with(' ') || after.starts_with('\t') {
            let gap = if after.is_empty() { 0 } else { 1 };
            return Some(Marker { indent, ordered: false, number: 0, content: indent + 2, text: start + 1 + gap });
        }
        return None;
    }
    if first.is_ascii_digit() {
        let digits = rest.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits > 9 {
            return None;
        }
        let after = &rest[digits..];
        let mut it = after.chars();
        let d = it.next()?;
        if d != '.' && d != ')' {
            return None;
        }
        let tail = &after[1..];
        if tail.is_empty() || tail.starts_with(' ') || tail.starts_with('\t') {
            let number = rest[..digits].parse().unwrap_or(1);
            let gap = if tail.is_empty() { 0 } else { 1 };
            return Some(Marker { indent, ordered: true, number, content: indent + digits + 2, text: start + digits + 1 + gap });
        }
    }
    None
}

fn split_cells(line: &str) -> Vec<String> {
    let t = line.trim();
    let t = t.strip_prefix('|').unwrap_or(t);
    let t = t.strip_suffix('|').unwrap_or(t);
    let mut cells = Vec::new();
    let mut cur = String::new();
    let mut esc = false;
    for c in t.chars() {
        if esc {
            if c != '|' {
                cur.push('\\');
            }
            cur.push(c);
            esc = false;
        } else if c == '\\' {
            esc = true;
        } else if c == '|' {
            cells.push(cur.trim().to_string());
            cur.clear();
        } else {
            cur.push(c);
        }
    }
    cells.push(cur.trim().to_string());
    cells
}

fn is_table_separator(line: &str) -> bool {
    let t = line.trim();
    t.contains('|') && t.contains('-') && t.chars().all(|c| matches!(c, '|' | '-' | ':' | ' ' | '\t'))
}

fn starts_block(line: &str) -> bool {
    let trimmed = line.trim_start();
    fence_start(trimmed).is_some() || heading(trimmed).is_some() || trimmed.starts_with('>') || is_rule(trimmed) || list_marker(line).is_some()
}

fn parse_lines(lines: &[&str]) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();
        if let Some((fc, fnum, lang)) = fence_start(trimmed) {
            let indent = indent_of(line);
            let mut body: Vec<&str> = Vec::new();
            i += 1;
            while i < lines.len() {
                if is_fence_end(lines[i].trim_start(), fc, fnum) {
                    i += 1;
                    break;
                }
                body.push(strip_indent(lines[i], indent));
                i += 1;
            }
            blocks.push(Block::Code { lang, text: body.join("\n") });
            continue;
        }
        if trimmed.is_empty() {
            i += 1;
            continue;
        }
        if let Some((level, text)) = heading(trimmed) {
            blocks.push(Block::Heading(level, parse_inline(text)));
            i += 1;
            continue;
        }
        if is_rule(trimmed) {
            blocks.push(Block::Rule);
            i += 1;
            continue;
        }
        if line.contains('|') && i + 1 < lines.len() && is_table_separator(lines[i + 1]) {
            let header: Vec<Vec<Span>> = split_cells(line).iter().map(|c| parse_inline(c)).collect();
            let cols = header.len();
            let mut rows = Vec::new();
            i += 2;
            while i < lines.len() && lines[i].contains('|') && !lines[i].trim().is_empty() {
                let mut cells: Vec<Vec<Span>> = split_cells(lines[i]).iter().map(|c| parse_inline(c)).collect();
                cells.resize(cols, Vec::new());
                rows.push(cells);
                i += 1;
            }
            blocks.push(Block::Table { header, rows });
            continue;
        }
        if trimmed.starts_with('>') {
            let mut inner: Vec<String> = Vec::new();
            while i < lines.len() && lines[i].trim_start().starts_with('>') {
                let l = lines[i].trim_start()[1..].to_string();
                inner.push(l.strip_prefix(' ').map(str::to_string).unwrap_or(l));
                i += 1;
            }
            let refs: Vec<&str> = inner.iter().map(String::as_str).collect();
            blocks.push(Block::Quote(parse_lines(&refs)));
            continue;
        }
        if let Some(first) = list_marker(line) {
            let (ordered, start) = (first.ordered, first.number);
            let mut items: Vec<Vec<Block>> = Vec::new();
            while i < lines.len() {
                let Some(m) = list_marker(lines[i]) else { break };
                if m.indent + 1 < first.indent || m.indent > first.indent + 1 || m.ordered != ordered {
                    break;
                }
                let content = m.content;
                let mut item_lines: Vec<String> = vec![lines[i][m.text.min(lines[i].len())..].to_string()];
                i += 1;
                let mut blanks = 0;
                while i < lines.len() {
                    let l = lines[i];
                    if l.trim().is_empty() {
                        blanks += 1;
                        i += 1;
                        continue;
                    }
                    if indent_of(l) >= content {
                        // part of the item: its paragraphs, nested lists, code
                        for _ in 0..blanks {
                            item_lines.push(String::new());
                        }
                        blanks = 0;
                        item_lines.push(strip_indent(l, content).to_string());
                        i += 1;
                    } else if blanks == 0 && !starts_block(l) {
                        // a lazy continuation line of the item's paragraph
                        item_lines.push(l.trim_start().to_string());
                        i += 1;
                    } else {
                        break;
                    }
                }
                let refs: Vec<&str> = item_lines.iter().map(String::as_str).collect();
                items.push(parse_lines(&refs));
            }
            blocks.push(Block::List { ordered, start, items });
            continue;
        }
        // a paragraph: lines up to a blank line or the start of another block
        let mut para: Vec<&str> = vec![trimmed];
        i += 1;
        while i < lines.len() {
            let l = lines[i];
            if l.trim().is_empty() || starts_block(l) || (l.contains('|') && i + 1 < lines.len() && is_table_separator(lines[i + 1])) {
                break;
            }
            para.push(l.trim_start());
            i += 1;
        }
        blocks.push(Block::Paragraph(parse_inline(&para.join("\n"))));
    }
    blocks
}

// ------------------------------------------------------------------------------------------------------------------ inline

pub fn parse_inline(s: &str) -> Vec<Span> {
    let chars: Vec<char> = s.chars().collect();
    inline(&chars)
}

fn is_escapable(c: char) -> bool {
    "\\`*_{}[]()#+-.!|~<>$".contains(c)
}

fn flush(out: &mut Vec<Span>, text: &mut String) {
    if !text.is_empty() {
        out.push(Span::Text(std::mem::take(text)));
    }
}

fn run_len(c: &[char], i: usize, ch: char) -> usize {
    c[i..].iter().take_while(|&&x| x == ch).count()
}

/// Index of the closing delimiter run of `n` characters `ch` after `from`, honouring the flanking rules of emphasis.
fn find_closer(c: &[char], from: usize, ch: char, n: usize) -> Option<usize> {
    let mut j = from;
    while j + n <= c.len() {
        if c[j] == '`' {
            // skip a code span
            let k = run_len(c, j, '`');
            if let Some(end) = find_backticks(c, j + k, k) {
                j = end + k;
                continue;
            }
        }
        if c[j] == '\\' {
            j += 2;
            continue;
        }
        if c[j] == ch && c[j..j + n].iter().all(|&x| x == ch) {
            let run = run_len(c, j, ch);
            let before_ok = j > from && !c[j - 1].is_whitespace();
            let after = c.get(j + run);
            let after_ok = ch != '_' || after.map_or(true, |a| !a.is_alphanumeric());
            if run >= n && before_ok && after_ok {
                // the closer is the last n characters of the run, so ***text*** works for both
                return Some(j + run - n);
            }
            j += run;
            continue;
        }
        j += 1;
    }
    None
}

fn find_backticks(c: &[char], from: usize, n: usize) -> Option<usize> {
    let mut j = from;
    while j < c.len() {
        if c[j] == '`' {
            let k = run_len(c, j, '`');
            if k == n {
                return Some(j);
            }
            j += k;
        } else {
            j += 1;
        }
    }
    None
}

fn inline(c: &[char]) -> Vec<Span> {
    let mut out: Vec<Span> = Vec::new();
    let mut text = String::new();
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        match ch {
            '\\' if i + 1 < c.len() && is_escapable(c[i + 1]) => {
                text.push(c[i + 1]);
                i += 2;
            }
            '`' => {
                let n = run_len(c, i, '`');
                if let Some(end) = find_backticks(c, i + n, n) {
                    flush(&mut out, &mut text);
                    let mut code: String = c[i + n..end].iter().collect();
                    code = code.replace('\n', " ");
                    if code.len() > 2 && code.starts_with(' ') && code.ends_with(' ') && code.trim().len() > 0 {
                        code = code[1..code.len() - 1].to_string();
                    }
                    out.push(Span::Code(code));
                    i = end + n;
                } else {
                    text.extend(std::iter::repeat('`').take(n));
                    i += n;
                }
            }
            '*' | '_' => {
                let run = run_len(c, i, ch);
                let prev_ok = ch != '_' || i == 0 || !c[i - 1].is_alphanumeric();
                let next_ok = c.get(i + run).map_or(false, |x| !x.is_whitespace());
                if prev_ok && next_ok {
                    // longest first: bold+italic (3), bold (2), italic (1)
                    let mut done = false;
                    let mut cands: Vec<usize> = Vec::new();
                    if run >= 3 {
                        cands.push(3);
                    }
                    if run >= 2 {
                        cands.push(2);
                    }
                    cands.push(1);
                    for n in cands {
                        if let Some(close) = find_closer(c, i + n, ch, n) {
                            if close > i + n {
                                flush(&mut out, &mut text);
                                let inner = inline(&c[i + n..close]);
                                out.push(match n {
                                    3 => Span::Bold(vec![Span::Italic(inner)]),
                                    2 => Span::Bold(inner),
                                    _ => Span::Italic(inner),
                                });
                                i = close + n;
                                done = true;
                                break;
                            }
                        }
                    }
                    if done {
                        continue;
                    }
                }
                text.extend(std::iter::repeat(ch).take(run));
                i += run;
            }
            '~' if c.get(i + 1) == Some(&'~') => {
                if let Some(close) = find_closer(c, i + 2, '~', 2) {
                    flush(&mut out, &mut text);
                    out.push(Span::Strike(inline(&c[i + 2..close])));
                    i = close + 2;
                } else {
                    text.push_str("~~");
                    i += 2;
                }
            }
            '[' => {
                // [text](url)
                let mut depth = 0;
                let mut close = None;
                for (k, &x) in c.iter().enumerate().skip(i) {
                    if x == '[' {
                        depth += 1;
                    } else if x == ']' {
                        depth -= 1;
                        if depth == 0 {
                            close = Some(k);
                            break;
                        }
                    }
                }
                let mut linked = false;
                if let Some(cl) = close {
                    if c.get(cl + 1) == Some(&'(') {
                        if let Some(rel) = c[cl + 2..].iter().position(|&x| x == ')') {
                            let url: String = c[cl + 2..cl + 2 + rel].iter().collect();
                            let url = url.split_whitespace().next().unwrap_or("").to_string();
                            if !url.is_empty() {
                                flush(&mut out, &mut text);
                                out.push(Span::Link { text: inline(&c[i + 1..cl]), url });
                                i = cl + 2 + rel + 1;
                                linked = true;
                            }
                        }
                    }
                }
                if !linked {
                    text.push('[');
                    i += 1;
                }
            }
            'h' if (c[i..].starts_with(&['h', 't', 't', 'p', ':', '/', '/']) || c[i..].starts_with(&['h', 't', 't', 'p', 's', ':', '/', '/'])) && (i == 0 || !c[i - 1].is_alphanumeric()) => {
                let mut end = i;
                while end < c.len() && !c[end].is_whitespace() && !matches!(c[end], '<' | '>' | '"' | ')' | ']') {
                    end += 1;
                }
                while end > i && matches!(c[end - 1], '.' | ',' | ';' | ':' | '!' | '?') {
                    end -= 1;
                }
                let url: String = c[i..end].iter().collect();
                flush(&mut out, &mut text);
                out.push(Span::Link { text: vec![Span::Text(url.clone())], url });
                i = end;
            }
            _ => {
                text.push(ch);
                i += 1;
            }
        }
    }
    flush(&mut out, &mut text);
    out
}

// ------------------------------------------------------------------------------------------------------------------ rendering

#[derive(Clone, Copy)]
pub struct MdStyle {
    pub size: f32,
    pub text: Color32,
    pub strong: Color32,
    pub muted: Color32,
    pub accent: Color32,
    pub code_bg: Color32,
    pub border: Color32,
}

struct Ctx<'a> {
    style: &'a MdStyle,
    salt: u64,
    counter: Cell<u32>,
    bold: egui::FontFamily,
}

impl Ctx<'_> {
    fn id(&self, ui: &Ui) -> egui::Id {
        let n = self.counter.get();
        self.counter.set(n + 1);
        ui.id().with(("md", self.salt, n))
    }
}

#[derive(Clone, Copy, Default)]
struct Fmt {
    bold: bool,
    italic: bool,
    strike: bool,
    code: bool,
    link: bool,
}

fn add_spans(job: &mut LayoutJob, links: &mut Vec<(usize, usize, String)>, pos: &mut usize, spans: &[Span], fmt: Fmt, size: f32, ctx: &Ctx) {
    for span in spans {
        match span {
            Span::Text(t) => append(job, pos, t, fmt, size, ctx),
            Span::Code(t) => append(job, pos, t, Fmt { code: true, ..fmt }, size, ctx),
            Span::Bold(inner) => add_spans(job, links, pos, inner, Fmt { bold: true, ..fmt }, size, ctx),
            Span::Italic(inner) => add_spans(job, links, pos, inner, Fmt { italic: true, ..fmt }, size, ctx),
            Span::Strike(inner) => add_spans(job, links, pos, inner, Fmt { strike: true, ..fmt }, size, ctx),
            Span::Link { text, url } => {
                let start = *pos;
                add_spans(job, links, pos, text, Fmt { link: true, ..fmt }, size, ctx);
                links.push((start, *pos, url.clone()));
            }
        }
    }
}

fn append(job: &mut LayoutJob, pos: &mut usize, t: &str, fmt: Fmt, size: f32, ctx: &Ctx) {
    let s = ctx.style;
    let color = if fmt.link {
        s.accent
    } else if fmt.bold {
        s.strong
    } else {
        s.text
    };
    let font_id = if fmt.code {
        FontId::monospace(size * 0.92)
    } else if fmt.bold {
        FontId::new(size, ctx.bold.clone())
    } else {
        FontId::proportional(size)
    };
    let format = TextFormat {
        font_id,
        color,
        background: if fmt.code { s.code_bg } else { Color32::TRANSPARENT },
        italics: fmt.italic,
        underline: if fmt.link { Stroke::new(1.0, s.accent) } else { Stroke::NONE },
        strikethrough: if fmt.strike { Stroke::new(1.0, color) } else { Stroke::NONE },
        ..Default::default()
    };
    job.append(t, 0.0, format);
    *pos += t.chars().count();
}

fn rich_paragraph(ui: &mut Ui, spans: &[Span], size: f32, fmt: Fmt, ctx: &Ctx) {
    let mut job = LayoutJob::default();
    let mut links = Vec::new();
    let mut pos = 0;
    add_spans(&mut job, &mut links, &mut pos, spans, fmt, size, ctx);
    if job.text.is_empty() {
        return;
    }
    let width = ui.available_width();
    let resp = ui.add(egui::Label::new(job.clone()).wrap().sense(Sense::click()));
    if links.is_empty() {
        return;
    }
    if let Some(p) = resp.hover_pos() {
        let mut j = job;
        j.wrap.max_width = width;
        let galley = ui.fonts_mut(|f| f.layout_job(j));
        let idx: usize = galley.cursor_from_pos(p - resp.rect.min).index.0;
        if let Some((_, _, url)) = links.iter().find(|(a, b, _)| idx >= *a && idx < *b) {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            if resp.clicked() {
                ui.ctx().open_url(egui::OpenUrl::new_tab(url));
            }
            resp.on_hover_text(url.as_str());
        }
    }
}

fn plain_text(spans: &[Span]) -> String {
    let mut s = String::new();
    for sp in spans {
        match sp {
            Span::Text(t) | Span::Code(t) => s.push_str(t),
            Span::Bold(i) | Span::Italic(i) | Span::Strike(i) => s.push_str(&plain_text(i)),
            Span::Link { text, .. } => s.push_str(&plain_text(text)),
        }
    }
    s
}

fn render_block(ui: &mut Ui, b: &Block, ctx: &Ctx) {
    let s = ctx.style;
    match b {
        Block::Heading(level, spans) => {
            let size = match level {
                1 => s.size * 1.55,
                2 => s.size * 1.35,
                3 => s.size * 1.18,
                _ => s.size * 1.05,
            };
            ui.add_space(4.0);
            rich_paragraph(ui, spans, size, Fmt { bold: true, ..Default::default() }, ctx);
        }
        Block::Paragraph(spans) => rich_paragraph(ui, spans, s.size, Fmt::default(), ctx),
        Block::Rule => {
            ui.add_space(2.0);
            ui.separator();
        }
        Block::Code { lang, text } => code_block(ui, lang, text, ctx),
        Block::Quote(inner) => {
            let r = Frame::new().inner_margin(Margin { left: 12, right: 0, top: 2, bottom: 2 }).show(ui, |ui| {
                ui.set_width(ui.available_width());
                render_blocks_inner(ui, inner, ctx);
            });
            let rect = r.response.rect;
            ui.painter().line_segment([rect.left_top(), rect.left_bottom()], Stroke::new(3.0, s.border));
        }
        Block::List { ordered, start, items } => {
            for (n, item) in items.iter().enumerate() {
                ui.horizontal_top(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    ui.add_space(6.0);
                    let marker = if *ordered { format!("{}.", start + n as u64) } else { "\u{2022}".to_string() };
                    ui.add_sized([if *ordered { 26.0 } else { 14.0 }, s.size * 1.3], egui::Label::new(RichText::new(marker).size(s.size).color(s.muted)));
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 4.0;
                        render_blocks_inner(ui, item, ctx);
                    });
                });
            }
        }
        Block::Table { header, rows } => {
            let id = ctx.id(ui);
            Frame::new().stroke(Stroke::new(1.0, s.border)).corner_radius(CornerRadius::same(6)).inner_margin(Margin::same(6)).show(ui, |ui| {
                egui::Grid::new(id).striped(true).spacing([14.0, 5.0]).show(ui, |ui| {
                    for cell in header {
                        ui.label(RichText::new(plain_text(cell)).strong().color(s.strong).size(s.size));
                    }
                    ui.end_row();
                    for row in rows {
                        for cell in row {
                            rich_paragraph(ui, cell, s.size, Fmt::default(), ctx);
                        }
                        ui.end_row();
                    }
                });
            });
        }
    }
}

fn code_block(ui: &mut Ui, lang: &str, text: &str, ctx: &Ctx) {
    let s = ctx.style;
    let id = ctx.id(ui);
    Frame::new().fill(s.code_bg).stroke(Stroke::new(1.0, s.border)).corner_radius(CornerRadius::same(8)).inner_margin(Margin::same(8)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(RichText::new(if lang.is_empty() { "code" } else { lang }).small().color(s.muted));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("Copy").clicked() {
                    ui.ctx().copy_text(text.to_string());
                }
            });
        });
        egui::ScrollArea::horizontal().id_salt(id).auto_shrink([false, true]).show(ui, |ui| {
            ui.add(egui::Label::new(RichText::new(text).monospace().size(s.size * 0.92).color(s.text)).extend());
        });
    });
}

fn render_blocks_inner(ui: &mut Ui, blocks: &[Block], ctx: &Ctx) {
    for (i, b) in blocks.iter().enumerate() {
        if i > 0 {
            ui.add_space(6.0);
        }
        render_block(ui, b, ctx);
    }
}

/// Draw parsed blocks. `salt` keeps the ids of scroll areas and tables of different messages apart.
pub fn render(ui: &mut Ui, blocks: &[Block], style: &MdStyle, salt: u64, bold: egui::FontFamily) {
    let ctx = Ctx { style, salt, counter: Cell::new(0), bold };
    ui.spacing_mut().item_spacing.y = 6.0;
    render_blocks_inner(ui, blocks, &ctx);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str) -> Span {
        Span::Text(s.to_string())
    }

    #[test]
    fn headings_paragraphs_and_rules() {
        let b = parse("# Title\n\nSome text\nsecond line\n\n---\n\n### Small");
        assert_eq!(b.len(), 4);
        assert_eq!(b[0], Block::Heading(1, vec![t("Title")]));
        assert_eq!(b[1], Block::Paragraph(vec![t("Some text\nsecond line")]));
        assert_eq!(b[2], Block::Rule);
        assert_eq!(b[3], Block::Heading(3, vec![t("Small")]));
    }

    #[test]
    fn emphasis_code_and_links() {
        let s = parse_inline("a **bold** and *it* and `code` and ~~gone~~ [site](https://x.org/a) end");
        assert_eq!(
            s,
            vec![
                t("a "),
                Span::Bold(vec![t("bold")]),
                t(" and "),
                Span::Italic(vec![t("it")]),
                t(" and "),
                Span::Code("code".into()),
                t(" and "),
                Span::Strike(vec![t("gone")]),
                t(" "),
                Span::Link { text: vec![t("site")], url: "https://x.org/a".into() },
                t(" end"),
            ]
        );
    }

    #[test]
    fn unterminated_markers_stay_text_while_streaming() {
        assert_eq!(parse_inline("half **bold"), vec![t("half **bold")]);
        assert_eq!(parse_inline("snake_case_word and 2 * 3 * 4"), vec![t("snake_case_word and 2 * 3 * 4")]);
        assert_eq!(parse_inline("`open"), vec![t("`open")]);
    }

    #[test]
    fn bare_urls_become_links_without_the_trailing_dot() {
        let s = parse_inline("see https://example.com/a?b=1.");
        assert_eq!(s, vec![t("see "), Span::Link { text: vec![t("https://example.com/a?b=1")], url: "https://example.com/a?b=1".into() }, t(".")]);
    }

    #[test]
    fn fenced_code_open_or_closed() {
        let b = parse("before\n```rust\nfn main() {}\n```\nafter");
        assert_eq!(b[1], Block::Code { lang: "rust".into(), text: "fn main() {}".into() });
        assert_eq!(b.len(), 3);
        let open = parse("```python\nprint(1)\nprint(2)");
        assert_eq!(open, vec![Block::Code { lang: "python".into(), text: "print(1)\nprint(2)".into() }]);
    }

    #[test]
    fn lists_nested_and_ordered() {
        let b = parse("- one\n- two\n  - nested a\n  - nested b\n- three\n\n1. first\n2. second");
        match &b[0] {
            Block::List { ordered, items, .. } => {
                assert!(!ordered);
                assert_eq!(items.len(), 3);
                assert!(matches!(items[1][1], Block::List { .. }));
            }
            other => panic!("{other:?}"),
        }
        match &b[1] {
            Block::List { ordered, start, items } => {
                assert!(*ordered);
                assert_eq!(*start, 1);
                assert_eq!(items.len(), 2);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn quotes_and_tables() {
        let b = parse("> quoted\n> text\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 |");
        assert!(matches!(&b[0], Block::Quote(inner) if inner.len() == 1));
        match &b[1] {
            Block::Table { header, rows } => {
                assert_eq!(header.len(), 2);
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[1].len(), 2);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_star_in_math_is_not_a_list() {
        let b = parse("2 * 3 = 6\n\n* real item");
        assert!(matches!(b[0], Block::Paragraph(_)));
        assert!(matches!(b[1], Block::List { .. }));
    }
}
