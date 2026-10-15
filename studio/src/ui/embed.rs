//! The embedding playground: what the chat screen shows while an embedding model is loaded. Type a few texts, one per line; the first line is the question, the others are ranked by how close their
//! vectors are to it, and every pair gets a similarity (the cosine of the two vectors).

use eframe::egui::{self, Align, Align2, Color32, CornerRadius, FontId, Key, Layout, RichText, ScrollArea, Sense, TextEdit, Ui};

use super::widgets::*;
use crate::app::{Action, EmbedOut, StudioApp};
use crate::client::cosine;
use crate::engine::EngineState;
use crate::theme;
use crate::util::{human_count, truncate};

const COLUMN: f32 = 880.0;

pub fn show(app: &mut StudioApp, ui: &mut Ui) {
    ScrollArea::vertical().id_salt("embed_scroll").auto_shrink([false, false]).show(ui, |ui| {
        let w = ui.available_width().min(COLUMN);
        ui.vertical_centered(|ui| {
            ui.allocate_ui_with_layout(egui::vec2(w, 0.0), Layout::top_down(Align::Min), |ui| body(app, ui));
        });
    });
}

fn body(app: &mut StudioApp, ui: &mut Ui) {
    ui.add_space(20.0);
    let state = app.engine.state();
    let (name, ready) = match &state {
        EngineState::Ready { model, .. } => (model.clone(), true),
        EngineState::Loading { model, .. } => (model.clone(), false),
        _ => (String::new(), false),
    };
    ui.label(RichText::new("Embeddings").size(28.0).color(theme::STRONG).family(theme::bold_family()));
    ui.label(RichText::new(truncate(&name, 80)).color(theme::MUTED));
    ui.add_space(8.0);
    muted(
        ui,
        "An embedding model turns a text into a list of numbers. Texts that mean similar things get lists that point the same way, which is how search, recommendations and \"chat with your documents\" find what is related. \
         Type one text per line: the first line is the question, the others are ranked by how close they are to it. Some models were trained with a prefix per task: \
         nomic-embed-text wants questions to start with \"search_query: \" and the passages to search with \"search_document: \", e5 wants \"query: \" and \"passage: \"; bge, MiniLM and gte need none.",
    );
    ui.add_space(10.0);
    card(ui, |ui| {
        let resp = ui.add(
            TextEdit::multiline(&mut app.embed_text)
                .hint_text("One text per line ...")
                .desired_rows(6)
                .desired_width(ui.available_width())
                .font(egui::TextStyle::Monospace),
        );
        let ctrl_enter = resp.has_focus() && ui.input(|i| i.key_pressed(Key::Enter) && i.modifiers.command);
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let can = ready && !app.embed_busy && !app.embed_text.trim().is_empty();
            if primary_enabled(ui, can, "Embed").on_hover_text("Ctrl+Enter").clicked() || (ctrl_enter && can) {
                app.actions.push(Action::Embed);
            }
            if ui.button("Sample").on_hover_text("Put the example lines back").clicked() {
                app.embed_text = crate::rag::sample_for(&name);
            }
            if app.embed_busy {
                ui.spinner();
            } else if !ready {
                ui.label(RichText::new("loading the model ...").color(theme::WARN));
            }
            let lines = app.embed_text.lines().filter(|l| !l.trim().is_empty()).count();
            right_aligned(ui, |ui| {
                ui.label(RichText::new(format!("{lines} text{}", if lines == 1 { "" } else { "s" })).small().color(theme::MUTED));
                if let Some(out) = &app.embed_out {
                    if ui.button("Copy vectors").on_hover_text("All vectors as JSON, one array per text").clicked() {
                        ui.ctx().copy_text(vectors_json(out));
                        app.actions.push(Action::Toast("Vectors copied as JSON".to_string()));
                    }
                }
            });
        });
    });
    if !app.embed_error.is_empty() {
        ui.add_space(8.0);
        banner(ui, &app.embed_error, theme::BAD, theme::BAD_BG);
    }
    if let Some(out) = app.embed_out.clone() {
        ui.add_space(14.0);
        results(ui, &out);
    }
    ui.add_space(24.0);
}

/// The vectors as JSON (an array of arrays, six digits).
fn vectors_json(out: &EmbedOut) -> String {
    let rows: Vec<String> = out.vectors.iter().map(|v| format!("[{}]", v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(","))).collect();
    format!("[{}]", rows.join(","))
}

/// Mix two colors: `t` = 0 is `a`, 1 is `b`.
fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgb(f(a.r(), b.r()), f(a.g(), b.g()), f(a.b(), b.b()))
}

fn results(ui: &mut Ui, out: &EmbedOut) {
    let n = out.texts.len();
    let dim = out.vectors.first().map_or(0, |v| v.len());
    let tokens_per_s = if out.millis > 0.0 { out.tokens as f32 / (out.millis / 1e3) } else { 0.0 };
    ui.horizontal_wrapped(|ui| {
        neutral_badge(ui, &format!("{n} text{}", if n == 1 { "" } else { "s" }));
        neutral_badge(ui, &format!("{} tokens", human_count(out.tokens)));
        neutral_badge(ui, &format!("{dim} dimensions"));
        neutral_badge(ui, &format!("{:.0} ms", out.millis)).on_hover_text(format!("{} tokens per second, including the request", human_count(tokens_per_s as u64)));
    });
    if n < 2 {
        ui.add_space(6.0);
        muted(ui, "Add a second line to see how close two texts are.");
        return;
    }
    section(ui, "Closest to the first line");
    let sims: Vec<f32> = (0..n).map(|i| cosine(&out.vectors[0], &out.vectors[i])).collect();
    let mut order: Vec<usize> = (1..n).collect();
    order.sort_by(|&a, &b| sims[b].partial_cmp(&sims[a]).unwrap_or(std::cmp::Ordering::Equal));
    card(ui, |ui| {
        ui.label(RichText::new(truncate(&out.texts[0], 160)).color(theme::STRONG));
        ui.add_space(6.0);
        for &i in &order {
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(52.0, 20.0), Sense::hover());
                ui.painter().text(rect.left_center(), Align2::LEFT_CENTER, format!("{:.3}", sims[i]), FontId::monospace(14.0), theme::ACCENT);
                let (bar_rect, _) = ui.allocate_exact_size(egui::vec2(120.0, 20.0), Sense::hover());
                let mid = bar_rect.shrink2(egui::vec2(0.0, 6.0));
                ui.painter().rect_filled(mid, CornerRadius::same(3), theme::BG);
                let mut fill = mid;
                fill.set_width(mid.width() * sims[i].clamp(0.0, 1.0));
                ui.painter().rect_filled(fill, CornerRadius::same(3), mix(theme::ACCENT_DIM, theme::ACCENT, sims[i]));
                ui.add(egui::Label::new(RichText::new(truncate(&out.texts[i], 120)).color(theme::TEXT)).truncate());
            });
        }
    });
    if n <= 10 {
        section(ui, "Every pair");
        muted(ui, "The cosine of the two vectors: 1 is the same direction, 0 is unrelated. The numbers are the lines, top to bottom.");
        ui.add_space(4.0);
        let cell = egui::vec2(58.0, 30.0);
        ui.horizontal(|ui| {
            ui.add_space(0.0);
            ui.allocate_exact_size(egui::vec2(30.0, 22.0), Sense::hover());
            for j in 0..n {
                let (r, _) = ui.allocate_exact_size(egui::vec2(cell.x, 22.0), Sense::hover());
                ui.painter().text(r.center(), Align2::CENTER_CENTER, format!("{}", j + 1), FontId::proportional(13.0), theme::MUTED);
            }
        });
        for i in 0..n {
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(egui::vec2(30.0, cell.y), Sense::hover());
                ui.painter().text(r.center(), Align2::CENTER_CENTER, format!("{}", i + 1), FontId::proportional(13.0), theme::MUTED);
                for j in 0..n {
                    let s = cosine(&out.vectors[i], &out.vectors[j]);
                    let (r, resp) = ui.allocate_exact_size(cell, Sense::hover());
                    let r = r.shrink(1.0);
                    let t = ((s - 0.3) / 0.7).clamp(0.0, 1.0);
                    ui.painter().rect_filled(r, CornerRadius::same(4), mix(theme::CARD, theme::ACCENT, t * 0.85));
                    let fg = if t > 0.55 { theme::ON_ACCENT } else { theme::TEXT };
                    ui.painter().text(r.center(), Align2::CENTER_CENTER, format!("{s:.2}"), FontId::proportional(13.5), fg);
                    resp.on_hover_text(format!("{} \u{2194} {}: {s:.4}", truncate(&out.texts[i], 50), truncate(&out.texts[j], 50)));
                }
            });
        }
    }
}
