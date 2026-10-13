//! "My models": the GGUF files on this computer, and the inspector that takes one apart.

use eframe::egui::{
    self, pos2, vec2, Color32, CornerRadius, Rect, RichText, ScrollArea, Sense, TextEdit, Ui,
};

use super::widgets::*;
use crate::app::{Action, Screen, StudioApp};
use crate::hardware::{fit, Fit};
use crate::library::{Inspect, ModelEntry};
use crate::settings::EngineKind;
use crate::theme;
use crate::util::{human_bytes, human_count};

pub const PALETTE: [Color32; 8] = [
    Color32::from_rgb(255, 143, 64),
    Color32::from_rgb(110, 190, 245),
    Color32::from_rgb(96, 208, 140),
    Color32::from_rgb(190, 140, 255),
    Color32::from_rgb(240, 200, 80),
    Color32::from_rgb(110, 210, 210),
    Color32::from_rgb(236, 110, 160),
    Color32::from_rgb(150, 156, 170),
];

pub fn show(app: &mut StudioApp, ui: &mut Ui) {
    if let Some((path, result)) = app.inspect.clone() {
        inspector(app, ui, &path, &result);
        return;
    }
    egui::Frame::new().inner_margin(egui::Margin::symmetric(22, 16)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            heading(ui, "My models");
            right_aligned(ui, |ui| {
                if ui.add_enabled(!app.scanning, egui::Button::new("Rescan")).clicked() {
                    app.actions.push(Action::Rescan);
                }
                if app.scanning {
                    ui.spinner();
                }
            });
        });
        muted(ui, "GGUF files from the folders below, LM Studio's folder and Ollama's store. Nothing is copied: models are used where they are.");
        ui.add_space(8.0);
        folders(app, ui);
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.add(TextEdit::singleline(&mut app.lib_filter).hint_text("Filter by name, architecture or quantization").desired_width(340.0));
            let n = app.library.iter().filter(|e| !e.projector).count();
            muted(ui, &format!("{n} model(s)"));
        });
        ui.add_space(6.0);
        let filter = app.lib_filter.to_lowercase();
        let entries: Vec<ModelEntry> = app.library.iter().filter(|e| filter.is_empty() || format!("{} {} {} {}", e.name, e.arch, e.quant, e.source).to_lowercase().contains(&filter)).cloned().collect();
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            if entries.is_empty() {
                card(ui, |ui| {
                    ui.label(RichText::new("No models here yet.").color(theme::TEXT));
                    muted(ui, "Add a folder that holds .gguf files, or download one from Discover.");
                    if primary_button(ui, "Open Discover").clicked() {
                        app.screen = Screen::Discover;
                    }
                });
            }
            for e in &entries {
                model_card(app, ui, e);
                ui.add_space(6.0);
            }
        });
    });
}

fn folders(app: &mut StudioApp, ui: &mut Ui) {
    egui::CollapsingHeader::new(
        RichText::new(format!("Model folders ({})", app.settings.model_dirs.len()))
            .color(theme::TEXT),
    )
    .default_open(false)
    .show(ui, |ui| {
        let dirs = app.settings.model_dirs.clone();
        for (i, d) in dirs.iter().enumerate() {
            ui.horizontal(|ui| {
                if ui.small_button("Remove").clicked() {
                    app.actions.push(Action::RemoveDir(i));
                }
                if ui.small_button("Open").clicked() {
                    crate::util::reveal(std::path::Path::new(d));
                }
                ui.label(RichText::new(d).color(theme::MUTED));
            });
        }
        ui.horizontal(|ui| {
            ui.add(
                TextEdit::singleline(&mut app.new_dir)
                    .hint_text("Paste a folder path")
                    .desired_width(420.0),
            );
            if ui.button("Add folder").clicked() {
                let d = std::mem::take(&mut app.new_dir);
                app.actions.push(Action::AddDir(d));
            }
        });
    });
}

fn model_card(app: &mut StudioApp, ui: &mut Ui, e: &ModelEntry) {
    card(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(&e.name).size(16.0).color(theme::STRONG).family(theme::bold_family()));
            right_aligned(ui, |ui| {
                if ui.button("Folder").on_hover_text(e.path.display().to_string()).clicked() {
                    crate::util::reveal(&e.path);
                }
                if ui.button("Inspect").on_hover_text("Layer map, formats, metadata and tensors").clicked() {
                    app.actions.push(Action::Inspect(e.path.clone()));
                }
                if crate::util::is_gguf_file(&e.path) {
                    if app.confirm_delete.as_ref() == Some(&e.path) {
                        if danger_button(ui, &format!("Delete {} for good", human_bytes(e.size))).clicked() {
                            app.actions.push(Action::DeleteModel(e.path.clone()));
                            app.confirm_delete = None;
                        }
                        if ui.button("Keep").clicked() {
                            app.confirm_delete = None;
                        }
                    } else if ui.button("Delete").on_hover_text("Remove this file from the disk (asks once more)").clicked() {
                        app.confirm_delete = Some(e.path.clone());
                    }
                }
                let issue = if app.settings.engine == EngineKind::Forge { e.forge_issue() } else if e.projector { Some("a vision projector goes with a model, it is not a model".to_string()) } else { e.error.clone() };
                let loadable = issue.is_none();
                let r = primary_enabled(ui, loadable, "Load");
                if let Some(why) = issue {
                    r.on_disabled_hover_text(format!("Cannot load this file: {why}. The llama.cpp engine in Settings may run it."));
                } else if r.clicked() {
                    app.actions.push(Action::LoadModel(e.path.clone()));
                    app.screen = Screen::Chat;
                }
            });
        });
        ui.horizontal_wrapped(|ui| {
            if let Some(err) = &e.error {
                badge(ui, &format!("unreadable: {}", crate::util::truncate(err, 60)), theme::BAD, theme::BAD_BG);
            } else {
                neutral_badge(ui, &e.arch);
                if e.params > 0 {
                    neutral_badge(ui, &format!("{} parameters", human_count(e.params)));
                }
                badge(ui, &e.quant, theme::ACCENT, theme::ACCENT_DIM);
                neutral_badge(ui, &human_bytes(e.size));
                if e.ctx_train > 0 {
                    neutral_badge(ui, &format!("{} context", human_count(e.ctx_train)));
                }
            }
            neutral_badge(ui, &e.source);
            if e.projector {
                badge(ui, "vision projector", theme::INFO, theme::CARD);
            } else if e.error.is_none() {
                match fit(&app.hw, e.size) {
                    Fit::Gpu => {
                        badge(ui, "fits in video memory", theme::GOOD, theme::CARD);
                    }
                    Fit::Split => {
                        badge(ui, "runs split between GPU and CPU", theme::WARN, theme::CARD);
                    }
                    Fit::TooBig => {
                        badge(ui, "too large for this computer", theme::BAD, theme::CARD);
                    }
                    Fit::Unknown => {}
                }
                if e.arch == "qwen35" && !e.projector && app.settings.engine == EngineKind::Forge {
                    if crate::mtp::find_head(&e.path, &{ let mut d = app.settings.dirs(); d.push(std::path::PathBuf::from(&app.settings.download_dir)); d }).is_some() {
                        badge(ui, "prediction head: faster answers", theme::GOOD, theme::CARD).on_hover_text("Speculative decoding is used when this model loads (Settings > When a model loads).");
                    } else if let Some(repo) = crate::mtp::repo_of(&e.path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()) {
                        if ui.small_button("Get the prediction head").on_hover_text(format!("About 0.2 to 0.9 GB from huggingface.co/{repo}. With it the model answers about 1.5 to 2 times faster (speculative decoding).")).clicked() {
                            app.actions.push(Action::GetHead(e.path.clone()));
                        }
                    }
                }
                if app.settings.engine == EngineKind::Forge {
                    if let Some(why) = e.forge_issue() {
                        badge(ui, "not supported by the ForgeInfer engine", theme::WARN, theme::CARD).on_hover_text(why);
                    }
                }
            }
        });
        ui.add(
            egui::Label::new(
                RichText::new(e.path.display().to_string())
                    .small()
                    .color(theme::MUTED),
            )
            .truncate(),
        );
    });
}

fn kind_color(kind: &str) -> Color32 {
    match kind {
        "Attention" => PALETTE[0],
        "DeltaNet (recurrent)" => PALETTE[1],
        "Attention + MoE" => PALETTE[3],
        _ => PALETTE[7],
    }
}

fn stacked_bar(ui: &mut Ui, parts: &[(String, u64)]) {
    let total: u64 = parts.iter().map(|p| p.1).sum::<u64>().max(1);
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 22.0), Sense::hover());
    let mut x = rect.left();
    let mut hover: Option<String> = None;
    for (i, (name, bytes)) in parts.iter().enumerate() {
        let w = rect.width() * (*bytes as f32 / total as f32);
        let r = Rect::from_min_max(
            pos2(x, rect.top()),
            pos2((x + w).min(rect.right()), rect.bottom()),
        );
        ui.painter()
            .rect_filled(r, CornerRadius::ZERO, PALETTE[i % PALETTE.len()]);
        if let Some(p) = resp.hover_pos() {
            if r.contains(p) {
                hover = Some(format!(
                    "{name}: {} ({:.1} %)",
                    human_bytes(*bytes),
                    100.0 * *bytes as f64 / total as f64
                ));
            }
        }
        x += w;
    }
    if let Some(t) = hover {
        resp.on_hover_text(t);
    }
    ui.add_space(4.0);
    ui.horizontal_wrapped(|ui| {
        for (i, (name, bytes)) in parts.iter().enumerate() {
            let (r, _) = ui.allocate_exact_size(vec2(10.0, 10.0), Sense::hover());
            ui.painter()
                .rect_filled(r, CornerRadius::same(2), PALETTE[i % PALETTE.len()]);
            ui.label(
                RichText::new(format!(
                    "{name}  {}  ({:.1} %)",
                    human_bytes(*bytes),
                    100.0 * *bytes as f64 / total as f64
                ))
                .small()
                .color(theme::TEXT),
            );
            ui.add_space(8.0);
        }
    });
}

fn layer_strip(ui: &mut Ui, i: &Inspect) {
    let layers = &i.layers;
    if layers.is_empty() {
        muted(ui, "This model has no numbered layers.");
        return;
    }
    let h = 96.0;
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), h), Sense::hover());
    ui.painter()
        .rect_filled(rect, CornerRadius::same(6), theme::BG);
    let n = layers.len();
    let gap = if n > 90 { 1.0 } else { 2.0 };
    let bw = ((rect.width() - gap * (n as f32 - 1.0)) / n as f32).max(1.0);
    let max = layers.iter().map(|l| l.bytes).max().unwrap_or(1) as f32;
    for (k, l) in layers.iter().enumerate() {
        let x = rect.left() + k as f32 * (bw + gap);
        let hh = (l.bytes as f32 / max) * (h - 6.0);
        let r = Rect::from_min_max(
            pos2(x, rect.bottom() - hh - 2.0),
            pos2(x + bw, rect.bottom() - 2.0),
        );
        ui.painter()
            .rect_filled(r, CornerRadius::same(1), kind_color(&l.kind));
    }
    if let Some(p) = resp.hover_pos() {
        let k = ((p.x - rect.left()) / (bw + gap)) as usize;
        if let Some(l) = layers.get(k) {
            resp.on_hover_text(format!(
                "Layer {}: {}\n{} in {}",
                l.index,
                l.kind,
                human_bytes(l.bytes),
                l.dtype
            ));
        }
    }
    let mut kinds: Vec<&str> = layers.iter().map(|l| l.kind.as_str()).collect();
    kinds.sort();
    kinds.dedup();
    ui.horizontal_wrapped(|ui| {
        for k in kinds {
            let count = layers.iter().filter(|l| l.kind == k).count();
            let (r, _) = ui.allocate_exact_size(vec2(10.0, 10.0), Sense::hover());
            ui.painter()
                .rect_filled(r, CornerRadius::same(2), kind_color(k));
            ui.label(
                RichText::new(format!("{k}: {count} layer(s)"))
                    .small()
                    .color(theme::TEXT),
            );
            ui.add_space(8.0);
        }
    });
}

fn inspector(
    app: &mut StudioApp,
    ui: &mut Ui,
    path: &std::path::Path,
    result: &Result<Inspect, String>,
) {
    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(22, 16))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                if ui.button("\u{2190}  Back to models").clicked() {
                    app.inspect = None;
                }
                if app.inspect_loading {
                    ui.spinner();
                }
            });
            ui.add_space(6.0);
            let i = match result {
                Ok(i) => i,
                Err(e) => {
                    heading(
                        ui,
                        &path
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default(),
                    );
                    muted(ui, e);
                    return;
                }
            };
            ui.horizontal(|ui| {
                heading(ui, &i.entry.name);
                right_aligned(ui, |ui| {
                    let loadable = !i.entry.projector
                        && i.entry.error.is_none()
                        && (app.settings.engine != EngineKind::Forge || i.entry.forge_supported());
                    if primary_enabled(ui, loadable, "Load this model").clicked() {
                        app.actions.push(Action::LoadModel(i.entry.path.clone()));
                        app.screen = Screen::Chat;
                    }
                });
            });
            ui.add(
                egui::Label::new(
                    RichText::new(i.entry.path.display().to_string())
                        .small()
                        .color(theme::MUTED),
                )
                .truncate(),
            );
            ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    section(ui, "Summary");
                    card(ui, |ui| {
                        kv(ui, "Architecture", &i.entry.arch);
                        kv(ui, "Parameters", human_count(i.entry.params));
                        kv(ui, "Layers", i.entry.layers.to_string());
                        kv(ui, "Trained context", human_count(i.entry.ctx_train));
                        kv(ui, "Vocabulary", i.vocab.to_string());
                        kv(ui, "Quantization", &i.entry.quant);
                        kv(ui, "File size", human_bytes(i.entry.size));
                        kv(ui, "Tensors", i.tensors.len().to_string());
                        let bits = if i.entry.params > 0 {
                            i.entry.size as f64 * 8.0 / i.entry.params as f64
                        } else {
                            0.0
                        };
                        kv(ui, "Bits per weight", format!("{bits:.2}"));
                    });
                    section(ui, "Layer map");
                    muted(
                        ui,
                        "One bar per layer, as tall as its weights. Hover for details.",
                    );
                    layer_strip(ui, i);
                    section(ui, "What the bytes are");
                    let roles: Vec<(String, u64)> = i.roles.clone();
                    stacked_bar(ui, &roles);
                    section(ui, "Quantization mix");
                    let formats: Vec<(String, u64)> = i
                        .formats
                        .iter()
                        .map(|f| (format!("{} ({} tensors)", f.0, f.2), f.1))
                        .collect();
                    stacked_bar(ui, &formats);
                    section(ui, "Metadata");
                    egui::CollapsingHeader::new(format!("{} entries", i.meta.len()))
                        .default_open(false)
                        .show(ui, |ui| {
                            egui::Grid::new("meta_grid")
                                .striped(true)
                                .spacing([14.0, 4.0])
                                .show(ui, |ui| {
                                    for (k, v) in &i.meta {
                                        ui.label(RichText::new(k).small().color(theme::MUTED));
                                        ui.add(
                                            egui::Label::new(
                                                RichText::new(v)
                                                    .small()
                                                    .monospace()
                                                    .color(theme::TEXT),
                                            )
                                            .wrap(),
                                        );
                                        ui.end_row();
                                    }
                                });
                        });
                    section(ui, "Tensors");
                    egui::CollapsingHeader::new(format!("{} tensors", i.tensors.len()))
                        .default_open(false)
                        .show(ui, |ui| {
                            let row_h = 18.0;
                            ScrollArea::vertical()
                                .id_salt("tensor_rows")
                                .max_height(380.0)
                                .auto_shrink([false, true])
                                .show_rows(ui, row_h, i.tensors.len(), |ui, range| {
                                    for t in &i.tensors[range] {
                                        ui.horizontal(|ui| {
                                            ui.add_sized(
                                                [360.0, row_h],
                                                egui::Label::new(
                                                    RichText::new(&t.name)
                                                        .small()
                                                        .monospace()
                                                        .color(theme::TEXT),
                                                )
                                                .truncate(),
                                            );
                                            ui.add_sized(
                                                [70.0, row_h],
                                                egui::Label::new(
                                                    RichText::new(&t.dtype)
                                                        .small()
                                                        .color(theme::ACCENT),
                                                ),
                                            );
                                            ui.add_sized(
                                                [150.0, row_h],
                                                egui::Label::new(
                                                    RichText::new(format!("{:?}", t.shape))
                                                        .small()
                                                        .color(theme::MUTED),
                                                ),
                                            );
                                            ui.label(
                                                RichText::new(human_bytes(t.bytes))
                                                    .small()
                                                    .color(theme::MUTED),
                                            );
                                        });
                                    }
                                });
                        });
                });
        });
}
