//! Discover: search Hugging Face for GGUF models and download them into the library.

use eframe::egui::{self, Key, RichText, ScrollArea, TextEdit, Ui};
use std::sync::atomic::Ordering;

use super::widgets::*;
use crate::app::{Action, DlState, StudioApp};
use crate::hardware::{fit, Fit};
use crate::hf;
use crate::theme;
use crate::util::{human_bytes, human_count};

const SUGGESTIONS: [&str; 8] = [
    "Qwen3.5",
    "Qwen3",
    "Llama 3.2",
    "SmolLM2",
    "Gemma",
    "Mistral",
    "Phi",
    "DeepSeek",
];

/// A download to offer before anything has been searched: the exact repository and file (names and sizes checked against Hugging Face), what the model is good for.
struct Pick {
    repo: &'static str,
    file: &'static str,
    size: u64,
    name: &'static str,
    note: &'static str,
}

const PICKS: [Pick; 6] = [
    Pick { repo: "unsloth/Qwen3.5-4B-GGUF", file: "Qwen3.5-4B-Q4_K_M.gguf", size: 2_740_937_888, name: "Qwen3.5 4B", note: "A fast all-round chat model with a long context and an optional prediction head for faster answers. A good first model." },
    Pick { repo: "unsloth/Qwen3.5-9B-GGUF", file: "Qwen3.5-9B-Q4_K_M.gguf", size: 5_680_522_464, name: "Qwen3.5 9B", note: "The best quality that still fits an 8 GB card (with a medium context)." },
    Pick { repo: "unsloth/Qwen3.5-0.8B-GGUF", file: "Qwen3.5-0.8B-Q4_K_M.gguf", size: 532_517_120, name: "Qwen3.5 0.8B", note: "Tiny and instant: for a first try and for computers without a big graphics card." },
    Pick { repo: "unsloth/Qwen3-4B-Instruct-2507-GGUF", file: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", size: 2_497_281_120, name: "Qwen3 4B Instruct 2507", note: "Answers at once without a thinking phase." },
    Pick { repo: "nomic-ai/nomic-embed-text-v1.5-GGUF", file: "nomic-embed-text-v1.5.Q4_K_M.gguf", size: 84_106_624, name: "nomic-embed-text v1.5", note: "An embedding model: lets a chat search long PDF, Word and text files instead of sending them whole, and serves /v1/embeddings." },
    Pick { repo: "CompendiumLabs/bge-small-en-v1.5-gguf", file: "bge-small-en-v1.5-q8_0.gguf", size: 36_806_944, name: "bge-small-en-v1.5", note: "A small English embedding model, the same job as nomic-embed-text in less than half the space: its context is 512 tokens, so a passage longer than about 350 words is cut." },
];

/// The suggestions, shown while nothing has been searched yet.
fn suggestions(app: &mut StudioApp, ui: &mut Ui) {
    section(ui, "Good first downloads");
    for p in PICKS.iter() {
        let have = app.library.iter().any(|e| e.path.file_name().is_some_and(|n| n == p.file));
        let running = app.downloads.iter().any(|d| d.state == DlState::Running && d.file.path == p.file);
        card(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(p.name).color(theme::STRONG).family(theme::bold_family()));
                neutral_badge(ui, &human_bytes(p.size));
                match fit(&app.hw, p.size) {
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
            });
            ui.add(egui::Label::new(RichText::new(p.note).small().color(theme::MUTED)).wrap());
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                if have {
                    ui.label(RichText::new("already in My models").small().color(theme::GOOD));
                } else if running {
                    ui.spinner();
                    ui.label(RichText::new("downloading ...").small().color(theme::MUTED));
                } else if ui.button("Download").on_hover_text(format!("{} from huggingface.co/{}", p.file, p.repo)).clicked() {
                    app.actions.push(Action::Download { repo: p.repo.to_string(), file: hf::RepoFile { path: p.file.to_string(), size: p.size } });
                }
            });
        });
        ui.add_space(4.0);
    }
}

pub fn show(app: &mut StudioApp, ui: &mut Ui) {
    egui::Frame::new().inner_margin(egui::Margin::symmetric(22, 16)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        heading(ui, "Discover models");
        muted(ui, "Search Hugging Face for GGUF files. Downloads go to the folder set in Settings and appear in My models.");
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let r = ui.add(TextEdit::singleline(&mut app.hf_query).hint_text("Search, for example qwen3.5 or llama 3.2 1b").desired_width(420.0));
            let enter = r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
            if (primary_enabled(ui, !app.hf_query.trim().is_empty() && !app.hf_busy, "Search").clicked() || enter) && !app.hf_query.trim().is_empty() {
                app.actions.push(Action::HfSearch);
            }
            if app.hf_busy {
                ui.spinner();
            }
        });
        ui.horizontal_wrapped(|ui| {
            muted(ui, "Try:");
            for s in SUGGESTIONS {
                if ui.small_button(s).clicked() {
                    app.hf_query = s.to_string();
                    app.actions.push(Action::HfSearch);
                }
            }
        });
        if !app.hf_error.is_empty() {
            ui.add_space(4.0);
            banner(ui, &app.hf_error.clone(), theme::BAD, theme::BAD_BG);
        }
        ui.add_space(8.0);
        let avail = ui.available_height();
        let downloads_h = if app.downloads.is_empty() { 0.0 } else { (app.downloads.len().min(3) as f32) * 66.0 + 34.0 };
        let top_h = (avail - downloads_h - 8.0).max(180.0);
        ui.horizontal_top(|ui| {
            let total = ui.available_width();
            let down = egui::Layout::top_down(egui::Align::Min);
            ui.allocate_ui_with_layout(egui::vec2(total * 0.40 - 6.0, top_h), down, |ui| results(app, ui, top_h));
            ui.allocate_ui_with_layout(egui::vec2(total * 0.60 - 6.0, top_h), down, |ui| files(app, ui, top_h));
        });
        if !app.downloads.is_empty() {
            ui.add_space(6.0);
            downloads(app, ui);
        }
    });
}

fn results(app: &mut StudioApp, ui: &mut Ui, height: f32) {
    ui.set_min_height(height);
    ScrollArea::vertical()
        .id_salt("hf_results")
        .max_height(height)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if app.hf_results.is_empty() && !app.hf_busy && app.hf_error.is_empty() {
                muted(
                    ui,
                    "Search for a model family to see the GGUF repositories that offer it.",
                );
                suggestions(app, ui);
            }
            for r in app.hf_results.clone() {
                let selected = app.hf_repo.as_deref() == Some(r.id.as_str());
                let resp = egui::Frame::new()
                    .fill(if selected {
                        theme::CARD_HOVER
                    } else {
                        theme::CARD
                    })
                    .corner_radius(egui::CornerRadius::same(8))
                    .stroke(egui::Stroke::new(
                        1.0,
                        if selected {
                            theme::ACCENT.gamma_multiply(0.6)
                        } else {
                            theme::BORDER
                        },
                    ))
                    .inner_margin(egui::Margin::symmetric(10, 7))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.add(
                            egui::Label::new(
                                RichText::new(&r.id)
                                    .color(theme::STRONG)
                                    .family(theme::bold_family()),
                            )
                            .truncate(),
                        );
                        ui.label(
                            RichText::new(format!(
                                "{} downloads  \u{00B7}  {} likes",
                                human_count(r.downloads),
                                human_count(r.likes)
                            ))
                            .small()
                            .color(theme::MUTED),
                        );
                    })
                    .response
                    .interact(egui::Sense::click())
                    .on_hover_cursor(egui::CursorIcon::PointingHand);
                if resp.clicked() {
                    app.actions.push(Action::HfOpen(r.id.clone()));
                }
                ui.add_space(4.0);
            }
        });
}

fn files(app: &mut StudioApp, ui: &mut Ui, height: f32) {
    ui.set_min_height(height);
    let Some(repo) = app.hf_repo.clone() else {
        muted(ui, "Pick a repository to see its files.");
        return;
    };
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(&repo)
                .size(16.0)
                .color(theme::STRONG)
                .family(theme::bold_family()),
        );
        if app.hf_files_busy {
            ui.spinner();
        }
        if ui.small_button("Open on huggingface.co").clicked() {
            ui.ctx().open_url(egui::OpenUrl::new_tab(format!(
                "https://huggingface.co/{repo}"
            )));
        }
    });
    ScrollArea::vertical()
        .id_salt("hf_files")
        .max_height(height - 30.0)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if app.hf_files.is_empty() && !app.hf_files_busy {
                muted(ui, "No GGUF files in this repository.");
            }
            for f in app.hf_files.clone() {
                let dest =
                    hf::destination(std::path::Path::new(&app.settings.download_dir), &repo, &f);
                let have = dest.is_file();
                let running = app
                    .downloads
                    .iter()
                    .find(|d| {
                        d.repo == repo && d.file.path == f.path && d.state == DlState::Running
                    })
                    .map(|d| d.id);
                card(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::Label::new(RichText::new(f.file_name()).color(theme::TEXT))
                                .truncate(),
                        );
                        right_aligned(ui, |ui| {
                            if have {
                                badge(ui, "downloaded", theme::GOOD, theme::CARD);
                            } else if let Some(id) = running {
                                if ui.button("Cancel").clicked() {
                                    app.actions.push(Action::CancelDownload(id));
                                }
                            } else if primary_button(ui, "Download").clicked() {
                                app.actions.push(Action::Download {
                                    repo: repo.clone(),
                                    file: f.clone(),
                                });
                            }
                        });
                    });
                    ui.horizontal_wrapped(|ui| {
                        let q = f.quant();
                        if !q.is_empty() {
                            badge(ui, &q, theme::ACCENT, theme::ACCENT_DIM);
                            let note = hf::quant_note(&q);
                            if !note.is_empty() {
                                muted(ui, note);
                            }
                        }
                        neutral_badge(ui, &human_bytes(f.size));
                        if f.is_projector() {
                            badge(ui, "vision projector", theme::INFO, theme::CARD);
                        } else if let Some((part, of)) = f.shard() {
                            badge(
                                ui,
                                &format!("part {part} of {of}: download every part"),
                                theme::WARN,
                                theme::CARD,
                            );
                        } else {
                            match fit(&app.hw, f.size) {
                                Fit::Gpu => {
                                    badge(ui, "fits in video memory", theme::GOOD, theme::CARD);
                                }
                                Fit::Split => {
                                    badge(ui, "split GPU + CPU", theme::WARN, theme::CARD);
                                }
                                Fit::TooBig => {
                                    badge(ui, "too large", theme::BAD, theme::CARD);
                                }
                                Fit::Unknown => {}
                            }
                        }
                    });
                });
                ui.add_space(4.0);
            }
        });
}

fn downloads(app: &mut StudioApp, ui: &mut Ui) {
    section(ui, "Downloads");
    ScrollArea::vertical()
        .id_salt("downloads")
        .max_height(210.0)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            for d in app.downloads.iter().rev() {
                card(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::Label::new(RichText::new(d.file.file_name()).color(theme::TEXT))
                                .truncate(),
                        );
                        right_aligned(ui, |ui| match &d.state {
                            DlState::Running => {
                                if ui.button("Cancel").clicked() {
                                    app.actions.push(Action::CancelDownload(d.id));
                                }
                            }
                            DlState::Done(p) => {
                                if ui.button("Show in folder").clicked() {
                                    crate::util::reveal(p);
                                }
                            }
                            DlState::Failed(_) => {}
                        });
                    });
                    let done = d.progress.done.load(Ordering::Relaxed);
                    let total = d
                        .progress
                        .total
                        .load(Ordering::Relaxed)
                        .max(d.file.size)
                        .max(1);
                    match &d.state {
                        DlState::Running => {
                            bar(ui, done as f32 / total as f32, theme::ACCENT, 8.0);
                            let eta = if d.speed > 1.0 {
                                format!(
                                    "  \u{00B7}  {:.0} s left",
                                    (total - done.min(total)) as f64 / d.speed
                                )
                            } else {
                                String::new()
                            };
                            muted(
                                ui,
                                &format!(
                                    "{} of {}  \u{00B7}  {}/s{eta}",
                                    human_bytes(done),
                                    human_bytes(total),
                                    human_bytes(d.speed as u64)
                                ),
                            );
                        }
                        DlState::Done(_) => muted(ui, &format!("Done: {}", human_bytes(done))),
                        DlState::Failed(e) => {
                            ui.label(RichText::new(e).small().color(theme::BAD));
                        }
                    }
                });
                ui.add_space(4.0);
            }
        });
}
