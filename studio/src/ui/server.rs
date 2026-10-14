//! The local server, the hardware report and the engine log.

use eframe::egui::{self, RichText, ScrollArea, Ui};

use super::widgets::*;
use crate::app::{Action, StudioApp};
use crate::engine::EngineState;
use crate::hardware::{fit, Fit};
use crate::settings::{find_forge, find_llama_server, EngineKind};
use crate::theme;
use crate::util::{human_bytes, human_gib};

fn code(ui: &mut Ui, id: &str, text: &str) {
    egui::Frame::new()
        .fill(theme::CODE_BG)
        .stroke(egui::Stroke::new(1.0, theme::BORDER))
        .corner_radius(egui::CornerRadius::same(8))
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                muted(ui, id);
                right_aligned(ui, |ui| {
                    if ui.small_button("Copy").clicked() {
                        ui.ctx().copy_text(text.to_string());
                    }
                });
            });
            ui.add(
                egui::Label::new(
                    RichText::new(text)
                        .monospace()
                        .size(12.5)
                        .color(theme::TEXT),
                )
                .wrap(),
            );
        });
}

pub fn show_server(app: &mut StudioApp, ui: &mut Ui) {
    egui::Frame::new().inner_margin(egui::Margin::symmetric(22, 16)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        heading(ui, "Local server");
        muted(ui, "Every model you load is also an OpenAI-compatible server: point any tool at it, the way you would at LM Studio or Ollama.");
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            section(ui, "Status");
            let state = app.engine.state();
            let (base, model) = match (&state, app.settings.engine) {
                (EngineState::Ready { model, port, .. }, _) => (Some(format!("http://127.0.0.1:{port}/v1")), model.clone()),
                (_, EngineKind::External) => (Some(format!("{}/v1", app.settings.external_base())), app.external_pick.clone()),
                _ => (None, "your-model".to_string()),
            };
            card(ui, |ui| match &base {
                Some(b) => {
                    ui.horizontal(|ui| {
                        dot(ui, theme::GOOD);
                        ui.label(RichText::new(if app.settings.engine == EngineKind::External { "Using an external server" } else { "Running" }).color(theme::TEXT));
                    });
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(b).monospace().color(theme::ACCENT));
                        if ui.small_button("Copy").clicked() {
                            ui.ctx().copy_text(b.clone());
                        }
                    });
                    kv(ui, "Model name", &model);
                }
                None => {
                    ui.horizontal(|ui| {
                        dot(ui, theme::MUTED);
                        ui.label(RichText::new("Not running: load a model and its server starts with it.").color(theme::TEXT));
                    });
                }
            });
            section(ui, "Access");
            card(ui, |ui| {
                ui.checkbox(&mut app.settings.serve_on_lan, "Let other computers on my network connect");
                muted(ui, "Off: only this computer can connect. On: the server listens on every network interface (reload the model for the change to apply). Set an API key before you turn this on.");
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("API key");
                    ui.add(egui::TextEdit::singleline(&mut app.settings.api_key).password(true).desired_width(280.0).hint_text("empty = no key"));
                });
                ui.horizontal(|ui| {
                    ui.label("Port");
                    ui.add(egui::DragValue::new(&mut app.settings.port).range(1024..=65535));
                    muted(ui, "1234 is the port LM Studio uses; the next free one is taken when it is busy.");
                });
            });
            if let Some(b) = base {
                let key = if app.settings.api_key.is_empty() { "not-needed".to_string() } else { app.settings.api_key.clone() };
                section(ui, "Use it from your tools");
                if app.active_is_embedding() {
                    code(
                        ui,
                        "curl",
                        &format!(
                            "curl {b}/embeddings \\\n  -H \"Content-Type: application/json\" \\\n  -H \"Authorization: Bearer {key}\" \\\n  -d '{{\"model\": \"{model}\", \"input\": [\"search_query: What is TCP?\", \"search_document: TCP is a transport protocol.\"]}}'"
                        ),
                    );
                    ui.add_space(6.0);
                    code(
                        ui,
                        "Python (openai package)",
                        &format!("from openai import OpenAI\n\nclient = OpenAI(base_url=\"{b}\", api_key=\"{key}\")\nreply = client.embeddings.create(\n    model=\"{model}\",\n    input=[\"search_query: What is TCP?\", \"search_document: TCP is a transport protocol.\"],\n)\nprint(len(reply.data[0].embedding), reply.data[0].embedding[:5])"),
                    );
                    ui.add_space(6.0);
                    muted(ui, "Start the texts you search through with \"search_document: \" and the questions with \"search_query: \" (nomic-embed-text was trained that way). `dimensions` keeps the first N numbers (64 to 768).");
                } else {
                    code(
                        ui,
                        "curl",
                        &format!(
                            "curl {b}/chat/completions \\\n  -H \"Content-Type: application/json\" \\\n  -H \"Authorization: Bearer {key}\" \\\n  -d '{{\"model\": \"{model}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}], \"stream\": true}}'"
                        ),
                    );
                    ui.add_space(6.0);
                    code(
                        ui,
                        "Python (openai package)",
                        &format!("from openai import OpenAI\n\nclient = OpenAI(base_url=\"{b}\", api_key=\"{key}\")\nreply = client.chat.completions.create(\n    model=\"{model}\",\n    messages=[{{\"role\": \"user\", \"content\": \"Hello!\"}}],\n)\nprint(reply.choices[0].message.content)"),
                    );
                }
                ui.add_space(6.0);
                code(ui, "Settings of another app", &format!("API base URL:  {b}\nAPI key:       {key}\nModel:         {model}"));
            }
        });
    });
}

pub fn show_hardware(app: &mut StudioApp, ui: &mut Ui) {
    egui::Frame::new().inner_margin(egui::Margin::symmetric(22, 16)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            heading(ui, "Hardware");
            right_aligned(ui, |ui| {
                if ui.button("Refresh").clicked() {
                    app.detect_hardware();
                }
            });
        });
        muted(ui, "What this computer offers to a model. Memory decides which models fit.");
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let hw = app.hw.clone();
            section(ui, "Processor and memory");
            card(ui, |ui| {
                kv(ui, "System", &hw.os);
                kv(ui, "Processor", &hw.cpu);
                kv(ui, "Threads", hw.threads.to_string());
                if hw.ram_total > 0 {
                    kv(ui, "Memory", format!("{} total{}", human_gib(hw.ram_total), if hw.ram_free > 0 { format!(", {} free", human_gib(hw.ram_free)) } else { String::new() }));
                    if hw.ram_free > 0 {
                        bar(ui, 1.0 - hw.ram_free as f32 / hw.ram_total as f32, theme::INFO, 8.0);
                    }
                }
            });
            section(ui, "Graphics");
            if hw.gpus.is_empty() {
                card(ui, |ui| {
                    ui.label(RichText::new("No NVIDIA GPU found.").color(theme::TEXT));
                    muted(ui, "The ForgeInfer engine needs an NVIDIA GPU (CUDA). Models still run through the llama.cpp engine or an external server: see Settings.");
                });
            }
            for g in &hw.gpus {
                card(ui, |ui| {
                    ui.label(RichText::new(&g.name).size(16.0).color(theme::STRONG).family(theme::bold_family()));
                    kv(ui, "Video memory", format!("{} total, {} in use", human_gib(g.vram_total), human_gib(g.vram_used)));
                    bar(ui, g.vram_used as f32 / g.vram_total.max(1) as f32, theme::ACCENT, 8.0);
                    if !g.driver.is_empty() {
                        kv(ui, "Driver", &g.driver);
                    }
                });
                ui.add_space(4.0);
            }
            section(ui, "What fits");
            card(ui, |ui| {
                muted(ui, "Typical 4-bit downloads (about 0.6 GB per billion parameters).");
                egui::Grid::new("fits").striped(true).spacing([24.0, 6.0]).show(ui, |ui| {
                    for b in [1.0f64, 3.0, 4.0, 8.0, 9.0, 14.0, 27.0, 32.0, 70.0] {
                        let bytes = (b * 0.6e9) as u64;
                        ui.label(RichText::new(format!("{b:.0}B model")).color(theme::TEXT));
                        ui.label(RichText::new(human_bytes(bytes)).color(theme::MUTED));
                        match fit(&hw, bytes) {
                            Fit::Gpu => ui.label(RichText::new("fits in video memory").color(theme::GOOD)),
                            Fit::Split => ui.label(RichText::new("runs split between GPU and CPU").color(theme::WARN)),
                            Fit::TooBig => ui.label(RichText::new("too large").color(theme::BAD)),
                            Fit::Unknown => ui.label(RichText::new("unknown").color(theme::MUTED)),
                        };
                        ui.end_row();
                    }
                });
            });
            section(ui, "Engine programs");
            card(ui, |ui| {
                match find_forge(&app.settings) {
                    Some(p) => kv(ui, "ForgeInfer engine", p.display().to_string()),
                    None => kv(ui, "ForgeInfer engine", "not found (Settings)"),
                }
                match find_llama_server(&app.settings) {
                    Some(p) => kv(ui, "llama.cpp server", p.display().to_string()),
                    None => kv(ui, "llama.cpp server", "not found (optional)"),
                }
                kv(ui, "In use", app.settings.engine.label());
            });
        });
    });
}

pub fn show_logs(app: &mut StudioApp, ui: &mut Ui) {
    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(22, 16))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                heading(ui, "Logs");
                right_aligned(ui, |ui| {
                    if ui.button("Copy all").clicked() {
                        ui.ctx().copy_text(app.engine.log().join("\n"));
                        app.actions.push(Action::Toast("Log copied".to_string()));
                    }
                });
            });
            muted(
                ui,
                "The command line and everything the engine prints: loading steps, speed, errors.",
            );
            ui.add_space(6.0);
            let log = app.engine.log();
            egui::Frame::new()
                .fill(theme::CODE_BG)
                .stroke(egui::Stroke::new(1.0, theme::BORDER))
                .corner_radius(egui::CornerRadius::same(8))
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    let row_h = 16.0;
                    let h = (ui.available_height() - 20.0).max(120.0);
                    ScrollArea::vertical()
                        .id_salt("log_rows")
                        .max_height(h)
                        .auto_shrink([false, false])
                        .stick_to_bottom(true)
                        .show_rows(ui, row_h, log.len().max(1), |ui, range| {
                            if log.is_empty() {
                                ui.label(
                                    RichText::new("Nothing yet: load a model.")
                                        .small()
                                        .color(theme::MUTED),
                                );
                                return;
                            }
                            for line in &log[range] {
                                let color = if line.to_lowercase().contains("error")
                                    || line.contains("cannot")
                                    || line.contains("failed")
                                {
                                    theme::BAD
                                } else if line.starts_with('$') {
                                    theme::ACCENT
                                } else {
                                    theme::TEXT
                                };
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(line).monospace().size(12.0).color(color),
                                    )
                                    .extend(),
                                );
                            }
                        });
                });
        });
}
