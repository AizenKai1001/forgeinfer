//! Settings: which engine runs the models, how a model is loaded, where files go.

use eframe::egui::{self, RichText, ScrollArea, TextEdit, Ui};

use super::widgets::*;
use crate::app::{Action, StudioApp};
use crate::settings::{data_dir, find_forge, find_llama_server, EngineKind, KvMode};
use crate::theme;

pub fn show(app: &mut StudioApp, ui: &mut Ui) {
    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(22, 16))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            heading(ui, "Settings");
            muted(ui, "Saved as you change them.");
            ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    engine(app, ui);
                    loading(app, ui);
                    files(app, ui);
                    interface(app, ui);
                    about(ui);
                });
        });
}

fn engine(app: &mut StudioApp, ui: &mut Ui) {
    section(ui, "Engine");
    card(ui, |ui| {
        for kind in EngineKind::ALL {
            let what = match kind {
                EngineKind::Forge => "Runs GGUF models on an NVIDIA GPU, with the KV cache compression, the CPU feed-forward offload and speculative decoding of the ForgeInfer engine.",
                EngineKind::LlamaCpp => "Runs GGUF models with llama.cpp's llama-server (any GPU or CPU, macOS included). You provide the program.",
                EngineKind::External => "Chat with a server that is already running: Ollama, LM Studio, vLLM, or the ForgeInfer engine on another computer.",
            };
            ui.radio_value(
                &mut app.settings.engine,
                kind,
                RichText::new(kind.label()).color(theme::TEXT),
            );
            ui.horizontal(|ui| {
                ui.add_space(24.0);
                muted(ui, what);
            });
            ui.add_space(2.0);
        }
        ui.add_space(6.0);
        match app.settings.engine {
            EngineKind::Forge => {
                let found = find_forge(&app.settings).map(|p| p.display().to_string());
                path_row(
                    ui,
                    "forge program",
                    &mut app.settings.forge_binary,
                    found,
                    "empty = next to this app, in a build folder or in the PATH",
                );
            }
            EngineKind::LlamaCpp => {
                let found = find_llama_server(&app.settings).map(|p| p.display().to_string());
                path_row(
                    ui,
                    "llama-server program",
                    &mut app.settings.llama_server,
                    found,
                    "empty = look in the PATH",
                );
            }
            EngineKind::External => {
                ui.horizontal(|ui| {
                    ui.label("Server address");
                    ui.add(
                        TextEdit::singleline(&mut app.settings.external_url)
                            .desired_width(320.0)
                            .hint_text("http://192.168.1.x:8080"),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("API key");
                    ui.add(
                        TextEdit::singleline(&mut app.settings.external_key)
                            .password(true)
                            .desired_width(320.0)
                            .hint_text("empty = none"),
                    );
                });
                ui.horizontal(|ui| {
                    if ui.button("Test connection").clicked() {
                        app.actions.push(Action::RefreshExternal);
                    }
                    muted(ui, &app.external_status.clone());
                });
            }
        }
    });
}

fn path_row(ui: &mut Ui, label: &str, value: &mut String, found: Option<String>, hint: &str) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(
            TextEdit::singleline(value)
                .desired_width(420.0)
                .hint_text(hint),
        );
    });
    match found {
        Some(p) => {
            ui.label(
                RichText::new(format!("In use: {p}"))
                    .small()
                    .color(theme::GOOD),
            );
        }
        None => {
            ui.label(
                RichText::new("Not found: type the full path of the program.")
                    .small()
                    .color(theme::WARN),
            );
        }
    }
}

fn loading(app: &mut StudioApp, ui: &mut Ui) {
    section(ui, "When a model loads");
    card(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Context length");
            ui.add(
                egui::DragValue::new(&mut app.settings.context_len)
                    .range(512..=1_048_576)
                    .speed(64.0),
            );
            for (label, n) in [("4k", 4096), ("8k", 8192), ("16k", 16384), ("32k", 32768)] {
                if ui.small_button(label).clicked() {
                    app.settings.context_len = n;
                }
            }
        });
        muted(
            ui,
            "Tokens of conversation the model can see. A longer context uses more memory.",
        );
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("KV cache");
            egui::ComboBox::from_id_salt("kv_mode")
                .selected_text(app.settings.kv_cache.label())
                .show_ui(ui, |ui| {
                    for k in KvMode::ALL {
                        ui.selectable_value(&mut app.settings.kv_cache, k, k.label());
                    }
                });
        });
        muted(ui, "The attention memory of a conversation. 8-bit halves it, which is what makes long contexts fit on a small GPU.");
        ui.add_space(4.0);
        ui.checkbox(
            &mut app.settings.speculative,
            "Speculative decoding with the prediction head of Qwen3.5 models, when one is found",
        );
        muted(ui, "A Qwen3.5 model comes with a small head that guesses the next tokens; the engine checks the guesses in one pass over the weights. About 1.5 to 2 times faster answers; get the head from the model's card in My models.");
        ui.checkbox(
            &mut app.settings.lookup,
            "Prompt lookup: draft repeated text from the conversation itself (any model)",
        );
        muted(ui, "When the answer repeats words from the prompt or from earlier in the answer (code edits, quotes, lists, JSON), the engine proposes the next tokens from there and checks them in one pass over the weights. Up to twice as fast on such answers, no cost elsewhere; the text is the same a model would write without it.");
        ui.horizontal(|ui| {
            ui.label("Speculative depth");
            ui.add(egui::DragValue::new(&mut app.settings.spec_depth).range(0..=7));
            muted(ui, "Models with a built-in prediction head (Qwen3.5) draft this many tokens ahead. 0 = the engine's choice, 2 to 3 is fastest.");
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Extra arguments");
            ui.add(
                TextEdit::singleline(&mut app.settings.extra_args)
                    .desired_width(420.0)
                    .hint_text("for example: --threads 6 --chunk 512"),
            );
        });
        ui.checkbox(
            &mut app.settings.auto_load_last,
            "Load the last model again when the app starts",
        );
        ui.horizontal(|ui| {
            ui.label("Unload after");
            ui.add(egui::DragValue::new(&mut app.settings.idle_eject_minutes).range(0..=1440).suffix(" min"));
            for (label, n) in [("never", 0), ("10", 10), ("30", 30), ("60", 60)] {
                if ui.small_button(label).clicked() {
                    app.settings.idle_eject_minutes = n;
                }
            }
            muted(ui, "idle: frees video memory for games and other programs; load again from the model bar");
        });
    });
}

fn files(app: &mut StudioApp, ui: &mut Ui) {
    section(ui, "Files");
    card(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Downloads go to");
            ui.add(TextEdit::singleline(&mut app.settings.download_dir).desired_width(420.0));
            if ui.button("Open").clicked() {
                let p = std::path::PathBuf::from(&app.settings.download_dir);
                let _ = std::fs::create_dir_all(&p);
                crate::util::reveal(&p);
            }
        });
        muted(ui, "Model folders to scan are listed in My models. The download folder is scanned too if it is in that list.");
        if !app.settings.model_dirs.contains(&app.settings.download_dir)
            && ui.button("Scan the download folder too").clicked()
        {
            let d = app.settings.download_dir.clone();
            app.actions.push(Action::AddDir(d));
        }
    });
}

fn interface(app: &mut StudioApp, ui: &mut Ui) {
    section(ui, "Interface");
    card(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Zoom");
            if ui
                .add(egui::Slider::new(&mut app.settings.zoom, 0.8..=1.6).step_by(0.05))
                .changed()
            {
                ui.ctx().set_zoom_factor(app.settings.zoom);
            }
        });
    });
}

fn about(ui: &mut Ui) {
    section(ui, "About");
    card(ui, |ui| {
        kv(ui, "Forge Studio", env!("CARGO_PKG_VERSION"));
        kv(ui, "Data folder", data_dir().display().to_string());
        ui.horizontal(|ui| {
            if ui.button("Open data folder").clicked() {
                crate::util::reveal(&data_dir());
            }
            ui.hyperlink_to(
                "Source, releases and documentation",
                "https://github.com/AizenKai1001/forgeinfer",
            );
        });
        muted(ui, "Chats, settings and downloads stay on this computer. The only network use is what you ask for: model searches and downloads from huggingface.co, and a server you connect to.");
    });
}
