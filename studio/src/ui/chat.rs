//! The chat screen: conversations on the left, the conversation in the middle, generation parameters on the right.

use eframe::egui::{
    self, Align, CornerRadius, Frame, Key, KeyboardShortcut, Layout, Margin, Modifiers, Panel,
    RichText, ScrollArea, Stroke, TextEdit, Ui,
};
use std::hash::{Hash, Hasher};

use super::widgets::*;
use crate::app::{Action, StudioApp};
use crate::chats::Role;
use crate::client::{split_think, PromptProgress};
use crate::engine::EngineState;
use crate::markdown::{self, MdStyle};
use crate::settings::EngineKind;
use crate::theme;
use crate::util::{human_bytes, human_count, human_secs, truncate};

const COLUMN: f32 = 880.0;

pub fn md_style(size: f32) -> MdStyle {
    MdStyle {
        size,
        text: theme::TEXT,
        strong: theme::STRONG,
        muted: theme::MUTED,
        accent: theme::ACCENT,
        code_bg: theme::CODE_BG,
        border: theme::BORDER,
    }
}

pub fn show(app: &mut StudioApp, ui: &mut Ui) {
    // an embedding model has no conversations and no sampling parameters: only the playground
    let embedding = app.active_is_embedding();
    if !embedding {
        Panel::left("chat_list")
            .default_size(232.0)
            .size_range(190.0..=340.0)
            .resizable(true)
            .frame(
                Frame::new()
                    .fill(theme::RAIL)
                    .inner_margin(Margin::same(10))
                    .stroke(Stroke::new(1.0, theme::BORDER)),
            )
            .show(ui, |ui| chat_list(app, ui));
    }
    if app.show_params && !embedding {
        Panel::right("chat_params")
            .default_size(300.0)
            .size_range(250.0..=440.0)
            .resizable(true)
            .frame(
                Frame::new()
                    .fill(theme::PANEL)
                    .inner_margin(Margin::same(12))
                    .stroke(Stroke::new(1.0, theme::BORDER)),
            )
            .show(ui, |ui| params(app, ui));
    }
    egui::CentralPanel::default()
        .frame(Frame::new().fill(theme::PANEL))
        .show(ui, |ui| conversation(app, ui));
}

fn chat_list(app: &mut StudioApp, ui: &mut Ui) {
    if primary_button(ui, "+  New chat").clicked() {
        app.actions.push(Action::NewChat);
    }
    ui.add_space(6.0);
    if app.chat_list.len() > 4 {
        ui.add(
            TextEdit::singleline(&mut app.chat_filter)
                .hint_text("Search chats")
                .desired_width(f32::INFINITY),
        );
    } else {
        app.chat_filter.clear();
    }
    ui.add_space(6.0);
    let query = app.chat_filter.trim().to_lowercase();
    ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let current = app.chat.id.clone();
            let mut shown_current = false;
            if app.chat.messages.is_empty() {
                let _ = ui.selectable_label(true, RichText::new("New chat").color(theme::STRONG));
                shown_current = true;
            }
            for m in app.chat_list.clone() {
                if m.id == current {
                    shown_current = true;
                }
                if !query.is_empty() && !m.title.to_lowercase().contains(&query) {
                    continue;
                }
                let renaming_this = matches!(&app.renaming, Some((rid, _, _)) if *rid == m.id);
                let (rect, resp) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 28.0),
                    if renaming_this {
                        egui::Sense::hover()
                    } else {
                        egui::Sense::click()
                    },
                );
                if renaming_this {
                    let mut commit = None;
                    let mut cancel = false;
                    if let Some((_, buf, fresh)) = app.renaming.as_mut() {
                        let place = rect.shrink2(egui::vec2(2.0, 1.0));
                        let te = ui.put(
                            place,
                            TextEdit::singleline(buf).desired_width(place.width()),
                        );
                        if *fresh {
                            te.request_focus();
                            *fresh = false;
                        }
                        if ui.input(|i| i.key_pressed(Key::Escape)) {
                            cancel = true;
                        } else if te.lost_focus() {
                            commit = Some(buf.clone());
                        }
                    }
                    if let Some(title) = commit {
                        app.actions.push(Action::RenameChat(m.id.clone(), title));
                        app.renaming = None;
                    } else if cancel {
                        app.renaming = None;
                    }
                    continue;
                }
                let selected = m.id == current;
                let fill = if selected {
                    theme::CARD_HOVER
                } else if resp.hovered() {
                    theme::CARD
                } else {
                    egui::Color32::TRANSPARENT
                };
                ui.painter().rect_filled(rect, CornerRadius::same(7), fill);
                let color = if selected { theme::STRONG } else { theme::TEXT };
                ui.painter().text(
                    rect.left_center() + egui::vec2(10.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    truncate(&m.title, 27),
                    egui::FontId::proportional(14.0),
                    color,
                );
                let x_rect = egui::Rect::from_center_size(
                    rect.right_center() - egui::vec2(14.0, 0.0),
                    egui::vec2(20.0, 20.0),
                );
                let x_resp = ui.interact(x_rect, resp.id.with("del"), egui::Sense::click());
                if resp.hovered() || x_resp.hovered() {
                    ui.painter().text(
                        x_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "\u{00D7}",
                        egui::FontId::proportional(17.0),
                        if x_resp.hovered() {
                            theme::BAD
                        } else {
                            theme::MUTED
                        },
                    );
                }
                resp.clone().on_hover_text("Double-click to rename");
                if x_resp.clicked() {
                    app.actions.push(Action::DeleteChat(m.id.clone()));
                } else if resp.double_clicked() {
                    app.renaming = Some((m.id.clone(), m.title.clone(), true));
                } else if resp.clicked() {
                    app.actions.push(Action::OpenChat(m.id.clone()));
                }
            }
            let _ = shown_current;
        });
}

fn conversation(app: &mut StudioApp, ui: &mut Ui) {
    Panel::top("chat_top")
        .exact_size(46.0)
        .resizable(false)
        .frame(
            Frame::new()
                .fill(theme::PANEL)
                .inner_margin(Margin::symmetric(14, 6))
                .stroke(Stroke::new(1.0, theme::BORDER)),
        )
        .show(ui, |ui| top_bar(app, ui));
    // an embedding model has no conversation: the playground takes the middle
    if app.active_is_embedding() {
        egui::CentralPanel::default()
            .frame(Frame::new().fill(theme::PANEL).inner_margin(Margin::symmetric(16, 8)))
            .show(ui, |ui| super::embed::show(app, ui));
        return;
    }
    Panel::bottom("chat_input")
        .resizable(false)
        .frame(
            Frame::new()
                .fill(theme::PANEL)
                .inner_margin(Margin::symmetric(16, 10)),
        )
        .show(ui, |ui| input_box(app, ui));
    egui::CentralPanel::default()
        .frame(
            Frame::new()
                .fill(theme::PANEL)
                .inner_margin(Margin::symmetric(16, 8)),
        )
        .show(ui, |ui| messages(app, ui));
}

fn top_bar(app: &mut StudioApp, ui: &mut Ui) {
    ui.horizontal_centered(|ui| {
        model_picker(app, ui);
        if app.active_is_embedding() {
            return;
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui
                .selectable_label(app.show_params, "\u{2699}  Parameters")
                .clicked()
            {
                app.show_params = !app.show_params;
            }
            if !app.chat.messages.is_empty()
                && ui
                    .button("Copy chat")
                    .on_hover_text("Copy the whole conversation as Markdown")
                    .clicked()
            {
                ui.ctx().copy_text(app.chat.to_markdown());
                app.actions
                    .push(Action::Toast("Chat copied as Markdown".to_string()));
            }
        });
    });
}

fn model_picker(app: &mut StudioApp, ui: &mut Ui) {
    // in a narrow window (the minimum is 900 points) the picker shrinks and the status words go: the buttons on the right need about 225 points, and the side panel shows the state and Eject anyway
    let avail = ui.available_width();
    let narrow = avail < 640.0;
    let picker_w = if narrow {
        (avail - 225.0).clamp(140.0, 380.0)
    } else {
        380.0
    };
    if app.settings.engine == EngineKind::External {
        let current = if app.external_pick.is_empty() {
            "no model found".to_string()
        } else {
            app.external_pick.clone()
        };
        egui::ComboBox::from_id_salt("model_pick_ext")
            .width(picker_w.min(320.0))
            .selected_text(current)
            .show_ui(ui, |ui| {
                for m in app.external_models.clone() {
                    if ui.selectable_label(m == app.external_pick, &m).clicked() {
                        app.external_pick = m;
                    }
                }
            });
        if ui.button("Refresh").clicked() {
            app.actions.push(Action::RefreshExternal);
        }
        if !narrow {
            muted(ui, &app.external_status.clone());
        }
        return;
    }
    let state = app.engine.state();
    let current = match &state {
        EngineState::Ready { model, .. } | EngineState::Loading { model, .. } => model.clone(),
        _ => "Select a model to load".to_string(),
    };
    let mut pick: Option<std::path::PathBuf> = None;
    egui::ComboBox::from_id_salt("model_pick").width(picker_w).selected_text(RichText::new(truncate(&current, if narrow { 22 } else { 46 })).color(theme::STRONG)).show_ui(ui, |ui| {
        ui.set_min_width(460.0);
        let mut any = false;
        for e in app.library.iter().filter(|e| !e.projector && e.error.is_none()) {
            any = true;
            let issue = if app.settings.engine == EngineKind::Forge { e.forge_issue() } else { None };
            let label = format!("{}    {} \u{00B7} {}", e.name, e.quant, human_bytes(e.size));
            let r = ui.add_enabled(issue.is_none(), egui::Button::selectable(e.name == current, label));
            if let Some(why) = issue {
                r.on_disabled_hover_text(format!("The ForgeInfer engine cannot run this file: {why}. The llama.cpp engine in Settings may."));
            } else if r.clicked() {
                pick = Some(e.path.clone());
                ui.close();
            }
        }
        if !any {
            ui.label(RichText::new("No models yet: add a folder in My models, or download one in Discover.").color(theme::MUTED));
        }
    });
    if let Some(p) = pick {
        app.actions.push(Action::LoadModel(p));
    }
    match state {
        EngineState::Loading { since, .. } => {
            ui.spinner();
            if !narrow {
                ui.label(
                    RichText::new(format!(
                        "loading ... {:.0} s",
                        since.elapsed().as_secs_f32()
                    ))
                    .color(theme::WARN),
                );
            }
        }
        EngineState::Ready { .. } if narrow => {}
        EngineState::Ready { load_secs, .. } => {
            let spec = app.engine.current().is_some_and(|l| l.mtp.is_some());
            let docs = app.engine.current().is_some_and(|l| l.embedding.is_some());
            ui.label(RichText::new(format!("ready \u{00B7} {load_secs:.1} s{}{}", if spec { " \u{00B7} speculative" } else { "" }, if docs { " \u{00B7} documents" } else { "" })).small().color(theme::GOOD)).on_hover_text("Time to load. Speculative: the model's prediction head drafts tokens ahead for faster answers. Documents: an embedding model runs next to this one, so a file that is too long for the context is searched instead of sent whole.");
            if ui.small_button("Eject").clicked() {
                app.actions.push(Action::Eject);
            }
        }
        EngineState::Failed { message, .. } => {
            // the reason fits between the picker and the buttons on the right
            let w = (ui.available_width() - 230.0).max(60.0);
            ui.allocate_ui_with_layout(
                egui::vec2(w, 20.0),
                Layout::left_to_right(Align::Center),
                |ui| {
                    ui.add(
                        egui::Label::new(RichText::new(&message).small().color(theme::BAD))
                            .truncate(),
                    )
                    .on_hover_text(&message);
                },
            );
        }
        EngineState::Idle => {}
    }
}

fn centered(ui: &mut Ui, width: f32, add: impl FnOnce(&mut Ui)) {
    let w = ui.available_width().min(width);
    ui.vertical_centered(|ui| {
        ui.allocate_ui_with_layout(egui::vec2(w, 0.0), Layout::top_down(Align::Min), add);
    });
}

fn messages(app: &mut StudioApp, ui: &mut Ui) {
    ScrollArea::vertical()
        .id_salt("messages")
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            centered(ui, COLUMN, |ui| {
                if app.chat.messages.is_empty() {
                    empty_state(app, ui);
                    return;
                }
                ui.add_space(8.0);
                let n = app.chat.messages.len();
                for i in 0..n {
                    message_view(app, ui, i, i + 1 == n);
                    ui.add_space(14.0);
                }
                ui.add_space(8.0);
            });
        });
}

fn empty_state(app: &mut StudioApp, ui: &mut Ui) {
    ui.add_space(70.0);
    ui.vertical_centered(|ui| {
        ui.label(
            RichText::new("Forge Studio")
                .size(30.0)
                .color(theme::STRONG)
                .family(theme::bold_family()),
        );
        ui.add_space(4.0);
        ui.label(
            RichText::new("Chat with a model that runs on this computer.").color(theme::MUTED),
        );
        ui.add_space(18.0);
        let ready = app.active_endpoint().is_some();
        if let (false, Some(problem)) = (ready, app.engine_program_problem()) {
            ui.allocate_ui_with_layout(
                egui::vec2(560.0, 0.0),
                Layout::top_down(Align::Min),
                |ui| banner(ui, &problem, theme::WARN, theme::CARD),
            );
            ui.add_space(8.0);
            if primary_button(ui, "Open Settings").clicked() {
                app.screen = crate::app::Screen::Settings;
            }
            ui.add_space(10.0);
        }
        if let EngineState::Failed { model, message, .. } = app.engine.state() {
            let m = message.to_lowercase();
            let no_gpu = m.contains("no_device") || m.contains("cuinit") || m.contains("nvcuda") || m.contains("no cuda");
            ui.allocate_ui_with_layout(egui::vec2(560.0, 0.0), Layout::top_down(Align::Min), |ui| {
                banner(ui, &format!("{model} did not load: {message}"), theme::BAD, theme::CARD);
                if no_gpu {
                    ui.add_space(6.0);
                    banner(ui, "The ForgeInfer engine needs an NVIDIA graphics card with an up-to-date driver, and none was found. In Settings the engine can be the llama.cpp server (any GPU, or the CPU) or a server that is already running.", theme::WARN, theme::CARD);
                }
            });
            ui.add_space(8.0);
            if no_gpu && primary_button(ui, "Open Settings").clicked() {
                app.screen = crate::app::Screen::Settings;
            }
            ui.add_space(10.0);
        }
        if ready {
            ui.label(RichText::new("Ask anything below.").color(theme::TEXT));
        } else if app.settings.engine == EngineKind::External {
            ui.label(
                RichText::new("No server answered yet. Check the address in Settings.")
                    .color(theme::TEXT),
            );
        } else if app
            .library
            .iter()
            .any(|e| !e.projector && e.error.is_none())
        {
            ui.label(RichText::new("Pick a model in the bar above to load it.").color(theme::TEXT));
            if !app.settings.last_model.is_empty() {
                let p = std::path::PathBuf::from(&app.settings.last_model);
                if p.is_file() {
                    ui.add_space(8.0);
                    let name = p
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default();
                    if primary_button(ui, &format!("Load {}", truncate(&name, 40))).clicked() {
                        app.actions.push(Action::LoadModel(p));
                    }
                }
            }
        } else {
            ui.label(RichText::new("No models found yet.").color(theme::TEXT));
            ui.add_space(8.0);
            if primary_button(ui, "Find a model to download").clicked() {
                app.screen = crate::app::Screen::Discover;
            }
        }
    });
}

/// Under a question about searched files: how the search is going, then the passages the model was given.
fn passages(app: &StudioApp, ui: &mut Ui, m: &crate::chats::Message) {
    if m.retrieving {
        ui.horizontal(|ui| {
            ui.spinner();
            let text = match &app.retrieval {
                Some(r) => {
                    let total = r.progress.total.load(std::sync::atomic::Ordering::Relaxed);
                    let done = r.progress.done.load(std::sync::atomic::Ordering::Relaxed);
                    if total > done || (total > 0 && done == 0) {
                        format!("reading your files: {} of {} passages ({} s)", human_count(done as u64), human_count(total as u64), r.started.elapsed().as_secs())
                    } else {
                        "looking for the passages that answer this ...".to_string()
                    }
                }
                None => "searching your files ...".to_string(),
            };
            ui.label(RichText::new(text).small().color(theme::ACCENT));
        });
        return;
    }
    if !m.by_retrieval {
        return;
    }
    let n = m.citations.len();
    let weak = m.citations.first().is_some_and(|c| c.weak);
    let title = if n == 0 {
        format!("no passage of your files matched (minimum score {:.2})", app.settings.docs_threshold)
    } else if weak {
        format!("no close match: the model read the {n} nearest passage{} of your files", if n == 1 { "" } else { "s" })
    } else {
        format!("{n} passage{} of your files read by the model", if n == 1 { "" } else { "s" })
    };
    egui::CollapsingHeader::new(RichText::new(title).small().color(theme::MUTED)).id_salt(("passages", &m.id)).default_open(app.opts.open_passages).show(ui, |ui| {
        for c in &m.citations {
            ui.label(RichText::new(format!("{} \u{00B7} score {:.2}", c.file, c.score)).small().color(theme::ACCENT));
            ui.add(egui::Label::new(RichText::new(truncate(&c.text, 900)).small().color(theme::TEXT)).wrap());
            ui.add_space(4.0);
        }
    });
}

fn id_hash(s: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn message_view(app: &mut StudioApp, ui: &mut Ui, i: usize, last: bool) {
    let m = app.chat.messages[i].clone();
    let width = ui.available_width();
    if i == app.chat.skip && app.chat.skip > 0 {
        // the conversation outgrew the context: what is above this line is no longer read by the model
        ui.add_space(4.0);
        ui.vertical_centered(|ui| {
            ui.label(RichText::new("the messages above are left out of what the model reads: the context is full").small().color(theme::MUTED))
                .on_hover_text("The conversation no longer fits the context the model was loaded with. The oldest turns stay on screen but are not sent. Settings: A conversation longer than the context.");
        });
        ui.add_space(4.0);
    }
    match m.role {
        Role::User | Role::System if matches!(&app.editing, Some((idx, _, _)) if *idx == i) => {
            let mut save = None;
            let mut cancel = false;
            ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                Frame::new()
                    .fill(theme::USER_BUBBLE)
                    .corner_radius(CornerRadius::same(14))
                    .inner_margin(Margin::symmetric(14, 10))
                    .show(ui, |ui| {
                        let w = (width * 0.82 - 28.0).max(240.0);
                        ui.set_width(w);
                        // the frame inherits the right-to-left layout of its row: stack the editor and its buttons explicitly
                        ui.vertical(|ui| {
                            if let Some((_, buf, fresh)) = app.editing.as_mut() {
                                let te = ui.add(
                                    TextEdit::multiline(buf)
                                        .desired_width(w)
                                        .desired_rows(2)
                                        .font(egui::FontId::proportional(15.0)),
                                );
                                if *fresh {
                                    te.request_focus();
                                    *fresh = false;
                                }
                                ui.add_space(4.0);
                                ui.horizontal(|ui| {
                                    if primary_button(ui, "Save and resend").clicked() {
                                        save = Some(buf.clone());
                                    }
                                    if ui.button("Cancel").clicked() {
                                        cancel = true;
                                    }
                                    ui.label(
                                        RichText::new(
                                            "The answers after this message are dropped.",
                                        )
                                        .small()
                                        .color(theme::MUTED),
                                    );
                                });
                            }
                        });
                    });
            });
            if let Some(text) = save {
                app.actions.push(Action::EditResend(i, text));
            } else if cancel || ui.input(|i| i.key_pressed(Key::Escape)) {
                app.editing = None;
            }
        }
        Role::User | Role::System => {
            ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                Frame::new()
                    .fill(theme::USER_BUBBLE)
                    .corner_radius(CornerRadius::same(14))
                    .inner_margin(Margin::symmetric(14, 10))
                    .show(ui, |ui| {
                        ui.set_max_width(width * 0.82);
                        ui.vertical(|ui| {
                            for a in &m.attachments {
                                ui.label(
                                    RichText::new(format!(
                                        "attached: {} \u{00B7} {}{}",
                                        a.name,
                                        human_bytes(a.text.len() as u64),
                                        if m.by_retrieval { " \u{00B7} searched, not sent whole" } else { "" }
                                    ))
                                    .small()
                                    .color(theme::MUTED),
                                );
                            }
                            ui.add(
                                egui::Label::new(
                                    RichText::new(&m.content).size(15.0).color(theme::TEXT),
                                )
                                .wrap(),
                            );
                            passages(app, ui, &m);
                        });
                    });
            });
            ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                if !app.is_generating()
                    && ui
                        .add(
                            egui::Button::new(RichText::new("Delete").small().color(theme::MUTED))
                                .frame(false),
                        )
                        .clicked()
                {
                    app.actions.push(Action::DeleteMessage(i));
                }
                if ui
                    .add(
                        egui::Button::new(RichText::new("Copy").small().color(theme::MUTED))
                            .frame(false),
                    )
                    .clicked()
                {
                    ui.ctx().copy_text(m.content.clone());
                }
                if m.role == Role::User
                    && !app.is_generating()
                    && ui
                        .add(
                            egui::Button::new(RichText::new("Edit").small().color(theme::MUTED))
                                .frame(false),
                        )
                        .clicked()
                {
                    app.editing = Some((i, m.content.clone(), true));
                }
            });
        }
        Role::Assistant => {
            ui.horizontal(|ui| {
                dot(ui, theme::ACCENT);
                let who = if app.chat.model.is_empty() {
                    "Assistant".to_string()
                } else {
                    app.chat.model.clone()
                };
                ui.label(RichText::new(who).small().color(theme::MUTED));
            });
            let (think, answer, think_open) = if !m.reasoning.is_empty() {
                (m.reasoning.clone(), m.content.clone(), false)
            } else {
                let (t, a, o) = split_think(&m.content);
                (t, a, o)
            };
            let thinking_now =
                m.streaming && answer.is_empty() && (!think.is_empty() || think_open);
            if !think.is_empty() {
                let title = if thinking_now {
                    "Thinking ...".to_string()
                } else if m.thought_secs > 0.0 {
                    format!("Thought for {}", human_secs(m.thought_secs as f64))
                } else {
                    "Reasoning".to_string()
                };
                egui::CollapsingHeader::new(RichText::new(title).color(theme::MUTED))
                    .id_salt((m.id.as_str(), thinking_now))
                    .default_open(thinking_now)
                    .show(ui, |ui| {
                        Frame::new()
                            .fill(theme::CODE_BG)
                            .corner_radius(CornerRadius::same(8))
                            .inner_margin(Margin::same(10))
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                let mut style = md_style(13.5);
                                style.text = theme::MUTED;
                                let blocks = markdown::parse(&think);
                                markdown::render(
                                    ui,
                                    &blocks,
                                    &style,
                                    id_hash(&m.id) ^ 1,
                                    theme::bold_family(),
                                );
                            });
                    });
            }
            if m.streaming && answer.is_empty() && think.is_empty() {
                // a server that reports how far it has read the prompt (a long document takes seconds): a bar and what is left
                let progress = if last { app.generation.as_ref().and_then(|g| g.progress) } else { None };
                ui.horizontal(|ui| {
                    ui.spinner();
                    let text = progress.map(|p| reading_text(&p)).unwrap_or_else(|| "waiting for the first token ...".to_string());
                    ui.label(RichText::new(text).small().color(theme::MUTED));
                });
                if let Some(p) = progress {
                    ui.add(egui::ProgressBar::new(p.fraction()).desired_width(280.0).desired_height(6.0).fill(theme::ACCENT));
                }
            }
            if !answer.is_empty() {
                let blocks = markdown::parse(&answer);
                markdown::render(
                    ui,
                    &blocks,
                    &md_style(15.0),
                    id_hash(&m.id),
                    theme::bold_family(),
                );
            }
            if m.streaming && !answer.is_empty() {
                ui.add_space(2.0);
                ui.spinner();
            }
            if let Some(e) = &m.error {
                banner(ui, e, theme::BAD, theme::BAD_BG);
            }
            if !m.streaming {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 10.0;
                    if let Some(s) = &m.stats {
                        ui.label(RichText::new(format!("{} tokens  \u{00B7}  {:.1} tok/s  \u{00B7}  {:.2} s to first token", s.completion_tokens, s.tok_per_s, s.first_token_s)).small().color(theme::MUTED));
                        // how much of the conversation the engine did not have to read again (its prefix reuse and its chat memory): only worth a line when it was a lot
                        if s.cached_tokens >= 256 && s.prompt_tokens >= s.cached_tokens {
                            ui.label(RichText::new(format!("{} of {} prompt tokens already in memory", grouped(s.cached_tokens), grouped(s.prompt_tokens))).small().color(theme::MUTED));
                        }
                    }
                    if !answer.is_empty() && ui.add(egui::Button::new(RichText::new("Copy").small().color(theme::MUTED)).frame(false)).clicked() {
                        ui.ctx().copy_text(answer.clone());
                    }
                    if last && !app.is_generating() && ui.add(egui::Button::new(RichText::new("Regenerate").small().color(theme::MUTED)).frame(false)).clicked() {
                        app.actions.push(Action::Regenerate);
                    }
                    if !app.is_generating() && ui.add(egui::Button::new(RichText::new("Delete").small().color(theme::MUTED)).frame(false)).clicked() {
                        app.actions.push(Action::DeleteMessage(i));
                    }
                });
            }
        }
    }
}

/// 12288 -> "12,288"
fn grouped(n: u32) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// "Reading the prompt: 12,288 of 27,000 tokens (45 %), about 4 s left": what the server reported while it works through a long prompt.
fn reading_text(p: &PromptProgress) -> String {
    let mut t = format!("Reading the prompt: {} of {} tokens ({:.0} %)", grouped(p.processed), grouped(p.total), 100.0 * p.fraction());
    if let Some(left) = p.seconds_left() {
        if left >= 1.0 {
            t.push_str(&format!(", about {} left", human_secs(left as f64)));
        }
    }
    t
}

/// "Context: 1,234 of 8,192 tokens": what the last answer's prompt and reply took of the loaded context (local engines only).
fn context_meter(app: &StudioApp) -> Option<String> {
    if app.settings.engine == EngineKind::External {
        return None;
    }
    let used = app
        .chat
        .messages
        .iter()
        .rev()
        .find_map(|m| m.stats.as_ref())
        .map(|s| s.prompt_tokens + s.completion_tokens)?;
    app.engine.current()?;
    let total = app.effective_ctx();
    Some(format!(
        "Context: {used} of {total} tokens ({:.0} %)",
        100.0 * used as f64 / total.max(1) as f64
    ))
}

fn input_box(app: &mut StudioApp, ui: &mut Ui) {
    centered(ui, COLUMN, |ui| {
        Frame::new().fill(theme::CARD).corner_radius(CornerRadius::same(14)).stroke(Stroke::new(1.0, theme::BORDER)).inner_margin(Margin::symmetric(12, 8)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            if !app.attachments.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    let mut remove = None;
                    for (i, a) in app.attachments.iter().enumerate() {
                        let label = format!("{} \u{00B7} {}", truncate(&a.name, 28), human_bytes(a.text.len() as u64));
                        Frame::new().fill(theme::CARD_HOVER).corner_radius(CornerRadius::same(8)).inner_margin(Margin::symmetric(8, 3)).show(ui, |ui| {
                            ui.label(RichText::new(label).small().color(theme::TEXT));
                            if ui.add(egui::Button::new(RichText::new("\u{00D7}").color(theme::MUTED)).frame(false)).on_hover_text("Remove this file").clicked() {
                                remove = Some(i);
                            }
                        });
                    }
                    if let Some(i) = remove {
                        app.actions.push(Action::RemoveAttachment(i));
                    }
                    // what the model would have to read, against the context it was loaded with
                    let chars: usize = app.chat.api_messages(&app.settings.system_prompt).iter().map(|(_, t)| t.chars().count()).sum::<usize>()
                        + app.attachments.iter().map(|a| a.text.chars().count() + 24).sum::<usize>()
                        + app.input.chars().count();
                    let tokens = crate::chats::estimate_tokens(chars);
                    let ctx = app.effective_ctx() as usize;
                    let over = tokens + 300 > ctx;
                    if app.would_search(&app.attachments, &app.input) {
                        ui.label(RichText::new(format!("about {} tokens: too much to send whole, the chat searches the files and the model reads the best passages for each question", human_count(tokens as u64))).small().color(theme::ACCENT));
                    } else {
                        let hint = if over && app.settings.engine == EngineKind::Forge && app.embedding_endpoint().is_none() { ", or let the chat search the files (Settings > Documents)" } else { "" };
                        ui.label(RichText::new(format!("about {} of {} tokens{}", human_count(tokens as u64), human_count(ctx as u64), if over { format!(": too long, raise the context length in Settings or attach less{hint}") } else { String::new() })).small().color(if over { theme::BAD } else { theme::MUTED }));
                    }
                });
                ui.add_space(4.0);
            }
            ui.horizontal(|ui| {
                let button_w = 92.0 + 78.0;
                let edit_w = (ui.available_width() - button_w - 12.0).max(120.0);
                let enter_sends = KeyboardShortcut::new(Modifiers::SHIFT, Key::Enter);
                let resp = ScrollArea::vertical()
                    .id_salt("input_scroll")
                    .max_height(170.0)
                    .auto_shrink([true, true])
                    .show(ui, |ui| {
                        ui.add(
                            TextEdit::multiline(&mut app.input)
                                .hint_text(if app.attachments.is_empty() {
                                    "Message the model ...   (Enter sends, Shift+Enter adds a line)"
                                } else {
                                    "What should the model do with the file?   (Enter sends)"
                                })
                                .desired_rows(1)
                                .desired_width(edit_w)
                                .frame(Frame::NONE)
                                .return_key(Some(enter_sends)),
                        )
                    })
                    .inner;
                if app.focus_input {
                    resp.request_focus();
                    app.focus_input = false;
                }
                let enter = resp.has_focus() && ui.input(|i| i.key_pressed(Key::Enter) && !i.modifiers.shift && !i.modifiers.command);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if app.is_generating() {
                        if danger_button(ui, "Stop").clicked() {
                            app.actions.push(Action::Stop);
                        }
                    } else {
                        // (added first, drawn last in this right-to-left row: it sits left of Send)
                        let think = app.settings.gen.thinking;
                        let can = !app.input.trim().is_empty() && app.active_endpoint().is_some();
                        if primary_enabled(ui, !app.input.trim().is_empty(), "Send").clicked() || (enter && can) {
                            app.actions.push(Action::Send);
                            app.focus_input = true;
                        } else if enter && !can && !app.input.trim().is_empty() {
                            app.actions.push(Action::Send);
                        }
                        let chip = egui::Button::new(RichText::new(format!("Think: {}", think.short())).small().color(if think == crate::settings::ThinkingMode::Off { theme::MUTED } else { theme::TEXT })).frame(false);
                        if ui.add(chip).on_hover_text("Whether the model reasons before it answers (Qwen3 and Qwen3.5). auto: the model's own default (the 4B and larger think, which can take a minute on a simple question); off: answers directly; on: always thinks. Click to change.").clicked() {
                            app.settings.gen.thinking = think.next();
                        }
                    }
                });
            });
        });
        if app.chat.messages.is_empty() && app.attachments.is_empty() && app.input.is_empty() {
            ui.add_space(2.0);
            ui.label(
                RichText::new(
                    "Tip: drop a text or code file on the window to attach it to your message.",
                )
                .small()
                .color(theme::MUTED),
            );
        }
        if let Some(text) = context_meter(app) {
            ui.add_space(2.0);
            ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                ui.label(RichText::new(text).small().color(theme::MUTED));
            });
        }
    });
}

fn params(app: &mut StudioApp, ui: &mut Ui) {
    heading(ui, "Parameters");
    muted(ui, "Used for the next message.");
    ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        section(ui, "Preset");
        ui.horizontal(|ui| {
            let current = if app.settings.preset.is_empty() { "none".to_string() } else { app.settings.preset.clone() };
            let mut apply: Option<String> = None;
            egui::ComboBox::from_id_salt("preset_pick").selected_text(current).show_ui(ui, |ui| {
                for p in &app.settings.presets {
                    if ui.selectable_label(p.name == app.settings.preset, &p.name).clicked() {
                        apply = Some(p.name.clone());
                    }
                }
                if app.settings.presets.is_empty() {
                    ui.label(RichText::new("No presets yet: type a name below and save.").color(theme::MUTED));
                }
            });
            if let Some(n) = apply {
                app.settings.apply_preset(&n);
            }
            if !app.settings.preset.is_empty() && ui.small_button("Delete").on_hover_text("Delete this preset").clicked() {
                let n = app.settings.preset.clone();
                app.settings.delete_preset(&n);
            }
        });
        ui.horizontal(|ui| {
            ui.add(TextEdit::singleline(&mut app.preset_name).hint_text("Name for a new preset").desired_width(150.0));
            if ui.small_button("Save").on_hover_text("Keep the system prompt and the sampler settings under this name (the same name replaces)").clicked() {
                let n = std::mem::take(&mut app.preset_name);
                if !app.settings.save_preset(&n) {
                    app.preset_name = n;
                }
            }
        });
        let g = &mut app.settings.gen;
        section(ui, "Sampling");
        ui.add(egui::Slider::new(&mut g.temperature, 0.0..=2.0).text("Temperature"));
        ui.add(egui::Slider::new(&mut g.top_p, 0.0..=1.0).text("Top P"));
        ui.add(egui::Slider::new(&mut g.top_k, 0..=200).text("Top K"));
        ui.add(egui::Slider::new(&mut g.min_p, 0.0..=0.5).text("Min P"));
        ui.add(egui::Slider::new(&mut g.repeat_penalty, 1.0..=1.5).text("Repeat penalty"));
        ui.horizontal(|ui| {
            ui.label("Max tokens");
            ui.add(egui::DragValue::new(&mut g.max_tokens).range(0..=200_000).speed(16.0));
            muted(ui, "0 = no limit");
        });
        ui.horizontal(|ui| {
            ui.label("Thinking");
            egui::ComboBox::from_id_salt("thinking_mode").selected_text(g.thinking.label()).show_ui(ui, |ui| {
                for m in crate::settings::ThinkingMode::ALL {
                    ui.selectable_value(&mut g.thinking, m, m.label());
                }
            });
        });
        muted(ui, "For Qwen3 and Qwen3.5. The 0.8B and 2B answer directly unless told to think; the 4B and larger think first (also on the quick switch beside Send).");
        section(ui, "System prompt");
        ui.add(TextEdit::multiline(&mut app.settings.system_prompt).desired_rows(5).desired_width(f32::INFINITY).hint_text("You are a helpful assistant."));
        muted(ui, "Sent first in every conversation.");
        section(ui, "This chat");
        ui.add(TextEdit::multiline(&mut app.chat.system).desired_rows(3).desired_width(f32::INFINITY).hint_text("A system prompt for this chat only (empty = the one above)"));
        ui.add_space(8.0);
        if ui.button("Reset sampling").clicked() {
            app.settings.gen = crate::settings::GenParams::default();
        }
        ui.add_space(10.0);
        muted(ui, "Context length and the KV cache are set when a model loads (Settings).");
    });
}

#[cfg(test)]
mod progress_tests {
    use super::*;

    #[test]
    fn the_reading_line_names_the_tokens_and_the_time_left() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(12288), "12,288");
        assert_eq!(grouped(1234567), "1,234,567");
        let p = PromptProgress { processed: 13288, total: 27000, cached: 1000, time_ms: 3000.0 };
        assert_eq!(reading_text(&p), "Reading the prompt: 13,288 of 27,000 tokens (49 %), about 3.3 s left");
        let start = PromptProgress { processed: 1024, total: 27000, cached: 0, time_ms: 10.0 };
        assert_eq!(reading_text(&start), "Reading the prompt: 1,024 of 27,000 tokens (4 %)");
    }
}
