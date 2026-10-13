//! The chat screen: conversations on the left, the conversation in the middle, generation parameters on the right.

use eframe::egui::{self, Align, CornerRadius, Frame, Key, KeyboardShortcut, Layout, Margin, Modifiers, Panel, RichText, ScrollArea, Stroke, TextEdit, Ui};
use std::hash::{Hash, Hasher};

use super::widgets::*;
use crate::app::{Action, StudioApp};
use crate::chats::Role;
use crate::client::split_think;
use crate::engine::EngineState;
use crate::markdown::{self, MdStyle};
use crate::settings::EngineKind;
use crate::theme;
use crate::util::{human_bytes, human_secs, truncate};

const COLUMN: f32 = 880.0;

pub fn md_style(size: f32) -> MdStyle {
    MdStyle { size, text: theme::TEXT, strong: theme::STRONG, muted: theme::MUTED, accent: theme::ACCENT, code_bg: theme::CODE_BG, border: theme::BORDER }
}

pub fn show(app: &mut StudioApp, ui: &mut Ui) {
    Panel::left("chat_list")
        .default_size(232.0)
        .size_range(190.0..=340.0)
        .resizable(true)
        .frame(Frame::new().fill(theme::RAIL).inner_margin(Margin::same(10)).stroke(Stroke::new(1.0, theme::BORDER)))
        .show(ui, |ui| chat_list(app, ui));
    if app.show_params {
        Panel::right("chat_params")
            .default_size(300.0)
            .size_range(250.0..=440.0)
            .resizable(true)
            .frame(Frame::new().fill(theme::PANEL).inner_margin(Margin::same(12)).stroke(Stroke::new(1.0, theme::BORDER)))
            .show(ui, |ui| params(app, ui));
    }
    egui::CentralPanel::default().frame(Frame::new().fill(theme::PANEL)).show(ui, |ui| conversation(app, ui));
}

fn chat_list(app: &mut StudioApp, ui: &mut Ui) {
    if primary_button(ui, "+  New chat").clicked() {
        app.actions.push(Action::NewChat);
    }
    ui.add_space(8.0);
    ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
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
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 28.0), egui::Sense::click());
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
            ui.painter().text(rect.left_center() + egui::vec2(10.0, 0.0), egui::Align2::LEFT_CENTER, truncate(&m.title, 27), egui::FontId::proportional(14.0), color);
            let x_rect = egui::Rect::from_center_size(rect.right_center() - egui::vec2(14.0, 0.0), egui::vec2(20.0, 20.0));
            let x_resp = ui.interact(x_rect, resp.id.with("del"), egui::Sense::click());
            if resp.hovered() || x_resp.hovered() {
                ui.painter().text(x_rect.center(), egui::Align2::CENTER_CENTER, "\u{00D7}", egui::FontId::proportional(17.0), if x_resp.hovered() { theme::BAD } else { theme::MUTED });
            }
            if x_resp.clicked() {
                app.actions.push(Action::DeleteChat(m.id.clone()));
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
        .frame(Frame::new().fill(theme::PANEL).inner_margin(Margin::symmetric(14, 6)).stroke(Stroke::new(1.0, theme::BORDER)))
        .show(ui, |ui| top_bar(app, ui));
    Panel::bottom("chat_input").resizable(false).frame(Frame::new().fill(theme::PANEL).inner_margin(Margin::symmetric(16, 10))).show(ui, |ui| input_box(app, ui));
    egui::CentralPanel::default().frame(Frame::new().fill(theme::PANEL).inner_margin(Margin::symmetric(16, 8))).show(ui, |ui| messages(app, ui));
}

fn top_bar(app: &mut StudioApp, ui: &mut Ui) {
    ui.horizontal_centered(|ui| {
        model_picker(app, ui);
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui.selectable_label(app.show_params, "\u{2699}  Parameters").clicked() {
                app.show_params = !app.show_params;
            }
            if !app.chat.messages.is_empty() && ui.button("Copy chat").on_hover_text("Copy the whole conversation as Markdown").clicked() {
                ui.ctx().copy_text(app.chat.to_markdown());
                app.actions.push(Action::Toast("Chat copied as Markdown".to_string()));
            }
        });
    });
}

fn model_picker(app: &mut StudioApp, ui: &mut Ui) {
    if app.settings.engine == EngineKind::External {
        let current = if app.external_pick.is_empty() { "no model found".to_string() } else { app.external_pick.clone() };
        egui::ComboBox::from_id_salt("model_pick_ext").width(320.0).selected_text(current).show_ui(ui, |ui| {
            for m in app.external_models.clone() {
                if ui.selectable_label(m == app.external_pick, &m).clicked() {
                    app.external_pick = m;
                }
            }
        });
        if ui.button("Refresh").clicked() {
            app.actions.push(Action::RefreshExternal);
        }
        muted(ui, &app.external_status.clone());
        return;
    }
    let state = app.engine.state();
    let current = match &state {
        EngineState::Ready { model, .. } | EngineState::Loading { model, .. } => model.clone(),
        _ => "Select a model to load".to_string(),
    };
    let mut pick: Option<std::path::PathBuf> = None;
    egui::ComboBox::from_id_salt("model_pick").width(380.0).selected_text(RichText::new(truncate(&current, 46)).color(theme::STRONG)).show_ui(ui, |ui| {
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
            ui.label(RichText::new(format!("loading ... {:.0} s", since.elapsed().as_secs_f32())).color(theme::WARN));
        }
        EngineState::Ready { load_secs, .. } => {
            ui.label(RichText::new(format!("ready \u{00B7} loaded in {load_secs:.1} s")).small().color(theme::GOOD));
            if ui.small_button("Eject").clicked() {
                app.actions.push(Action::Eject);
            }
        }
        EngineState::Failed { message, .. } => {
            ui.add(egui::Label::new(RichText::new(truncate(&message, 90)).small().color(theme::BAD)).truncate()).on_hover_text(message);
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
    ScrollArea::vertical().id_salt("messages").auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
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
        ui.label(RichText::new("Forge Studio").size(30.0).color(theme::STRONG).family(theme::bold_family()));
        ui.add_space(4.0);
        ui.label(RichText::new("Chat with a model that runs on this computer.").color(theme::MUTED));
        ui.add_space(18.0);
        let ready = app.active_endpoint().is_some();
        if ready {
            ui.label(RichText::new("Ask anything below.").color(theme::TEXT));
        } else if app.settings.engine == EngineKind::External {
            ui.label(RichText::new("No server answered yet. Check the address in Settings.").color(theme::TEXT));
        } else if app.library.iter().any(|e| !e.projector && e.error.is_none()) {
            ui.label(RichText::new("Pick a model in the bar above to load it.").color(theme::TEXT));
            if !app.settings.last_model.is_empty() {
                let p = std::path::PathBuf::from(&app.settings.last_model);
                if p.is_file() {
                    ui.add_space(8.0);
                    let name = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
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

fn id_hash(s: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn message_view(app: &mut StudioApp, ui: &mut Ui, i: usize, last: bool) {
    let m = app.chat.messages[i].clone();
    let width = ui.available_width();
    match m.role {
        Role::User | Role::System => {
            ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                Frame::new().fill(theme::USER_BUBBLE).corner_radius(CornerRadius::same(14)).inner_margin(Margin::symmetric(14, 10)).show(ui, |ui| {
                    ui.set_max_width(width * 0.82);
                    ui.add(egui::Label::new(RichText::new(&m.content).size(15.0).color(theme::TEXT)).wrap());
                });
            });
            ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                if !app.is_generating() && ui.add(egui::Button::new(RichText::new("Delete").small().color(theme::MUTED)).frame(false)).clicked() {
                    app.actions.push(Action::DeleteMessage(i));
                }
                if ui.add(egui::Button::new(RichText::new("Copy").small().color(theme::MUTED)).frame(false)).clicked() {
                    ui.ctx().copy_text(m.content.clone());
                }
            });
        }
        Role::Assistant => {
            ui.horizontal(|ui| {
                dot(ui, theme::ACCENT);
                let who = if app.chat.model.is_empty() { "Assistant".to_string() } else { app.chat.model.clone() };
                ui.label(RichText::new(who).small().color(theme::MUTED));
            });
            let (think, answer, think_open) = if !m.reasoning.is_empty() {
                (m.reasoning.clone(), m.content.clone(), false)
            } else {
                let (t, a, o) = split_think(&m.content);
                (t, a, o)
            };
            let thinking_now = m.streaming && answer.is_empty() && (!think.is_empty() || think_open);
            if !think.is_empty() {
                let title = if thinking_now {
                    "Thinking ...".to_string()
                } else if m.thought_secs > 0.0 {
                    format!("Thought for {}", human_secs(m.thought_secs as f64))
                } else {
                    "Reasoning".to_string()
                };
                egui::CollapsingHeader::new(RichText::new(title).color(theme::MUTED)).id_salt((m.id.as_str(), thinking_now)).default_open(thinking_now).show(ui, |ui| {
                    Frame::new().fill(theme::CODE_BG).corner_radius(CornerRadius::same(8)).inner_margin(Margin::same(10)).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        let mut style = md_style(13.5);
                        style.text = theme::MUTED;
                        let blocks = markdown::parse(&think);
                        markdown::render(ui, &blocks, &style, id_hash(&m.id) ^ 1, theme::bold_family());
                    });
                });
            }
            if m.streaming && answer.is_empty() && think.is_empty() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new("waiting for the first token ...").small().color(theme::MUTED));
                });
            }
            if !answer.is_empty() {
                let blocks = markdown::parse(&answer);
                markdown::render(ui, &blocks, &md_style(15.0), id_hash(&m.id), theme::bold_family());
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

fn input_box(app: &mut StudioApp, ui: &mut Ui) {
    centered(ui, COLUMN, |ui| {
        Frame::new().fill(theme::CARD).corner_radius(CornerRadius::same(14)).stroke(Stroke::new(1.0, theme::BORDER)).inner_margin(Margin::symmetric(12, 8)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let button_w = 92.0;
                let edit_w = (ui.available_width() - button_w - 12.0).max(120.0);
                let enter_sends = KeyboardShortcut::new(Modifiers::SHIFT, Key::Enter);
                let resp = ScrollArea::vertical()
                    .id_salt("input_scroll")
                    .max_height(170.0)
                    .auto_shrink([true, true])
                    .show(ui, |ui| {
                        ui.add(
                            TextEdit::multiline(&mut app.input)
                                .hint_text("Message the model ...   (Enter sends, Shift+Enter adds a line)")
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
                        let can = !app.input.trim().is_empty() && app.active_endpoint().is_some();
                        if primary_enabled(ui, !app.input.trim().is_empty(), "Send").clicked() || (enter && can) {
                            app.actions.push(Action::Send);
                            app.focus_input = true;
                        } else if enter && !can && !app.input.trim().is_empty() {
                            app.actions.push(Action::Send);
                        }
                    }
                });
            });
        });
    });
}

fn params(app: &mut StudioApp, ui: &mut Ui) {
    heading(ui, "Parameters");
    muted(ui, "Used for the next message.");
    ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
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
        muted(ui, "For Qwen3 and Qwen3.5. The small Qwen3.5 models answer directly unless told to think.");
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
