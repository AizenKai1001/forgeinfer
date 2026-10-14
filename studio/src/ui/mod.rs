//! The window: a rail of screens on the left, a status bar at the bottom, the active screen in the middle.

pub mod chat;
pub mod discover;
pub mod embed;
pub mod library;
pub mod server;
pub mod settings;
pub mod widgets;

use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, FontId, Frame, Layout, Margin, Panel, RichText,
    Sense, Stroke, Ui,
};

use crate::app::{Action, Screen, StudioApp};
use crate::engine::EngineState;
use crate::settings::EngineKind;
use crate::theme;
use crate::util::human_gib;

use widgets::dot;

pub const RAIL_WIDTH: f32 = 188.0;

pub fn draw(app: &mut StudioApp, ui: &mut Ui) {
    Panel::left("rail")
        .exact_size(RAIL_WIDTH)
        .resizable(false)
        .frame(
            Frame::new()
                .fill(theme::RAIL)
                .inner_margin(Margin::same(10))
                .stroke(Stroke::new(1.0, theme::BORDER)),
        )
        .show(ui, |ui| rail(app, ui));
    Panel::bottom("status")
        .exact_size(28.0)
        .resizable(false)
        .frame(
            Frame::new()
                .fill(theme::RAIL)
                .inner_margin(Margin::symmetric(12, 0))
                .stroke(Stroke::new(1.0, theme::BORDER)),
        )
        .show(ui, |ui| status_bar(app, ui));
    egui::CentralPanel::default()
        .frame(Frame::new().fill(theme::PANEL))
        .show(ui, |ui| match app.screen {
            Screen::Chat => chat::show(app, ui),
            Screen::Library => library::show(app, ui),
            Screen::Discover => discover::show(app, ui),
            Screen::Server => server::show_server(app, ui),
            Screen::Hardware => server::show_hardware(app, ui),
            Screen::Logs => server::show_logs(app, ui),
            Screen::Settings => settings::show(app, ui),
        });
    toasts(app, ui.ctx());
}

fn icon(screen: Screen) -> &'static str {
    match screen {
        Screen::Chat => "\u{1F4AC}",
        Screen::Library => "\u{1F4E6}",
        Screen::Discover => "\u{1F50D}",
        Screen::Server => "\u{1F310}",
        Screen::Hardware => "\u{1F5A5}",
        Screen::Logs => "\u{1F4DC}",
        Screen::Settings => "\u{2699}",
    }
}

fn rail(app: &mut StudioApp, ui: &mut Ui) {
    ui.add_space(4.0);
    ui.label(
        RichText::new("Forge Studio")
            .size(21.0)
            .color(theme::STRONG)
            .family(theme::bold_family()),
    );
    ui.label(
        RichText::new("local models, your hardware")
            .small()
            .color(theme::MUTED),
    );
    ui.add_space(14.0);
    for s in Screen::ALL {
        if nav_item(ui, icon(s), s.label(), app.screen == s) {
            app.screen = s;
        }
        ui.add_space(1.0);
    }
    // the engine chip sits at the bottom of the rail
    let busy = matches!(
        app.engine.state(),
        EngineState::Ready { .. } | EngineState::Loading { .. }
    );
    let chip_h = if busy { 104.0 } else { 56.0 };
    ui.add_space((ui.available_height() - chip_h).max(6.0));
    engine_chip(app, ui);
}

fn nav_item(ui: &mut Ui, icon: &str, label: &str, selected: bool) -> bool {
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 34.0), Sense::click());
    let painter = ui.painter_at(rect);
    let fill = if selected {
        theme::CARD_HOVER
    } else if response.hovered() {
        theme::CARD
    } else {
        Color32::TRANSPARENT
    };
    painter.rect_filled(rect, CornerRadius::same(8), fill);
    if selected {
        painter.rect_filled(
            egui::Rect::from_min_size(
                rect.min + egui::vec2(0.0, 8.0),
                egui::vec2(3.0, rect.height() - 16.0),
            ),
            CornerRadius::same(2),
            theme::ACCENT,
        );
    }
    let color = if selected {
        theme::STRONG
    } else {
        theme::MUTED
    };
    painter.text(
        rect.left_center() + egui::vec2(16.0, 0.0),
        Align2::LEFT_CENTER,
        icon,
        FontId::proportional(15.0),
        if selected { theme::ACCENT } else { color },
    );
    painter.text(
        rect.left_center() + egui::vec2(42.0, 0.0),
        Align2::LEFT_CENTER,
        label,
        FontId::proportional(14.5),
        color,
    );
    response
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .clicked()
}

fn engine_chip(app: &mut StudioApp, ui: &mut Ui) {
    let (title, detail, color) = engine_summary(app);
    Frame::new()
        .fill(theme::CARD)
        .corner_radius(CornerRadius::same(8))
        .stroke(Stroke::new(1.0, theme::BORDER))
        .inner_margin(Margin::same(8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.with_layout(Layout::top_down(Align::Min), |ui| {
                ui.horizontal(|ui| {
                    dot(ui, color);
                    ui.label(RichText::new(title).small().color(theme::TEXT));
                });
                if !detail.is_empty() {
                    ui.add(
                        egui::Label::new(RichText::new(detail).small().color(theme::MUTED))
                            .truncate(),
                    );
                }
                if matches!(
                    app.engine.state(),
                    EngineState::Ready { .. } | EngineState::Loading { .. }
                ) && ui.small_button("Eject").clicked()
                {
                    app.actions.push(Action::Eject);
                }
            });
        });
}

/// (headline, detail, colour) of the engine for the chips and the status bar.
pub fn engine_summary(app: &StudioApp) -> (String, String, Color32) {
    if app.settings.engine == EngineKind::External {
        let ok = !app.external_models.is_empty();
        return (
            if ok {
                "Connected".to_string()
            } else {
                "Not connected".to_string()
            },
            if ok {
                app.external_pick.clone()
            } else {
                app.settings.external_url.clone()
            },
            if ok { theme::GOOD } else { theme::MUTED },
        );
    }
    match app.engine.state() {
        EngineState::Idle => ("No model loaded".to_string(), String::new(), theme::MUTED),
        EngineState::Loading { model, since } => (
            format!("Loading ({:.0} s)", since.elapsed().as_secs_f32()),
            model,
            theme::WARN,
        ),
        EngineState::Ready { model, port, .. } => (
            "Ready".to_string(),
            format!("{model}  :{port}"),
            theme::GOOD,
        ),
        EngineState::Failed { model, .. } => ("Failed to load".to_string(), model, theme::BAD),
    }
}

fn status_bar(app: &mut StudioApp, ui: &mut Ui) {
    ui.horizontal_centered(|ui| {
        let (title, detail, color) = engine_summary(app);
        dot(ui, color);
        ui.label(
            RichText::new(if detail.is_empty() {
                title
            } else {
                format!("{title}: {detail}")
            })
            .small()
            .color(theme::TEXT),
        );
        if let EngineState::Failed { message, .. } = app.engine.state() {
            // the reason fits between the state and the figures on the right, whatever its length
            let w = (ui.available_width() - 360.0).max(60.0);
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
        right_to_left(ui, app);
    });
}

fn right_to_left(ui: &mut Ui, app: &StudioApp) {
    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
        if let Some(s) = app.last_speed.filter(|s| *s > 0.0) {
            ui.label(
                RichText::new(format!("{s:.1} tok/s"))
                    .small()
                    .color(theme::ACCENT),
            );
            ui.separator();
        }
        if let Some(g) = app.hw.gpus.first() {
            let used = if g.vram_used > 0 {
                format!("  {} / {}", human_gib(g.vram_used), human_gib(g.vram_total))
            } else {
                format!("  {}", human_gib(g.vram_total))
            };
            // under about 700 MiB free the Windows driver starts to page video memory and every GPU program slows down (a 2.4 times slower engine at 160 MiB free was measured)
            let tight = g.vram_used > 0 && g.vram_total.saturating_sub(g.vram_used) < (700u64 << 20);
            let r = ui.label(
                RichText::new(format!("{}{}", g.name, used))
                    .small()
                    .color(if tight { theme::WARN } else { theme::MUTED }),
            );
            if tight {
                r.on_hover_text("Little video memory is left. The driver may start to page and everything on the GPU slows down. Close other programs, lower the context length or turn on the 8-bit KV cache in Settings.");
            }
        }
    });
}

fn toasts(app: &mut StudioApp, ctx: &egui::Context) {
    if app.toasts.is_empty() {
        return;
    }
    egui::Area::new(egui::Id::new("toasts"))
        .anchor(Align2::RIGHT_BOTTOM, [-16.0, -44.0])
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            for t in &app.toasts {
                Frame::new()
                    .fill(theme::CARD_HOVER)
                    .corner_radius(CornerRadius::same(8))
                    .stroke(Stroke::new(1.0, theme::ACCENT.gamma_multiply(0.6)))
                    .inner_margin(Margin::symmetric(12, 8))
                    .show(ui, |ui| {
                        ui.set_max_width(380.0);
                        ui.add(egui::Label::new(RichText::new(&t.text).color(theme::TEXT)).wrap());
                    });
                ui.add_space(4.0);
            }
        });
}
