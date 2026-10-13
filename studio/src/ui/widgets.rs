//! Small drawing helpers shared by the screens.

use eframe::egui::{
    self, Align, Color32, CornerRadius, Frame, Layout, Margin, Response, RichText, Sense, Stroke,
    Ui,
};

use crate::theme;

/// A filled accent button for the main action of a screen.
pub fn primary_button(ui: &mut Ui, text: &str) -> Response {
    ui.add(
        egui::Button::new(RichText::new(text).color(theme::ON_ACCENT).strong())
            .fill(theme::ACCENT)
            .min_size(egui::vec2(0.0, 28.0)),
    )
}

pub fn primary_enabled(ui: &mut Ui, enabled: bool, text: &str) -> Response {
    ui.add_enabled(
        enabled,
        egui::Button::new(RichText::new(text).color(theme::ON_ACCENT).strong())
            .fill(theme::ACCENT)
            .min_size(egui::vec2(0.0, 28.0)),
    )
}

pub fn danger_button(ui: &mut Ui, text: &str) -> Response {
    ui.add(
        egui::Button::new(RichText::new(text).color(Color32::WHITE))
            .fill(Color32::from_rgb(160, 54, 62))
            .min_size(egui::vec2(0.0, 28.0)),
    )
}

/// A small rounded tag.
pub fn badge(ui: &mut Ui, text: &str, fg: Color32, bg: Color32) -> Response {
    Frame::new()
        .fill(bg)
        .corner_radius(CornerRadius::same(5))
        .inner_margin(Margin::symmetric(7, 1))
        .stroke(Stroke::new(1.0, fg.gamma_multiply(0.45)))
        .show(ui, |ui| {
            ui.label(RichText::new(text).small().color(fg));
        })
        .response
}

pub fn neutral_badge(ui: &mut Ui, text: &str) -> Response {
    badge(ui, text, theme::MUTED, theme::CARD)
}

/// A filled, rounded block spanning the available width.
pub fn card<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    Frame::new()
        .fill(theme::CARD)
        .corner_radius(CornerRadius::same(10))
        .stroke(Stroke::new(1.0, theme::BORDER))
        .inner_margin(Margin::same(12))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui)
        })
        .inner
}

pub fn muted(ui: &mut Ui, text: &str) {
    ui.add(egui::Label::new(RichText::new(text).small().color(theme::MUTED)).wrap());
}

pub fn heading(ui: &mut Ui, text: &str) {
    ui.label(
        RichText::new(text)
            .size(22.0)
            .color(theme::STRONG)
            .family(theme::bold_family()),
    );
}

pub fn section(ui: &mut Ui, text: &str) {
    ui.add_space(8.0);
    ui.label(
        RichText::new(text)
            .size(16.0)
            .color(theme::STRONG)
            .family(theme::bold_family()),
    );
    ui.add_space(2.0);
}

pub fn dot(ui: &mut Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), Sense::hover());
    ui.painter().circle_filled(rect.center(), 4.0, color);
}

pub fn banner(ui: &mut Ui, text: &str, fg: Color32, bg: Color32) {
    Frame::new()
        .fill(bg)
        .corner_radius(CornerRadius::same(8))
        .stroke(Stroke::new(1.0, fg.gamma_multiply(0.5)))
        .inner_margin(Margin::symmetric(10, 7))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.add(egui::Label::new(RichText::new(text).color(fg)).wrap());
        });
}

/// A thin horizontal progress bar with a label on the right.
pub fn bar(ui: &mut Ui, fraction: f32, color: Color32, height: f32) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), height), Sense::hover());
    ui.painter()
        .rect_filled(rect, CornerRadius::same(3), theme::BG);
    let mut fill = rect;
    fill.set_width(rect.width() * fraction.clamp(0.0, 1.0));
    ui.painter().rect_filled(fill, CornerRadius::same(3), color);
}

/// Label and value in two columns, for the info screens.
pub fn kv(ui: &mut Ui, key: &str, value: impl Into<String>) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(150.0, 20.0), Sense::hover());
        ui.painter().text(
            rect.left_center(),
            egui::Align2::LEFT_CENTER,
            key,
            egui::FontId::proportional(14.5),
            theme::MUTED,
        );
        ui.add(egui::Label::new(RichText::new(value.into()).color(theme::TEXT)).wrap());
    });
}

pub fn right_aligned<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.with_layout(Layout::right_to_left(Align::Center), add)
        .inner
}
