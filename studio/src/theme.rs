//! The look: a dark graphite theme with one ember-orange accent, and system fonts when they exist (a real bold needs one).

use eframe::egui::{
    self, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Stroke, TextStyle,
    Theme, Visuals,
};
use std::sync::Arc;

pub const BG: Color32 = Color32::from_rgb(16, 17, 20);
pub const PANEL: Color32 = Color32::from_rgb(23, 24, 28);
pub const RAIL: Color32 = Color32::from_rgb(13, 14, 17);
pub const CARD: Color32 = Color32::from_rgb(31, 33, 38);
pub const CARD_HOVER: Color32 = Color32::from_rgb(40, 43, 50);
pub const BORDER: Color32 = Color32::from_rgb(46, 49, 57);
pub const USER_BUBBLE: Color32 = Color32::from_rgb(38, 41, 49);
pub const CODE_BG: Color32 = Color32::from_rgb(14, 15, 18);

pub const TEXT: Color32 = Color32::from_rgb(228, 230, 236);
pub const STRONG: Color32 = Color32::from_rgb(250, 250, 252);
pub const MUTED: Color32 = Color32::from_rgb(140, 146, 160);
pub const ACCENT: Color32 = Color32::from_rgb(255, 143, 64);
pub const ACCENT_HOVER: Color32 = Color32::from_rgb(255, 167, 104);
pub const ACCENT_DIM: Color32 = Color32::from_rgb(70, 44, 26);
pub const ON_ACCENT: Color32 = Color32::from_rgb(34, 18, 6);
pub const GOOD: Color32 = Color32::from_rgb(96, 208, 140);
pub const WARN: Color32 = Color32::from_rgb(240, 184, 72);
pub const BAD: Color32 = Color32::from_rgb(244, 124, 124);
pub const BAD_BG: Color32 = Color32::from_rgb(60, 28, 32);
pub const INFO: Color32 = Color32::from_rgb(110, 190, 245);

/// The family used for bold text: a real bold font when one was found, else the normal one.
pub fn bold_family() -> FontFamily {
    FontFamily::Name("bold".into())
}

pub fn apply(ctx: &egui::Context) {
    ctx.set_theme(Theme::Dark);
    ctx.all_styles_mut(|style| {
        let mut v = Visuals::dark();
        v.panel_fill = PANEL;
        v.window_fill = CARD;
        v.extreme_bg_color = BG;
        v.faint_bg_color = Color32::from_rgb(27, 29, 33);
        v.window_stroke = Stroke::new(1.0, BORDER);
        v.window_corner_radius = CornerRadius::same(12);
        v.menu_corner_radius = CornerRadius::same(8);
        v.hyperlink_color = ACCENT;
        v.selection.bg_fill = ACCENT_DIM;
        v.selection.stroke = Stroke::new(1.0, ACCENT);
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
        v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
        v.widgets.inactive.bg_fill = CARD_HOVER;
        v.widgets.inactive.weak_bg_fill = CARD_HOVER;
        v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
        v.widgets.inactive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(82, 88, 102));
        v.widgets.hovered.weak_bg_fill = Color32::from_rgb(52, 56, 66);
        v.widgets.hovered.bg_fill = Color32::from_rgb(52, 56, 66);
        v.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
        v.widgets.active.weak_bg_fill = Color32::from_rgb(64, 69, 82);
        v.widgets.active.bg_fill = Color32::from_rgb(64, 69, 82);
        v.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
        for w in [
            &mut v.widgets.inactive,
            &mut v.widgets.hovered,
            &mut v.widgets.active,
            &mut v.widgets.noninteractive,
        ] {
            w.corner_radius = CornerRadius::same(7);
        }
        style.visuals = v;
        style
            .text_styles
            .insert(TextStyle::Heading, FontId::proportional(22.0));
        style
            .text_styles
            .insert(TextStyle::Body, FontId::proportional(14.5));
        style
            .text_styles
            .insert(TextStyle::Button, FontId::proportional(14.5));
        style
            .text_styles
            .insert(TextStyle::Small, FontId::proportional(12.5));
        style
            .text_styles
            .insert(TextStyle::Monospace, FontId::monospace(13.0));
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(11.0, 5.0);
        style.spacing.interact_size.y = 27.0;
        style.spacing.scroll = egui::style::ScrollStyle::solid();
    });
}

fn first_existing(paths: &[&str]) -> Option<Vec<u8>> {
    paths.iter().find_map(|p| std::fs::read(p).ok())
}

/// Use the system's UI font and its bold when they can be found; the built-in fonts stay behind them for symbols and emoji.
pub fn install_fonts(ctx: &egui::Context) {
    let (regular, bold) = if cfg!(windows) {
        (
            first_existing(&["C:\\Windows\\Fonts\\segoeui.ttf"]),
            first_existing(&["C:\\Windows\\Fonts\\segoeuib.ttf"]),
        )
    } else if cfg!(target_os = "macos") {
        (
            first_existing(&[
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
            ]),
            first_existing(&[
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
            ]),
        )
    } else {
        (
            first_existing(&[
                "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            ]),
            first_existing(&[
                "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]),
        )
    };
    let mut fonts = FontDefinitions::default();
    if let Some(bytes) = regular {
        fonts.font_data.insert(
            "system-regular".to_string(),
            Arc::new(FontData::from_owned(bytes)),
        );
        fonts
            .families
            .entry(FontFamily::Proportional)
            .or_default()
            .insert(0, "system-regular".to_string());
    }
    let mut bold_list = fonts
        .families
        .get(&FontFamily::Proportional)
        .cloned()
        .unwrap_or_default();
    if let Some(bytes) = bold {
        fonts.font_data.insert(
            "system-bold".to_string(),
            Arc::new(FontData::from_owned(bytes)),
        );
        bold_list.insert(0, "system-bold".to_string());
    }
    fonts.families.insert(bold_family(), bold_list);
    ctx.set_fonts(fonts);
}
