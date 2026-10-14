//! `forge-studio`: the desktop app. A plain window on Windows (no console) in release builds.
//!
//! Options (all optional; they exist for testing and screenshots):
//!   --screen NAME      open on a screen: chat, library, discover, server, hardware, logs, settings
//!   --demo             start with a sample conversation        --open-last  open the newest saved chat
//!   --load PATH        load this GGUF file at start          --select NAME   load the library model whose name contains NAME
//!   --send TEXT        send a message as soon as a model is ready (repeatable: the next one goes when the answer to the one before is complete)
//!   --search TEXT      run a Hugging Face search at start     --inspect PATH  open the inspector on a GGUF file
//!   --shot PATH        save a screenshot as PNG and quit       --shot-delay S  wait S seconds first       --shot-after-reply
//!   --stop-after S     press Stop S seconds after a reply starts (tests)
//!   --params           open the parameters panel (screenshots)
//!   --open-passages    show the passages the model read open under questions about searched files (screenshots)
//!   --embed            with an embedding model: embed the sample lines of the playground once it is ready (tests and screenshots)
//!   --attach FILE      attach a text file to the first message (repeatable)
//!   --edit-msg N       open the editor on message N        --rename-first  start renaming the newest chat of the list (screenshots)
//!   --size WxH         window size in points

#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use eframe::egui;
use forge_studio::app::{Screen, StartupOpts, StudioApp};

fn icon() -> egui::IconData {
    const N: usize = 64;
    let mut rgba = vec![0u8; N * N * 4];
    for y in 0..N {
        for x in 0..N {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let (cx, cy) = (fx.clamp(15.0, 49.0), fy.clamp(15.0, 49.0));
            let inside = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt() <= 12.0
                && fx > 3.0
                && fx < 61.0
                && fy > 3.0
                && fy < 61.0;
            let letter = ((19..28).contains(&x) && (14..50).contains(&y))
                || ((19..46).contains(&x) && (14..23).contains(&y))
                || ((19..40).contains(&x) && (28..37).contains(&y));
            let px = if !inside {
                [0, 0, 0, 0]
            } else if letter {
                [34, 18, 6, 255]
            } else {
                [255, 143, 64, 255]
            };
            rgba[(y * N + x) * 4..(y * N + x) * 4 + 4].copy_from_slice(&px);
        }
    }
    egui::IconData {
        rgba,
        width: N as u32,
        height: N as u32,
    }
}

fn parse_args() -> Result<StartupOpts, String> {
    let mut o = StartupOpts::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "--screen" => {
                let v = value("--screen")?;
                o.screen =
                    Some(Screen::from_name(&v).ok_or_else(|| format!("unknown screen {v:?}"))?);
            }
            "--demo" => o.demo = true,
            "--open-last" => o.open_last = true,
            "--edit-msg" => {
                o.edit_msg = Some(
                    value("--edit-msg")?
                        .parse()
                        .map_err(|_| "--edit-msg needs a number".to_string())?,
                )
            }
            "--rename-first" => o.rename_first = true,
            "--params" => o.show_params = true,
            "--embed" => o.embed = true,
            "--open-passages" => o.open_passages = true,
            "--attach" => o.attach.push(value("--attach")?.into()),
            "--load" => o.load = Some(value("--load")?.into()),
            "--select" => o.select_model = Some(value("--select")?),
            "--send" => o.send.push(value("--send")?),
            "--search" => o.hf_query = Some(value("--search")?),
            "--inspect" => o.inspect = Some(value("--inspect")?.into()),
            "--stop-after" => {
                o.stop_after = Some(
                    value("--stop-after")?
                        .parse()
                        .map_err(|_| "--stop-after needs a number".to_string())?,
                )
            }
            "--shot" => o.shot = Some(value("--shot")?.into()),
            "--shot-delay" => {
                o.shot_delay = value("--shot-delay")?
                    .parse()
                    .map_err(|_| "--shot-delay needs a number".to_string())?
            }
            "--shot-after-reply" => o.shot_after_reply = true,
            "--size" => {
                let v = value("--size")?;
                let (w, h) = v.split_once('x').ok_or("--size needs WxH")?;
                o.size = Some((
                    w.parse().map_err(|_| "bad width")?,
                    h.parse().map_err(|_| "bad height")?,
                ));
            }
            "--help" | "-h" => return Err(String::new()),
            other => return Err(format!("unknown option {other:?}")),
        }
    }
    Ok(o)
}

fn main() -> eframe::Result {
    let opts = match parse_args() {
        Ok(o) => o,
        Err(m) => {
            if !m.is_empty() {
                eprintln!("forge-studio: {m}");
            }
            eprintln!("usage: forge-studio [--screen NAME] [--demo] [--load PATH] [--select NAME] [--send TEXT] [--embed] [--search TEXT] [--inspect PATH] [--shot PATH [--shot-delay S] [--shot-after-reply]] [--size WxH]");
            std::process::exit(if m.is_empty() { 0 } else { 2 });
        }
    };
    let (w, h) = opts.size.unwrap_or((1280.0, 800.0));
    let viewport = egui::ViewportBuilder::default()
        .with_title("Forge Studio")
        .with_inner_size([w, h])
        .with_min_inner_size([900.0, 560.0])
        .with_icon(icon());
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "Forge Studio",
        options,
        Box::new(move |cc| Ok(Box::new(StudioApp::new(cc, opts.clone())))),
    )
}
