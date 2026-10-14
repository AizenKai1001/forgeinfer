//! Speed of the GPU encoder: cargo run --release -p forge-embed-gpu --example embed_gpu_bench -- MODEL.gguf [GPU]
//! Embeds texts of 16 to 512 tokens in packs of 1 to 128 texts and prints texts and tokens per second (and, for the CPU path of forge-embed, the same on one pack for comparison).

use forge_embed::{Embedder, Encode};
use forge_embed_gpu::GpuEmbedder;
use std::time::Instant;

fn text_of(tokens: usize, salt: usize) -> String {
    // "wordN " is two WordPiece tokens ("word", "##N" or a digit piece) at worst; the exact count is printed from the result
    (0..tokens / 2).map(|i| format!("tok{} ", (i * 7 + salt) % 1000)).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: embed_gpu_bench MODEL.gguf [GPU]");
        return;
    };
    let ordinal: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(0);
    let t0 = Instant::now();
    let cpu = Embedder::load(path).expect("model loads");
    let gpu = GpuEmbedder::new(Embedder::load(path).expect("model loads").into_parts(), ordinal).expect("GPU encoder");
    println!("{} on {} (loaded in {:.1} s)", gpu.name(), gpu.backend(), t0.elapsed().as_secs_f64());
    println!("{:>6} {:>7} {:>9} {:>10} {:>12}", "tokens", "texts", "ms/pack", "texts/s", "tokens/s");
    for &len in &[16usize, 64, 256, 512] {
        for &count in &[1usize, 8, 32, 128] {
            let texts: Vec<String> = (0..count).map(|i| text_of(len, i)).collect();
            let refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
            let warm = gpu.embed_many(&refs).expect("embeds");
            let tokens: usize = warm.iter().map(|e| e.tokens).sum();
            let reps = (4000 / tokens.max(1)).clamp(3, 200);
            let t = Instant::now();
            for _ in 0..reps {
                gpu.embed_many(&refs).expect("embeds");
            }
            let secs = t.elapsed().as_secs_f64() / reps as f64;
            println!("{:>6} {:>7} {:>9.2} {:>10.0} {:>12.0}", tokens / count, count, secs * 1e3, count as f64 / secs, tokens as f64 / secs);
        }
    }
    // the CPU path on one pack, for scale
    let texts: Vec<String> = (0..32).map(|i| text_of(64, i)).collect();
    let refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
    cpu.embed_many(&refs).expect("embeds");
    let t = Instant::now();
    let r = cpu.embed_many(&refs).expect("embeds");
    let secs = t.elapsed().as_secs_f64();
    let tokens: usize = r.iter().map(|e| e.tokens).sum();
    println!("CPU ({}): 32 texts of ~{} tokens: {:.0} ms, {:.0} texts/s, {:.0} tokens/s", cpu.backend(), tokens / 32, secs * 1e3, 32.0 / secs, tokens as f64 / secs);
}
