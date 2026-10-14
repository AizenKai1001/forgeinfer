//! `cargo run --release -p forge-embed --example embed_bench -- MODEL.gguf`: milliseconds per text for texts of 8 to 1,024 tokens, on this CPU.
use forge_embed::Embedder;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("usage: embed_bench MODEL.gguf");
    let t0 = Instant::now();
    let e = Embedder::load(&path).expect("the model loads");
    println!("loaded in {:.2} s ({} threads, {} layers, {} dimensions)", t0.elapsed().as_secs_f64(), std::env::var("FORGE_EMBED_THREADS").unwrap_or_else(|_| "all".into()), e.config().n_layer, e.dim());
    let base = "the harbour town of aldermere keeps careful records of every ship that docks noting the cargo the captain and the weather of the day ";
    for words in [6usize, 24, 96, 384, 768] {
        let text: String = base.split(' ').filter(|w| !w.is_empty()).cycle().take(words).collect::<Vec<_>>().join(" ");
        let ids = e.tokenizer().encode_truncated(&text, e.config().ctx);
        let _ = e.embed_ids(&ids);
        let reps = if words <= 96 { 5 } else { 2 };
        let t = Instant::now();
        for _ in 0..reps {
            let _ = e.embed_ids(&ids);
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("{:4} tokens: {:8.1} ms  ({:6.0} tokens/s)", ids.len(), ms, ids.len() as f64 / (ms / 1e3));
    }
}
