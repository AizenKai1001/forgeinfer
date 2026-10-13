#[path = "../../eval_lambada.rs"]
mod eval_lambada;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(eval_lambada::lambada(&args));
}
