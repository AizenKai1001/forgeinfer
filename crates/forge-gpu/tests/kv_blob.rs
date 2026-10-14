//! The KV blob of the prompt cache: the KV entries of a conversation (and a hybrid model's recurrent state) copied to host memory, the cache used for another prompt, and the blob and state copied back must
//! continue the conversation exactly as if nothing had happened: the greedy tokens that follow are the same as without the detour, for a dense model and a hybrid one, with the plain fp16 cache and the
//! compressed one (FORGE_KV=int8), and with fewer positions imported than were exported (a dense model resumes from a shared prefix).
//!
//! Dense model: FORGE_QWEN2_GGUF (any dense GGUF; default the Qwen2.5-0.5B of the other tests), hybrid model: FORGE_VARIANTS_MODEL (Qwen3.5-0.8B). A test is skipped when its model or a CUDA device is missing.

use forge_cuda::{Device, PinnedBuffer};
use forge_gpu::{Fmt, GpuModel, GpuSession};
use forge_model::Model;
use std::path::PathBuf;

mod common;

fn models_dir() -> PathBuf {
    match std::env::var("FORGE_MODELS") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("forgeinfer/oracle/models"),
    }
}

fn dense_path() -> PathBuf {
    match std::env::var("FORGE_QWEN2_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => models_dir().join("Qwen__Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    }
}

fn hybrid_path() -> PathBuf {
    match std::env::var("FORGE_VARIANTS_MODEL") {
        Ok(p) => PathBuf::from(p),
        Err(_) => models_dir().join("unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"),
    }
}

/// Token ids of natural text: the chat prompts of the Qwen2 / Qwen3 tests (dense models) or of the Qwen3.5 golden file (hybrid ones), repeated to `n` tokens and rotated by `shift`
fn prompt(n: usize, shift: usize, hybrid: bool) -> Vec<u32> {
    let base: Vec<u32> = if hybrid {
        let gold = forge_core::Json::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/fwd_qwen35.json")).unwrap();
        gold["prompts"].as_array().unwrap().iter().flat_map(|p| p["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect::<Vec<_>>()).collect()
    } else {
        vec![
            151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 7985, 264, 6386, 38242, 911, 279, 9396, 13, 151645, 198,
            151644, 77091, 198, 785, 9396, 374, 6884, 323, 6783, 11, 151645, 198, 151644, 872, 198, 9330, 432, 438, 264, 2805, 31365, 13, 151645, 198,
        ]
    };
    (0..n).map(|i| base[(i + shift) % base.len()]).collect()
}

/// Read tokens the way the engine does: the batched path for a piece of enough tokens, the single-token path below that.
fn feed(sess: &mut GpuSession, t: &[u32]) {
    if t.len() >= sess.prefill_batched_min() {
        sess.prefill_batched(t).unwrap();
    } else {
        sess.prefill(t).unwrap();
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    ab / (aa.sqrt() * bb.sqrt())
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0
}

/// The conversation of `n` tokens, read in two parts (all but the last token, then the last one: how the engine resumes); the logits that start the answer. The detour: the first part is read, copied to host
/// memory (`export_kv`, with a snapshot for a hybrid model), another prompt overwrites the session, the blob is copied back (`adopt`, the first `use_n` positions) and the rest is read. The logits must be those
/// of the straight run: bit for bit when the straight run is reproducible (it is read twice to know), else close.
fn run(path: &PathBuf, label: &str, n: usize, use_fewer: bool, hybrid: bool) {
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped ({label}): {} or a CUDA device is missing", path.display());
        return;
    }
    let model = Model::load_native(path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    let mut sess = GpuSession::new(&gpu, 4096).unwrap();
    assert_eq!(hybrid, !sess.can_truncate(), "{label}: the model is not of the kind the test expects");
    let ids = prompt(n, 0, hybrid);
    let other = prompt(900, 17, hybrid);

    let mut straight = || {
        sess.reset().unwrap();
        feed(&mut sess, &ids[..n - 1]);
        feed(&mut sess, &ids[n - 1..]);
        sess.logits().unwrap()
    };
    let want = straight();
    let again = straight();
    let reproducible = want == again;

    // with the detour
    sess.reset().unwrap();
    feed(&mut sess, &ids[..n - 1]);
    let state = if hybrid { Some(sess.snapshot_state().unwrap()) } else { None };
    let blob = sess.export_kv(n - 1).unwrap();
    assert_eq!(blob.positions(), n - 1);
    assert_eq!(blob.bytes(), sess.kv_blob_bytes(n - 1));
    sess.reset().unwrap();
    feed(&mut sess, &other);
    let _ = sess.decode_greedy(8).unwrap(); // (the other conversation goes on: the entries it wrote beyond its prompt are not ours either)
    // a dense model may import fewer positions than the blob has and read the rest again; a hybrid model needs the position of its state
    let use_n = if use_fewer && !hybrid { (n - 1) * 2 / 3 } else { n - 1 };
    sess.adopt(&blob, use_n, state.as_ref()).unwrap();
    assert_eq!(sess.position(), use_n);
    if use_n < n - 1 {
        feed(&mut sess, &ids[use_n..n - 1]);
    }
    feed(&mut sess, &ids[n - 1..]);
    let got = sess.logits().unwrap();
    // the tokens that follow: the same after the detour as after another conversation's decoding and a straight read (the generation state is not left over from the conversation that was on the GPU)
    let got_tokens = sess.decode_greedy(12).unwrap();
    sess.reset().unwrap();
    feed(&mut sess, &other);
    let _ = sess.decode_greedy(8).unwrap();
    sess.reset().unwrap();
    feed(&mut sess, &ids[..n - 1]);
    feed(&mut sess, &ids[n - 1..]);
    let want_tokens = sess.decode_greedy(12).unwrap();
    // (`decode_greedy` returns the device's output ring from its first slot: after a straight read the slot 0 holds the token the batched part of the prompt predicted, which a resumed conversation that reads
    // only its last token does not have: the tokens are compared from the one the last token predicted)
    if use_n == n - 1 {
        assert_eq!(&got_tokens[..11], &want_tokens[1..], "{label}: the tokens that follow differ after the round trip");
    } else {
        assert_eq!(got_tokens, want_tokens, "{label}: the tokens that follow differ after the round trip");
    }
    let cos = cosine(&want, &got);
    let maxdiff = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("{label}: {} positions in {:.1} MiB; straight run reproducible: {reproducible}; logits cosine {cos:.8}, max |difference| {maxdiff:.2e}, top-1 {} / {}", n - 1, blob.bytes() as f64 / 1048576.0, argmax(&want), argmax(&got));
    if reproducible && use_n == n - 1 {
        assert!(want == got, "{label}: the logits differ after the round trip through host memory (cosine {cos}, max difference {maxdiff})");
    } else {
        assert!(cos > 0.99999 && argmax(&want) == argmax(&got), "{label}: the logits of the conversation changed (cosine {cos}, top-1 {} / {})", argmax(&want), argmax(&got));
    }
}

/// The buffer of a blob that has come back (`into_buffer`) takes the next conversation that is stored (`export_kv_with`: locking 1 GB of host memory is slow, the prompt cache keeps the buffer): the blob in it
/// must carry its conversation exactly as a fresh buffer would, also when the buffer is bigger than the conversation, and a buffer that is too small is replaced.
fn reuse_run(path: &PathBuf, label: &str, hybrid: bool) {
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped ({label}): {} or a CUDA device is missing", path.display());
        return;
    }
    let model = Model::load_native(path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    let mut sess = GpuSession::new(&gpu, 4096).unwrap();
    assert_eq!(hybrid, !sess.can_truncate(), "{label}: the model is not of the kind the test expects");
    let (a, b) = (prompt(1500, 0, hybrid), prompt(1100, 29, hybrid));

    // the conversation b read straight (twice: bit for bit is only asked where the straight run is reproducible)
    let mut straight = || {
        sess.reset().unwrap();
        feed(&mut sess, &b[..1099]);
        feed(&mut sess, &b[1099..]);
        sess.logits().unwrap()
    };
    let want = straight();
    let reproducible = want == straight();

    // a goes out, comes back, and its buffer is kept
    sess.reset().unwrap();
    feed(&mut sess, &a[..1499]);
    let state_a = if hybrid { Some(sess.snapshot_state().unwrap()) } else { None };
    let blob_a = sess.export_kv(1499).unwrap();
    let capacity = blob_a.capacity();
    assert_eq!(capacity, blob_a.bytes());
    sess.reset().unwrap();
    sess.adopt(&blob_a, 1499, state_a.as_ref()).unwrap();
    let spare = blob_a.into_buffer();
    assert_eq!(spare.len(), capacity);

    // b is stored into it (fewer positions than the buffer holds)
    sess.reset().unwrap();
    feed(&mut sess, &b[..1099]);
    let state_b = if hybrid { Some(sess.snapshot_state().unwrap()) } else { None };
    let blob_b = sess.export_kv_with(1099, Some(spare)).unwrap();
    assert_eq!(blob_b.capacity(), capacity, "{label}: the buffer was not reused");
    assert_eq!(blob_b.bytes(), sess.kv_blob_bytes(1099));
    assert!(blob_b.bytes() < capacity);

    // the cache is used for a again, then b comes back from the reused buffer
    sess.reset().unwrap();
    feed(&mut sess, &a[..1499]);
    sess.adopt(&blob_b, 1099, state_b.as_ref()).unwrap();
    feed(&mut sess, &b[1099..]);
    let got = sess.logits().unwrap();
    if reproducible {
        assert!(want == got, "{label}: the conversation that was stored in a reused buffer came back different");
    } else {
        assert!(cosine(&want, &got) > 0.99999 && argmax(&want) == argmax(&got), "{label}: the conversation that was stored in a reused buffer came back different");
    }

    // a buffer that is too small is replaced by one that fits
    let blob_c = sess.export_kv_with(1099, Some(PinnedBuffer::alloc(16).unwrap())).unwrap();
    assert_eq!(blob_c.capacity(), blob_c.bytes());
    assert!(blob_c.capacity() > 16);
    eprintln!("{label}: a buffer of {:.1} MiB took a conversation of {:.1} MiB; the logits are the same", capacity as f64 / 1048576.0, blob_b.bytes() as f64 / 1048576.0);
}

#[test]
fn a_blob_buffer_is_reused_for_the_next_conversation_dense() {
    let _g = common::gpu_lock();
    reuse_run(&dense_path(), "dense, buffer reuse", false);
}

#[test]
fn a_blob_buffer_is_reused_for_the_next_conversation_hybrid() {
    let _g = common::gpu_lock();
    reuse_run(&hybrid_path(), "hybrid, buffer reuse", true);
}

#[test]
fn a_dense_model_continues_after_a_round_trip_through_host_memory() {
    let _g = common::gpu_lock();
    run(&dense_path(), "dense, fp16 cache", 1500, false, false);
}

#[test]
fn a_dense_model_resumes_from_part_of_a_blob() {
    let _g = common::gpu_lock();
    run(&dense_path(), "dense, two thirds of the blob", 1500, true, false);
}

#[test]
fn a_hybrid_model_continues_after_a_round_trip_through_host_memory() {
    let _g = common::gpu_lock();
    run(&hybrid_path(), "hybrid, fp16 cache", 1500, false, true);
}

#[test]
fn the_compressed_cache_makes_the_same_round_trip() {
    let _g = common::gpu_lock();
    std::env::set_var("FORGE_KV", "int8");
    run(&dense_path(), "dense, int8 cache", 1500, false, false);
    run(&hybrid_path(), "hybrid, int8 cache", 1500, false, true);
    std::env::remove_var("FORGE_KV");
}
