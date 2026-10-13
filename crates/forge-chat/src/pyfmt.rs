//! Python-compatible text helpers: the chat templates are Jinja templates that call `strip()`, `tojson` (json.dumps with `ensure_ascii=False`) and print numbers with `str()`; the rendered prompt must
//! match byte for byte, so these reproduce Python's behaviour, not Rust's.

use forge_core::Json;

/// Python's `str.isspace` per character: Unicode White_Space plus U+001C..U+001F.
pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

pub fn py_lstrip(s: &str) -> &str {
    s.trim_start_matches(is_py_space)
}

pub fn py_rstrip(s: &str) -> &str {
    s.trim_end_matches(is_py_space)
}

/// `repr(float)`: the shortest round-trip digits, fixed notation for decimal exponents -4..15, otherwise scientific with at least two exponent digits ("1e-05", "1.5e+16").
pub fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let sign = if f < 0.0 { "-" } else { "" };
    let sci = format!("{:e}", f.abs()); // "d.ddde-7" or "de5"
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
        let body = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if (point as usize) >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{body}")
    } else {
        let m = if digits.len() == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        format!(
            "{sign}{m}e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
    }
}

fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `json.dumps(v, ensure_ascii=False)`: separators ", " and ": ", key order kept, non-ASCII kept.
pub fn py_json(v: &Json) -> String {
    let mut out = String::new();
    write_py_json(v, &mut out);
    out
}

fn write_py_json(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Int(i) => out.push_str(&i.to_string()),
        Json::Float(f) => {
            if f.is_nan() {
                out.push_str("NaN");
            } else if f.is_infinite() {
                out.push_str(if *f > 0.0 { "Infinity" } else { "-Infinity" });
            } else {
                out.push_str(&py_float(*f));
            }
        }
        Json::Str(s) => write_json_string(s, out),
        Json::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(x, out);
            }
            out.push(']');
        }
        Json::Obj(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json_string(k, out);
                out.push_str(": ");
                write_py_json(x, out);
            }
            out.push('}');
        }
    }
}

/// `str(v)` of a JSON scalar as Python prints it (True / False / None, floats by repr); containers are printed with `py_json`, which is what the templates do for them.
pub fn py_str(v: &Json) -> String {
    match v {
        Json::Null => "None".into(),
        Json::Bool(true) => "True".into(),
        Json::Bool(false) => "False".into(),
        Json::Int(i) => i.to_string(),
        Json::Float(f) => py_float(*f),
        Json::Str(s) => s.clone(),
        other => py_json(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_like_python() {
        for (f, want) in [
            (1.0, "1.0"),
            (0.5, "0.5"),
            (1.5, "1.5"),
            (100.0, "100.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (-2.25, "-2.25"),
            (123456.789, "123456.789"),
            (0.1, "0.1"),
        ] {
            assert_eq!(py_float(f), want, "{f}");
        }
    }

    #[test]
    fn json_dumps_matches_python() {
        let v = Json::parse(
            r#"{"a": [1, 2.5, "x\ny\u00e9", null, true], "b": {"c": "\"q\""}, "d": "\u0001"}"#,
        )
        .unwrap();
        assert_eq!(py_json(&v), "{\"a\": [1, 2.5, \"x\\ny\u{e9}\", null, true], \"b\": {\"c\": \"\\\"q\\\"\"}, \"d\": \"\\u0001\"}");
    }

    #[test]
    fn strip_matches_python_whitespace() {
        assert_eq!(py_strip("\u{1c} a b \u{a0}\n"), "a b");
        assert_eq!(py_rstrip("x \u{3000}"), "x");
        assert_eq!(py_lstrip("\u{2003}x"), "x");
    }
}
