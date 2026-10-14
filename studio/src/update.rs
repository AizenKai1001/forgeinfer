//! "Check for updates": the newest release of the project on GitHub, compared with this version. Only when the button in Settings is pressed: Forge Studio never looks on its own.

use std::time::Duration;

use serde_json::Value;

/// A published release.
#[derive(Clone, Debug, PartialEq)]
pub struct Release {
    pub tag: String,
    pub url: String,
    pub prerelease: bool,
    /// "2026-10-14" (the day it was published).
    pub published: String,
}

/// The newest release that is not a draft in the answer of `GET /repos/OWNER/REPO/releases` (newest first); None when there is none.
pub fn parse_releases(body: &str) -> Result<Option<Release>, String> {
    let v: Value = serde_json::from_str(body).map_err(|e| format!("not a list of releases: {e}"))?;
    let list = v.as_array().ok_or("not a list of releases")?;
    Ok(list.iter().find(|r| !r["draft"].as_bool().unwrap_or(false) && r["tag_name"].as_str().is_some()).map(|r| Release {
        tag: r["tag_name"].as_str().unwrap_or_default().to_string(),
        url: r["html_url"].as_str().unwrap_or("https://github.com/AizenKai1001/forgeinfer/releases").to_string(),
        prerelease: r["prerelease"].as_bool().unwrap_or(false),
        published: r["published_at"].as_str().unwrap_or_default().chars().take(10).collect(),
    }))
}

/// Is `tag` ("v0.1.12") a later version than `current` ("0.1.11")? The numbers are compared one by one.
pub fn is_newer(current: &str, tag: &str) -> bool {
    let numbers = |s: &str| -> Vec<u64> { s.trim_start_matches('v').split(|c: char| !c.is_ascii_digit()).filter(|p| !p.is_empty()).map(|p| p.parse().unwrap_or(0)).collect() };
    numbers(tag) > numbers(current)
}

/// Ask GitHub for the newest release (one request, 60 an hour are allowed without a login).
pub fn check() -> Result<Option<Release>, String> {
    let agent = ureq::Agent::new_with_config(ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(15))).build());
    let mut resp = agent
        .get("https://api.github.com/repos/AizenKai1001/forgeinfer/releases?per_page=5")
        .header("User-Agent", "forge-studio")
        .header("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| format!("could not reach github.com: {e}"))?;
    let body = resp.body_mut().read_to_string().map_err(|e| e.to_string())?;
    parse_releases(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_are_compared_number_by_number() {
        assert!(is_newer("0.1.11", "v0.1.12"));
        assert!(is_newer("0.1.9", "v0.1.10"));
        assert!(is_newer("0.1.12", "v0.2.0"));
        assert!(is_newer("0.9.9", "1.0"));
        assert!(!is_newer("0.1.12", "v0.1.12"));
        assert!(!is_newer("0.1.12", "v0.1.9"));
        assert!(!is_newer("0.2.0", "v0.1.30"));
        assert!(!is_newer("0.1.12", "nightly"));
    }

    /// Asks github.com: cargo test --lib live_ -- --ignored --nocapture
    #[test]
    #[ignore]
    fn live_check() {
        let r = check().unwrap().expect("at least one release");
        println!("{r:?}");
        assert!(r.tag.starts_with('v') && r.url.starts_with("https://github.com/"));
    }

    #[test]
    fn the_newest_release_that_is_not_a_draft_is_taken() {
        let body = r#"[{"tag_name":"v0.1.13","draft":true,"html_url":"https://x/13"},{"tag_name":"v0.1.12","draft":false,"prerelease":true,"html_url":"https://github.com/o/r/releases/tag/v0.1.12","published_at":"2026-10-14T22:45:03Z"},{"tag_name":"v0.1.11"}]"#;
        let r = parse_releases(body).unwrap().unwrap();
        assert_eq!((r.tag.as_str(), r.url.as_str(), r.prerelease, r.published.as_str()), ("v0.1.12", "https://github.com/o/r/releases/tag/v0.1.12", true, "2026-10-14"));
        assert_eq!(parse_releases("[]").unwrap(), None);
        assert!(parse_releases("{\"message\":\"rate limit\"}").is_err());
        assert!(parse_releases("not json").is_err());
    }
}
