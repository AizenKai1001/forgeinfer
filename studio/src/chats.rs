//! Conversations: the data model and one JSON file per chat in the app's data folder.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::util::{new_id, truncate, unix_secs};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn api(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// What a finished answer cost.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Stats {
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    pub tok_per_s: f32,
    pub first_token_s: f32,
    pub total_s: f32,
    pub finish: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub id: String,
    pub role: Role,
    pub content: String,
    /// The model's own reasoning text (thinking models), shown folded.
    #[serde(default)]
    pub reasoning: String,
    #[serde(default)]
    pub stats: Option<Stats>,
    #[serde(default)]
    pub error: Option<String>,
    /// True while the answer is still arriving.
    #[serde(skip)]
    pub streaming: bool,
    /// Seconds spent in the reasoning part (measured while it streamed).
    #[serde(default)]
    pub thought_secs: f32,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Message {
        Message { id: new_id(), role, content: content.into(), reasoning: String::new(), stats: None, error: None, streaming: false, thought_secs: 0.0 }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Chat {
    pub id: String,
    pub title: String,
    pub created: u64,
    pub updated: u64,
    /// The model the last answer came from.
    #[serde(default)]
    pub model: String,
    /// A system prompt of this chat; empty = the one in the settings.
    #[serde(default)]
    pub system: String,
    pub messages: Vec<Message>,
}

pub const NEW_TITLE: &str = "New chat";

impl Chat {
    pub fn new() -> Chat {
        let now = unix_secs();
        Chat { id: new_id(), title: NEW_TITLE.to_string(), created: now, updated: now, model: String::new(), system: String::new(), messages: Vec::new() }
    }

    /// The title comes from the first thing the user wrote.
    pub fn retitle(&mut self) {
        if self.title == NEW_TITLE {
            if let Some(m) = self.messages.iter().find(|m| m.role == Role::User) {
                let one_line: String = m.content.split_whitespace().collect::<Vec<_>>().join(" ");
                if !one_line.is_empty() {
                    self.title = truncate(&one_line, 42);
                }
            }
        }
    }

    /// The conversation as the server wants it: the system prompt, then every finished message (an answer that is empty or failed is left out).
    pub fn api_messages(&self, default_system: &str) -> Vec<(Role, String)> {
        let mut out = Vec::new();
        let system = if self.system.trim().is_empty() { default_system } else { self.system.as_str() };
        if !system.trim().is_empty() {
            out.push((Role::System, system.to_string()));
        }
        for m in &self.messages {
            if m.error.is_some() || (m.role == Role::Assistant && m.content.trim().is_empty()) {
                continue;
            }
            out.push((m.role, m.content.clone()));
        }
        out
    }

    pub fn to_markdown(&self) -> String {
        let mut s = format!("# {}\n\n", self.title);
        for m in &self.messages {
            let who = match m.role {
                Role::User => "You",
                Role::Assistant => "Assistant",
                Role::System => "System",
            };
            s.push_str(&format!("**{who}**\n\n{}\n\n", m.content));
        }
        s
    }
}

impl Default for Chat {
    fn default() -> Self {
        Chat::new()
    }
}

/// One line of the chat list.
#[derive(Clone, Debug)]
pub struct ChatMeta {
    pub id: String,
    pub title: String,
    pub updated: u64,
}

pub struct Store {
    pub dir: PathBuf,
}

impl Store {
    pub fn new(dir: impl Into<PathBuf>) -> Store {
        Store { dir: dir.into() }
    }

    fn path(&self, id: &str) -> PathBuf {
        let safe: String = id.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').collect();
        self.dir.join(format!("{safe}.json"))
    }

    pub fn save(&self, chat: &Chat) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let path = self.path(&chat.id);
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string(chat).unwrap_or_default())?;
        std::fs::rename(&tmp, &path)
    }

    pub fn load(&self, id: &str) -> Option<Chat> {
        let text = std::fs::read_to_string(self.path(id)).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn delete(&self, id: &str) {
        let _ = std::fs::remove_file(self.path(id));
    }

    /// Newest first.
    pub fn list(&self) -> Vec<ChatMeta> {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return out };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else { continue };
            if let Ok(c) = serde_json::from_str::<Chat>(&text) {
                out.push(ChatMeta { id: c.id, title: c.title, updated: c.updated });
            }
        }
        out.sort_by(|a, b| b.updated.cmp(&a.updated));
        out
    }
}

pub fn chats_dir(data: &Path) -> PathBuf {
    data.join("chats")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("forge-studio-test-{name}-{}", new_id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_chat_round_trips_through_its_file() {
        let dir = temp_dir("chats");
        let store = Store::new(&dir);
        let mut c = Chat::new();
        c.messages.push(Message::new(Role::User, "Hello there,\n how are you?"));
        let mut a = Message::new(Role::Assistant, "Fine.");
        a.reasoning = "thinking".into();
        a.stats = Some(Stats { completion_tokens: 3, tok_per_s: 40.5, ..Default::default() });
        c.messages.push(a);
        c.retitle();
        assert_eq!(c.title, "Hello there, how are you?");
        store.save(&c).unwrap();
        let back = store.load(&c.id).unwrap();
        assert_eq!(back.messages.len(), 2);
        assert_eq!(back.messages[1].reasoning, "thinking");
        assert_eq!(back.messages[1].stats.as_ref().unwrap().tok_per_s, 40.5);
        assert_eq!(store.list().len(), 1);
        store.delete(&c.id);
        assert!(store.load(&c.id).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_and_empty_answers_stay_out_of_the_prompt() {
        let mut c = Chat::new();
        c.messages.push(Message::new(Role::User, "a"));
        let mut bad = Message::new(Role::Assistant, "");
        bad.error = Some("boom".into());
        c.messages.push(bad);
        c.messages.push(Message::new(Role::User, "b"));
        c.messages.push(Message::new(Role::Assistant, ""));
        let m = c.api_messages("be brief");
        assert_eq!(m.len(), 3);
        assert_eq!(m[0], (Role::System, "be brief".to_string()));
        assert_eq!(m[2].1, "b");
    }
}
