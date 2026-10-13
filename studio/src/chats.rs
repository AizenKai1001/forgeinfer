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

/// A text file attached to a message: its text goes to the model in front of the message.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Attachment {
    pub name: String,
    pub text: String,
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
    /// Files attached to a user message.
    #[serde(default)]
    pub attachments: Vec<Attachment>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Message {
        Message {
            id: new_id(),
            role,
            content: content.into(),
            reasoning: String::new(),
            stats: None,
            error: None,
            streaming: false,
            thought_secs: 0.0,
            attachments: Vec::new(),
        }
    }

    /// What the model reads: the attached files (each between `<file name=...>` tags), then the text.
    pub fn prompt_text(&self) -> String {
        if self.attachments.is_empty() {
            return self.content.clone();
        }
        let mut s = String::new();
        for a in &self.attachments {
            s.push_str(&format!(
                "<file name=\"{}\">\n{}\n</file>\n\n",
                a.name,
                a.text.trim_end()
            ));
        }
        s.push_str(&self.content);
        s
    }
}

/// A rough size of a text in tokens (3 characters per token: code and other languages are denser than English prose).
pub fn estimate_tokens(chars: usize) -> usize {
    chars / 3 + 1
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

/// The title a chat gets from its first message: one line, at most 42 characters.
pub fn auto_title(text: &str) -> String {
    let one_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate(&one_line, 42)
}

impl Chat {
    pub fn new() -> Chat {
        let now = unix_secs();
        Chat {
            id: new_id(),
            title: NEW_TITLE.to_string(),
            created: now,
            updated: now,
            model: String::new(),
            system: String::new(),
            messages: Vec::new(),
        }
    }

    /// The title comes from the first thing the user wrote.
    pub fn retitle(&mut self) {
        if self.title == NEW_TITLE {
            if let Some(m) = self.messages.iter().find(|m| m.role == Role::User) {
                let title = auto_title(&m.content);
                if !title.is_empty() {
                    self.title = title;
                }
            }
        }
    }

    /// A title typed by the user: one line of at most 80 characters. False (and nothing changes) when it is empty.
    pub fn set_title(&mut self, title: &str) -> bool {
        let one_line: String = title.split_whitespace().collect::<Vec<_>>().join(" ");
        if one_line.is_empty() {
            return false;
        }
        self.title = truncate(&one_line, 80);
        true
    }

    /// The edit of user message `i`: it gets the new text and everything after it is dropped (the conversation is asked again from there).
    /// False (and nothing changes) when `i` is not a user message or the text is empty.
    pub fn replace_from(&mut self, i: usize, text: &str) -> bool {
        let text = text.trim();
        if text.is_empty() || !matches!(self.messages.get(i), Some(m) if m.role == Role::User) {
            return false;
        }
        if i == 0 && self.title == auto_title(&self.messages[0].content) {
            self.title = NEW_TITLE.to_string();
        }
        let attachments = std::mem::take(&mut self.messages[i].attachments);
        self.messages.truncate(i);
        let mut msg = Message::new(Role::User, text);
        msg.attachments = attachments;
        self.messages.push(msg);
        self.retitle();
        true
    }

    /// The conversation as the server wants it: the system prompt, then every finished message (an answer that is empty or failed is left out).
    pub fn api_messages(&self, default_system: &str) -> Vec<(Role, String)> {
        let mut out = Vec::new();
        let system = if self.system.trim().is_empty() {
            default_system
        } else {
            self.system.as_str()
        };
        if !system.trim().is_empty() {
            out.push((Role::System, system.to_string()));
        }
        for m in &self.messages {
            if m.error.is_some() || (m.role == Role::Assistant && m.content.trim().is_empty()) {
                continue;
            }
            out.push((m.role, m.prompt_text()));
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
            for a in &m.attachments {
                s.push_str(&format!("*attached: {}*\n\n", a.name));
            }
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
        let safe: String = id
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
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
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return out;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            if let Ok(c) = serde_json::from_str::<Chat>(&text) {
                out.push(ChatMeta {
                    id: c.id,
                    title: c.title,
                    updated: c.updated,
                });
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
        c.messages
            .push(Message::new(Role::User, "Hello there,\n how are you?"));
        let mut a = Message::new(Role::Assistant, "Fine.");
        a.reasoning = "thinking".into();
        a.stats = Some(Stats {
            completion_tokens: 3,
            tok_per_s: 40.5,
            ..Default::default()
        });
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
    fn renaming_and_editing_a_chat() {
        let mut c = Chat::new();
        for (role, text) in [
            (Role::User, "first question"),
            (Role::Assistant, "answer"),
            (Role::User, "second"),
            (Role::Assistant, "answer 2"),
        ] {
            c.messages.push(Message::new(role, text));
        }
        c.retitle();
        assert_eq!(c.title, "first question");
        assert!(!c.set_title("   "));
        assert_eq!(c.title, "first question");
        assert!(c.set_title("My\n  project   notes"));
        assert_eq!(c.title, "My project notes");
        // editing the second user message drops what follows it and keeps a title the user chose
        assert!(c.replace_from(2, "  second, better  "));
        assert_eq!(c.messages.len(), 3);
        assert_eq!(c.messages[2].content, "second, better");
        assert_eq!(c.title, "My project notes");
        // an answer, an empty text and a missing message are refused
        assert!(!c.replace_from(1, "x"));
        assert!(!c.replace_from(2, "  "));
        assert!(!c.replace_from(9, "x"));
        assert_eq!(c.messages.len(), 3);
        // editing the first message of a chat with an automatic title renames it too
        let mut d = Chat::new();
        d.messages.push(Message::new(Role::User, "old question"));
        d.messages.push(Message::new(Role::Assistant, "a"));
        d.retitle();
        assert!(d.replace_from(0, "new question"));
        assert_eq!(d.title, "new question");
        assert_eq!(d.messages.len(), 1);
        // a long title is cut
        assert!(d.set_title(&"x".repeat(200)));
        assert!(d.title.chars().count() <= 81);
    }

    #[test]
    fn attached_files_go_in_front_of_the_text_and_survive_an_edit() {
        let mut c = Chat::new();
        let mut m = Message::new(Role::User, "Add type hints.");
        m.attachments.push(Attachment {
            name: "a.py".into(),
            text: "def f(x):\n    return x\n\n".into(),
        });
        c.messages.push(m);
        c.messages.push(Message::new(Role::Assistant, "done"));
        let p = c.api_messages("");
        assert_eq!(
            p[0].1,
            "<file name=\"a.py\">\ndef f(x):\n    return x\n</file>\n\nAdd type hints."
        );
        assert!(estimate_tokens(p[0].1.chars().count()) > 10);
        assert!(c.replace_from(0, "Add docstrings."));
        assert_eq!(c.messages[0].attachments.len(), 1);
        assert!(c.api_messages("")[0].1.ends_with("Add docstrings."));
        // a chat file of an older version has no attachments field
        let old = r#"{"id":"x","title":"t","created":1,"updated":1,"messages":[{"id":"m","role":"user","content":"hi"}]}"#;
        let back: Chat = serde_json::from_str(old).unwrap();
        assert!(back.messages[0].attachments.is_empty());
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
