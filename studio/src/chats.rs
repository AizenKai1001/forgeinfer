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
    /// Prompt tokens the engine already had from an earlier request (its prefix reuse and prompt cache); 0 when the server does not say.
    #[serde(default)]
    pub cached_tokens: u32,
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

/// A passage of an attached file that was found for a question (chat with documents).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Citation {
    pub file: String,
    pub text: String,
    /// Cosine of the question's vector and the passage's.
    pub score: f32,
    /// Below the minimum score: one of the nearest passages of a question that matched nothing well.
    #[serde(default)]
    pub weak: bool,
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
    /// A user message whose files are searched instead of sent whole: the model reads the `citations` found for it (and for every later question of the chat).
    #[serde(default)]
    pub by_retrieval: bool,
    #[serde(default)]
    pub citations: Vec<Citation>,
    /// True while the passages are being looked for.
    #[serde(skip)]
    pub retrieving: bool,
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
            by_retrieval: false,
            citations: Vec::new(),
            retrieving: false,
        }
    }

    /// What the model reads: the attached files (each between `<file name=...>` tags), then the text; for a question about files that are searched, the passages found and the question.
    pub fn prompt_text(&self) -> String {
        if self.by_retrieval {
            let passages: Vec<(String, String)> = self.citations.iter().map(|c| (c.file.clone(), c.text.clone())).collect();
            return crate::rag::citations_prompt(&passages, self.citations.first().is_some_and(|c| c.weak), &self.content);
        }
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
    /// How many messages from the start of the conversation are left out of what the model reads because the context was full (whole turns, set by `fit`). It is kept from request to request, so that the
    /// text the engine already holds stays the beginning of the next prompt until the window has to move again.
    #[serde(default)]
    pub skip: usize,
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
            skip: 0,
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
        let by_retrieval = self.messages[i].by_retrieval;
        self.messages.truncate(i);
        self.skip = self.skip.min(i);
        let mut msg = Message::new(Role::User, text);
        msg.attachments = attachments;
        // the passages of the old wording are stale: the question is searched again
        msg.by_retrieval = by_retrieval;
        self.messages.push(msg);
        self.retitle();
        true
    }

    /// The conversation as the server wants it: the system prompt, then every finished message (an answer that is empty or failed is left out).
    pub fn api_messages(&self, default_system: &str) -> Vec<(Role, String)> {
        self.api_messages_from(default_system, self.skip)
    }

    /// The same with the first `skip` messages left out.
    fn api_messages_from(&self, default_system: &str, skip: usize) -> Vec<(Role, String)> {
        let mut out = Vec::new();
        let system = if self.system.trim().is_empty() {
            default_system
        } else {
            self.system.as_str()
        };
        if !system.trim().is_empty() {
            out.push((Role::System, system.to_string()));
        }
        // the passages found for a question are read for that question only: once a newer question has been asked, the older ones keep just their words (the answer to them is in the conversation)
        // and the context is not filled with the passages of every earlier turn
        let last_user = self.messages.iter().rposition(|m| m.role == Role::User);
        for (i, m) in self.messages.iter().enumerate().skip(skip) {
            if m.error.is_some() || (m.role == Role::Assistant && m.content.trim().is_empty()) {
                continue;
            }
            let text = if m.by_retrieval && Some(i) != last_user { m.content.clone() } else { m.prompt_text() };
            out.push((m.role, text));
        }
        out
    }

    /// Moves the start of what the model reads forward, by whole turns, when the conversation (system prompt included) is estimated at more than `budget_tokens`: to the first turn from which what is left is
    /// at most 70 percent of the budget, so that the next requests fit without moving the window again (a moved window changes the beginning of the prompt, and the engine has to read it all anew); the last user
    /// message, the one being answered, always stays. Returns the number of messages newly left out (0: nothing changed).
    pub fn fit(&mut self, default_system: &str, budget_tokens: usize) -> usize {
        let tokens = |chat: &Chat, skip: usize| -> usize {
            estimate_tokens(chat.api_messages_from(default_system, skip).iter().map(|(_, t)| t.chars().count()).sum())
        };
        self.skip = self.skip.min(self.messages.len());
        if tokens(self, self.skip) <= budget_tokens {
            return 0;
        }
        let Some(last_user) = self.messages.iter().rposition(|m| m.role == Role::User) else {
            return 0;
        };
        let starts: Vec<usize> = (self.skip + 1..=last_user).filter(|&i| self.messages[i].role == Role::User).collect();
        let target = budget_tokens * 7 / 10;
        let pick = starts.iter().copied().find(|&s| tokens(self, s) <= target).or_else(|| starts.last().copied());
        match pick {
            Some(s) => {
                let dropped = s - self.skip;
                self.skip = s;
                dropped
            }
            None => 0,
        }
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
        // a chat saved before the field existed still loads
        let old: Stats = serde_json::from_str(r#"{"completion_tokens":5,"prompt_tokens":7,"tok_per_s":1.5,"first_token_s":0.1,"total_s":1.0,"finish":"stop"}"#).unwrap();
        assert_eq!((old.prompt_tokens, old.cached_tokens), (7, 0));
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
        assert!(!back.messages[0].by_retrieval && back.messages[0].citations.is_empty());
    }

    #[test]
    fn a_question_about_searched_files_reads_the_passages_not_the_files() {
        let mut c = Chat::new();
        let mut m = Message::new(Role::User, "When does the ship dock?");
        m.attachments.push(Attachment { name: "log.txt".into(), text: "x".repeat(100_000) });
        m.by_retrieval = true;
        m.citations.push(Citation { file: "log.txt".into(), text: "The ship docks at six.".into(), score: 0.8, weak: false });
        c.messages.push(m);
        c.messages.push(Message::new(Role::Assistant, "At six."));
        // a later question has no files of its own and is searched too
        let mut q = Message::new(Role::User, "And the cargo?");
        q.by_retrieval = true;
        c.messages.push(q);
        let p = c.api_messages("");
        // the last question reads its passages (here: none matched), the earlier one keeps its words only
        assert_eq!(p[0].1, "When does the ship dock?");
        assert!(p[2].1.contains("No passage") && p[2].1.ends_with("And the cargo?"), "{}", p[2].1);
        // while it is the last question, the passages are read, and the file itself is never sent
        let mut only = Chat::new();
        only.messages.push(c.messages[0].clone());
        let q = only.api_messages("");
        assert!(q[0].1.contains("Citation 1 (from log.txt): \"The ship docks at six.\"") && q[0].1.ends_with("When does the ship dock?"), "{}", q[0].1);
        assert!(!q[0].1.contains("xxxxxxxx"), "the file itself is not sent");
        // the citations survive the chat file; editing keeps the mode but drops the stale passages
        let back: Chat = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back.messages[0].citations.len(), 1);
        assert!(back.messages[0].by_retrieval);
        assert!(c.replace_from(0, "When does it leave?"));
        assert!(c.messages[0].by_retrieval && c.messages[0].citations.is_empty());
        assert_eq!(c.messages[0].attachments.len(), 1);
    }

    /// A chat of `turns` user / assistant pairs, every message of `chars` characters.
    fn long_chat(turns: usize, chars: usize) -> Chat {
        let mut c = Chat::new();
        for t in 0..turns {
            c.messages.push(Message::new(Role::User, format!("{t:04}{}", "q".repeat(chars - 4))));
            c.messages.push(Message::new(Role::Assistant, format!("{t:04}{}", "a".repeat(chars - 4))));
        }
        c.messages.push(Message::new(Role::User, "the last question"));
        c
    }

    #[test]
    fn a_conversation_that_fits_is_left_alone() {
        let mut c = long_chat(3, 300);
        assert_eq!(c.fit("", 10_000), 0);
        assert_eq!(c.skip, 0);
        assert_eq!(c.api_messages("").len(), c.messages.len());
    }

    #[test]
    fn the_oldest_turns_are_left_out_by_whole_turns_down_to_seventy_percent_of_the_budget() {
        // 20 turns of 2 x 300 characters (about 200 tokens a turn) and a question: about 4,000 tokens
        let mut c = long_chat(20, 300);
        let before = c.api_messages("").len();
        let dropped = c.fit("", 3_000);
        assert!(dropped > 0 && dropped % 2 == 0, "whole turns: {dropped}");
        assert_eq!(c.skip, dropped);
        assert_eq!(c.messages[c.skip].role, Role::User, "the window starts at a user message");
        let kept = c.api_messages("");
        assert_eq!(kept.len(), before - dropped);
        let tokens = estimate_tokens(kept.iter().map(|(_, t)| t.chars().count()).sum());
        assert!(tokens <= 3_000 * 7 / 10, "{tokens} tokens: down to 70 percent of the budget");
        assert!(tokens > 3_000 * 7 / 10 - 400, "{tokens} tokens: not much more than needed was dropped");
        // the question being answered is always kept
        assert_eq!(kept.last().unwrap().1, "the last question");
    }

    #[test]
    fn the_window_stays_where_it_is_until_the_conversation_outgrows_it_again() {
        let mut c = long_chat(20, 300);
        c.fit("", 3_000);
        let skip = c.skip;
        // a few more turns still fit in the 30 percent that were freed: the start does not move (the engine's cached prefix stays valid)
        for t in 0..3 {
            c.messages.push(Message::new(Role::Assistant, format!("{t}{}", "a".repeat(299))));
            c.messages.push(Message::new(Role::User, format!("{t}{}", "q".repeat(299))));
            assert_eq!(c.fit("", 3_000), 0);
            assert_eq!(c.skip, skip);
        }
        // many more do not
        for t in 0..20 {
            c.messages.push(Message::new(Role::Assistant, format!("{t}{}", "a".repeat(299))));
            c.messages.push(Message::new(Role::User, format!("{t}{}", "q".repeat(299))));
        }
        assert!(c.fit("", 3_000) > 0);
        assert!(c.skip > skip);
    }

    #[test]
    fn a_single_message_longer_than_the_budget_is_still_sent_and_an_edit_pulls_the_window_back() {
        let mut c = Chat::new();
        c.messages.push(Message::new(Role::User, "x".repeat(30_000)));
        assert_eq!(c.fit("", 1_000), 0, "nothing to leave out: the server will say what it thinks");
        assert_eq!(c.skip, 0);
        let mut c = long_chat(20, 300);
        c.fit("", 3_000);
        assert!(c.skip > 0);
        // editing a message before the window (or in it) cuts the conversation there: the window cannot start beyond the cut
        assert!(c.replace_from(0, "a new start"));
        assert_eq!(c.skip, 0);
        assert_eq!(c.api_messages("").len(), 1);
        // a chat file of an older version has no window
        let old = r#"{"id":"x","title":"t","created":1,"updated":1,"messages":[]}"#;
        assert_eq!(serde_json::from_str::<Chat>(old).unwrap().skip, 0);
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
