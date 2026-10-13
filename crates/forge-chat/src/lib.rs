//! ForgeInfer chat layer: chat templates as plain code and an incremental parser of model output (reasoning, answer, tool calls). No Jinja, no regex, std only.

pub mod parse;
pub mod pyfmt;
pub mod template;

pub use parse::{parse, CallFormat, Event, OutputParser, Parsed};
pub use template::{
    render_chatml, render_llama3, render_qwen3, render_qwen35, render_smollm2, split_think, today_dd_mon_yyyy, wire_to_template,
};

use forge_core::{Json, Result};

/// The chat dialect of a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatFormat {
    /// plain ChatML without an injected system turn and without tool support (base models fine-tuned on ChatML, our own checkpoints)
    ChatMl,
    SmolLm2,
    /// Llama 3.x instruct models (`<|start_header_id|>` ... `<|eot_id|>`), no tools
    Llama3,
    Qwen3,
    /// `thinking_default`: what an absent `enable_thinking` means for this checkpoint's template (false for 0.8B / 2B, true for 4B / 9B).
    Qwen35 {
        thinking_default: bool,
    },
}

impl ChatFormat {
    /// Render the prompt for `messages` (OpenAI layout, tool-call arguments already objects: see `wire_to_template`).
    pub fn render(
        &self,
        messages: &[Json],
        tools: &[Json],
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        match self {
            ChatFormat::ChatMl => render_chatml(messages, add_generation_prompt),
            ChatFormat::SmolLm2 => render_smollm2(messages, add_generation_prompt),
            ChatFormat::Llama3 => render_llama3(messages, add_generation_prompt, &today_dd_mon_yyyy()),
            ChatFormat::Qwen3 => {
                render_qwen3(messages, tools, add_generation_prompt, enable_thinking)
            }
            ChatFormat::Qwen35 { thinking_default } => render_qwen35(
                messages,
                tools,
                add_generation_prompt,
                enable_thinking,
                *thinking_default,
            ),
        }
    }

    /// How the model writes tool calls (None: the template has no tool support).
    pub fn call_format(&self) -> Option<CallFormat> {
        match self {
            ChatFormat::ChatMl | ChatFormat::SmolLm2 | ChatFormat::Llama3 => None,
            ChatFormat::Qwen3 => Some(CallFormat::Json),
            ChatFormat::Qwen35 { .. } => Some(CallFormat::Xml),
        }
    }

    /// True when the rendered generation prompt ends inside an open `<think>` block (the model's first tokens are reasoning).
    pub fn thinking_open(&self, enable_thinking: Option<bool>) -> bool {
        match self {
            ChatFormat::Qwen35 { thinking_default } => enable_thinking.unwrap_or(*thinking_default),
            _ => false,
        }
    }
}
