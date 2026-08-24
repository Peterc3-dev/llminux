use rustyline::DefaultEditor;
use serde::{Deserialize, Serialize};
use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const FLM_URL: &str = "http://localhost:52625/api/chat";
const FLM_MODEL: &str = "qwen3:1.7b";
const ESCALATION_URL: &str = "http://localhost:8090/v1/chat/completions";

static SYSTEM_PROMPT: &str = include_str!("../sentinel_prompt.txt");

#[derive(Debug, Deserialize)]
struct Verb {
    verb: String,
    args: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct DisplayVerb {
    display: String,
    content: serde_json::Value,
}

fn display(kind: &str, content: serde_json::Value) -> DisplayVerb {
    DisplayVerb { display: kind.to_string(), content }
}

fn render(dv: &DisplayVerb) {
    match dv.display.as_str() {
        "show_text" => {
            if let Some(s) = dv.content.as_str() {
                println!("{s}");
            } else {
                println!("{}", serde_json::to_string_pretty(&dv.content).unwrap_or_default());
            }
        }
        "show_table" => {
            if let Some(rows) = dv.content.as_array() {
                for row in rows {
                    println!("{row}");
                }
            } else {
                println!("{}", serde_json::to_string_pretty(&dv.content).unwrap_or_default());
            }
        }
        "confirm" => {
            if let Some(msg) = dv.content.as_str() {
                eprint!("\x1b[33m{msg}\x1b[0m [y/N] ");
                std::io::stderr().flush().ok();
            }
        }
        "notify" => {
            if let Some(s) = dv.content.as_str() {
                eprintln!("\x1b[36m{s}\x1b[0m");
            }
        }
        "error" => {
            if let Some(s) = dv.content.as_str() {
                eprintln!("\x1b[31merror: {s}\x1b[0m");
            }
        }
        _ => {
            println!("{}", serde_json::to_string_pretty(&dv.content).unwrap_or_default());
        }
    }
}

fn ask_sentinel(input: &str, context: &str) -> Result<Verb, String> {
    let mut messages = vec![
        serde_json::json!({"role": "system", "content": SYSTEM_PROMPT}),
    ];
    if !context.is_empty() {
        messages.push(serde_json::json!({
            "role": "system",
            "content": format!("Previous output (for context):\n{context}")
        }));
    }
    messages.push(serde_json::json!({"role": "user", "content": input.to_string()}));

    let payload = serde_json::json!({
        "model": FLM_MODEL,
        "messages": messages,
        "stream": false,
        "options": {"num_predict": 100, "temperature": 0}
    });

    let resp: serde_json::Value = ureq::post(FLM_URL)
        .header("Content-Type", "application/json")
        .send_json(&payload)
        .map_err(|e| format!("sentinel unreachable: {e}"))?
        .body_mut()
        .read_json()
        .map_err(|e| format!("sentinel JSON error: {e}"))?;

    let raw = resp["message"]["content"]
        .as_str()
        .ok_or("no content in sentinel response")?
        .to_string();

    parse_sentinel_output(&raw)
}

fn parse_sentinel_output(raw: &str) -> Result<Verb, String> {
    let mut s = raw.trim();
    if let Some(pos) = s.find("</think>") {
        s = s[pos + 8..].trim();
    }
    let s = s
        .trim_start_matches("```json")
        .trim_end_matches("```")
        .trim();

    if let Ok(v) = serde_json::from_str::<Verb>(s) {
        return Ok(v);
    }

    let mut fixed = s.to_string();
    while fixed.ends_with('}') && fixed.matches('{').count() < fixed.matches('}').count() {
        fixed.pop();
    }
    serde_json::from_str::<Verb>(&fixed).map_err(|e| format!("parse error: {e}\nraw: {s}"))
}

fn is_destructive(cmd: &str) -> bool {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    let first = tokens.first().copied().unwrap_or("");
    matches!(
        first,
        "rm" | "rmdir" | "mkfs" | "dd" | "shred" | "shutdown" | "reboot" | "poweroff"
    ) || cmd.contains("rm -rf")
        || cmd.contains("rm -r")
        || (first == "kill" && tokens.len() > 1)
}

fn confirm(msg: &str) -> bool {
    let prompt = format!("\x1b[33m{msg}\x1b[0m [y/N] ");
    let mut rl = match DefaultEditor::new() {
        Ok(rl) => rl,
        Err(_) => return false,
    };
    match rl.readline(&prompt) {
        Ok(line) => matches!(line.trim().to_lowercase().as_str(), "y" | "yes"),
        Err(_) => false,
    }
}

fn execute(verb: &Verb, cwd: &mut PathBuf) -> DisplayVerb {
    let args = &verb.args;
    match verb.verb.as_str() {
        "open_file" => {
            let path = args["path"].as_str().unwrap_or("");
            let resolved = resolve_path(path, cwd);
            match std::fs::read_to_string(&resolved) {
                Ok(content) => display("show_text", serde_json::json!(content)),
                Err(e) => display("error", serde_json::json!(format!("{resolved}: {e}"))),
            }
        }
        "list_dir" => {
            let path = args["path"].as_str().unwrap_or(".");
            let resolved = resolve_path(path, cwd);
            match std::fs::read_dir(&resolved) {
                Ok(entries) => {
                    let mut items: Vec<String> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| {
                            let name = e.file_name().to_string_lossy().to_string();
                            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                                format!("{name}/")
                            } else {
                                name
                            }
                        })
                        .collect();
                    items.sort();
                    display("show_text", serde_json::json!(items.join("\n")))
                }
                Err(e) => display("error", serde_json::json!(format!("{resolved}: {e}"))),
            }
        }
        "disk_usage" => {
            run_shell("df -h --output=target,size,used,avail,pcent / /home 2>/dev/null || df -h", cwd)
        }
        "play_media" => {
            let query = args["query"].as_str().unwrap_or("music");
            display("notify", serde_json::json!(format!("play_media: '{query}' — media dispatch not wired yet")))
        }
        "run_command" => {
            let cmd = args["cmd"].as_str().unwrap_or("");
            if cmd.is_empty() {
                return display("error", serde_json::json!("empty command"));
            }
            if is_destructive(cmd) && !confirm(&format!("execute: {cmd}")) {
                return display("notify", serde_json::json!("cancelled"));
            }
            if let Some(target) = cmd.strip_prefix("cd ") {
                let target = target.trim();
                let resolved = resolve_path(target, cwd);
                let p = PathBuf::from(&resolved);
                if p.is_dir() {
                    *cwd = p;
                    return display("notify", serde_json::json!(format!("→ {}", cwd.display())));
                } else {
                    return display("error", serde_json::json!(format!("not a directory: {resolved}")));
                }
            }
            run_shell(cmd, cwd)
        }
        "escalate" => {
            let task = args["task"].as_str().unwrap_or(
                args.as_str().unwrap_or(""),
            );
            escalate(task)
        }
        "show_text" => {
            let text = args["text"].as_str().unwrap_or("");
            display("show_text", serde_json::json!(text))
        }
        _ => display("error", serde_json::json!(format!("unknown verb: {}", verb.verb))),
    }
}

fn run_shell(cmd: &str, cwd: &PathBuf) -> DisplayVerb {
    match Command::new("bash")
        .arg("-c")
        .arg(cmd)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => {
            let output = match child.wait_with_output() {
                Ok(o) => o,
                Err(e) => return display("error", serde_json::json!(format!("wait failed: {e}"))),
            };
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let mut result = stdout.to_string();
            if !stderr.is_empty() {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str(&stderr);
            }
            if result.is_empty() {
                display("notify", serde_json::json!(format!("✓ (exit {})", output.status.code().unwrap_or(-1))))
            } else {
                display("show_text", serde_json::json!(result.trim_end()))
            }
        }
        Err(e) => display("error", serde_json::json!(format!("spawn failed: {e}"))),
    }
}

fn escalate(task: &str) -> DisplayVerb {
    if task.is_empty() {
        return display("error", serde_json::json!("empty escalation task"));
    }

    let payload = serde_json::json!({
        "model": "local",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant running as the brain tier of LLMINUX. Answer concisely."},
            {"role": "user", "content": task}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    });

    match ureq::post(ESCALATION_URL)
        .header("Content-Type", "application/json")
        .send_json(&payload)
    {
        Ok(mut r) => {
            let body: serde_json::Value = r.body_mut().read_json().unwrap_or_default();
            let content = body["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("(no response from brain tier)");
            display("show_text", serde_json::json!(content))
        }
        Err(_) => display("error", serde_json::json!("brain tier (30B) unreachable — start llama-qwen38-27b.service")),
    }
}

fn resolve_path(path: &str, cwd: &PathBuf) -> String {
    let expanded = if path.starts_with('~') {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/raz".to_string());
        path.replacen('~', &home, 1)
    } else {
        path.to_string()
    };
    let p = PathBuf::from(&expanded);
    if p.is_absolute() {
        expanded
    } else {
        cwd.join(&expanded).to_string_lossy().to_string()
    }
}

fn truncate_context(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        let start = text.floor_char_boundary(text.len() - max_chars);
        format!("…{}", &text[start..])
    }
}

fn main() {
    let mut cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/home/raz"));
    let mut last_output = String::new();

    if std::env::args().len() > 1 {
        let input: String = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
        match ask_sentinel(&input, "") {
            Ok(verb) => {
                let dv = execute(&verb, &mut cwd);
                render(&dv);
            }
            Err(e) => {
                eprintln!("\x1b[31merror: {e}\x1b[0m");
                std::process::exit(1);
            }
        }
        return;
    }

    println!("\x1b[32m┌──────────────────────────────┐\x1b[0m");
    println!("\x1b[32m│  LLMINUX  Hybridized Edition  │\x1b[0m");
    println!("\x1b[32m│  sentinel: qwen3:1.7b (NPU)  │\x1b[0m");
    println!("\x1b[32m│  brain: 30B MoE (GPU)        │\x1b[0m");
    println!("\x1b[32m└──────────────────────────────┘\x1b[0m");
    println!();

    let mut rl = DefaultEditor::new().expect("failed to init readline");

    loop {
        let prompt = format!("\x1b[32mllminux:{}\x1b[0m> ", cwd.display());
        let line = match rl.readline(&prompt) {
            Ok(l) => l,
            Err(rustyline::error::ReadlineError::Interrupted | rustyline::error::ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("readline error: {e}");
                break;
            }
        };

        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "quit" | "exit") {
            break;
        }

        rl.add_history_entry(input).ok();

        let context = truncate_context(&last_output, 500);
        match ask_sentinel(input, &context) {
            Ok(verb) => {
                let dv = execute(&verb, &mut cwd);
                if dv.display == "show_text" {
                    if let Some(s) = dv.content.as_str() {
                        last_output = s.to_string();
                    }
                }
                render(&dv);
            }
            Err(e) => {
                eprintln!("\x1b[31msentinel error: {e}\x1b[0m");
                last_output.clear();
            }
        }
        println!();
    }
}
