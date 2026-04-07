"""
app.py - Gradio UI for Email-OpenEnv
Provides an interactive interface to process emails with an AI agent.
"""

import os
import json
import gradio as gr
from openai import OpenAI

from environment import EmailEnvironment
from models import AgentAction, ExtractedEntities, EmailCategory, PriorityLevel
import re

# ─── Global environment instance ──────────────────────────────────────────────
env = EmailEnvironment()
_state = env.reset()

# Category color badges (HTML)
CATEGORY_COLORS = {
    "Work": "#3B82F6",
    "Personal": "#10B981",
    "Spam": "#EF4444",
    "Urgent": "#F59E0B",
}

PRIORITY_COLORS = {
    "Low": "#6B7280",
    "Medium": "#3B82F6",
    "High": "#F59E0B",
    "Critical": "#EF4444",
}

SYSTEM_PROMPT = """You are an expert email processing AI agent.

For each email, respond ONLY with valid JSON (no markdown, no extra text):

{
  "email_id": <integer>,
  "category": "<Work | Personal | Spam | Urgent>",
  "priority": "<Low | Medium | High | Critical>",
  "extracted_entities": {
    "name": "<sender full name or null>",
    "date": "<relevant date or null>",
    "deadline": "<deadline or null>",
    "request": "<main request in one sentence or null>",
    "organization": "<org name or null>",
    "contact": "<phone/email or null>"
  },
  "reply": "<professional email reply>",
  "reasoning": "<why you classified it this way>"
}
"""


def call_openai(email_obj, api_key: str) -> AgentAction:
    """Call OpenAI API to process an email."""
    client = OpenAI(api_key=api_key)
    prompt = (
        f"EMAIL ID: {email_obj.id}\n"
        f"FROM: {email_obj.sender} <{email_obj.sender_email}>\n"
        f"SUBJECT: {email_obj.subject}\n"
        f"DATE: {email_obj.timestamp}\n"
        f"BODY:\n{email_obj.body}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    raw = response.choices[0].message.content.strip()
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    data = json.loads(clean)
    entities = data.get("extracted_entities", {})
    return AgentAction(
        email_id=email_obj.id,
        category=EmailCategory(data.get("category", "Work")),
        priority=PriorityLevel(data.get("priority", "Medium")),
        extracted_entities=ExtractedEntities(**{k: v for k, v in entities.items() if v}),
        reply=data.get("reply", ""),
        reasoning=data.get("reasoning", ""),
    )


def format_email_html(email) -> str:
    return f"""
    <div style="font-family:'Segoe UI',sans-serif; background:#0f172a; color:#e2e8f0;
                border-radius:12px; padding:24px; border:1px solid #1e293b;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px;">
            <div>
                <div style="font-size:18px; font-weight:700; color:#f1f5f9; margin-bottom:4px;">
                    {email.subject}
                </div>
                <div style="color:#94a3b8; font-size:13px;">
                    <span style="color:#60a5fa;">●</span> From: <strong>{email.sender}</strong>
                    &lt;{email.sender_email}&gt;
                </div>
            </div>
            <div style="color:#64748b; font-size:12px; white-space:nowrap; padding-top:4px;">
                {email.timestamp}
            </div>
        </div>
        <div style="background:#1e293b; border-radius:8px; padding:16px;
                    line-height:1.7; font-size:14px; color:#cbd5e1; white-space:pre-wrap;">{email.body}</div>
    </div>
    """


def format_result_html(action: AgentAction, reward: float, feedback: str) -> str:
    cat_color = CATEGORY_COLORS.get(str(action.category), "#6B7280")
    pri_color = PRIORITY_COLORS.get(str(action.priority), "#6B7280")
    cat_val = action.category.value if hasattr(action.category, "value") else action.category
    pri_val = action.priority.value if hasattr(action.priority, "value") else action.priority
    e = action.extracted_entities
    entities_rows = ""
    for label, val in [("Name", e.name), ("Date", e.date), ("Deadline", e.deadline),
                        ("Request", e.request), ("Organization", e.organization), ("Contact", e.contact)]:
        if val:
            entities_rows += f"""
            <tr>
                <td style="color:#94a3b8; padding:4px 12px 4px 0; font-size:13px;">{label}</td>
                <td style="color:#e2e8f0; padding:4px 0; font-size:13px;">{val}</td>
            </tr>"""

    feedback_html = feedback.replace("\n", "<br>")
    reward_pct = int((reward / 4.0) * 100)
    reward_color = "#10B981" if reward >= 3 else ("#F59E0B" if reward >= 2 else "#EF4444")

    return f"""
    <div style="font-family:'Segoe UI',sans-serif; color:#e2e8f0;">

      <!-- Badges row -->
      <div style="display:flex; gap:10px; margin-bottom:16px; flex-wrap:wrap;">
        <span style="background:{cat_color}22; color:{cat_color}; border:1px solid {cat_color}44;
                     padding:4px 14px; border-radius:20px; font-size:13px; font-weight:600;">
          📂 {cat_val}
        </span>
        <span style="background:{pri_color}22; color:{pri_color}; border:1px solid {pri_color}44;
                     padding:4px 14px; border-radius:20px; font-size:13px; font-weight:600;">
          🚦 {pri_val} Priority
        </span>
        <span style="background:{reward_color}22; color:{reward_color}; border:1px solid {reward_color}44;
                     padding:4px 14px; border-radius:20px; font-size:13px; font-weight:700;">
          ⭐ Score: {reward:.2f}/4.0 ({reward_pct}%)
        </span>
      </div>

      <!-- Extracted Entities -->
      <div style="background:#1e293b; border-radius:10px; padding:16px; margin-bottom:14px;">
        <div style="font-size:13px; font-weight:700; color:#94a3b8; text-transform:uppercase;
                    letter-spacing:.08em; margin-bottom:10px;">🔍 Extracted Entities</div>
        <table style="border-collapse:collapse; width:100%;">
          {entities_rows if entities_rows else '<tr><td style="color:#64748b;font-size:13px;">None extracted</td></tr>'}
        </table>
      </div>

      <!-- Reasoning -->
      {'<div style="background:#1e293b; border-radius:10px; padding:14px; margin-bottom:14px;"><div style="font-size:13px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">💡 Reasoning</div><div style="font-size:13px;color:#cbd5e1;">'+action.reasoning+'</div></div>' if action.reasoning else ''}

      <!-- AI Reply -->
      <div style="background:#1e293b; border-radius:10px; padding:16px; margin-bottom:14px;">
        <div style="font-size:13px; font-weight:700; color:#94a3b8; text-transform:uppercase;
                    letter-spacing:.08em; margin-bottom:10px;">✉️ AI-Generated Reply</div>
        <div style="font-size:14px; color:#cbd5e1; white-space:pre-wrap; line-height:1.7;">{action.reply}</div>
      </div>

      <!-- Grader feedback -->
      <div style="background:#172033; border:1px solid #1e293b; border-radius:10px; padding:14px;">
        <div style="font-size:13px; font-weight:700; color:#94a3b8; text-transform:uppercase;
                    letter-spacing:.08em; margin-bottom:8px;">🧮 Grader Breakdown</div>
        <div style="font-size:13px; color:#94a3b8; line-height:1.8;">{feedback_html}</div>
      </div>
    </div>
    """


# ─── Main processing function ──────────────────────────────────────────────────

def process_email(api_key: str):
    global _state, env

    if not api_key.strip():
        return (
            gr.update(value="<p style='color:#EF4444;'>⚠️ Please enter your OpenAI API key.</p>"),
            gr.update(value=""),
            gr.update(value="Waiting…"),
        )

    if _state.task_complete:
        total = env.total_reward
        max_r = env.max_possible_reward
        pct = (total / max_r) * 100 if max_r else 0
        done_html = f"""
        <div style="text-align:center; padding:40px; font-family:'Segoe UI',sans-serif; color:#10B981;">
          <div style="font-size:48px; margin-bottom:12px;">🎉</div>
          <div style="font-size:24px; font-weight:700;">All emails processed!</div>
          <div style="font-size:18px; margin-top:10px; color:#94a3b8;">
            Final Score: <strong style="color:#10B981;">{total:.2f} / {max_r:.1f}</strong> ({pct:.1f}%)
          </div>
        </div>"""
        return gr.update(value=done_html), gr.update(value=""), gr.update(value="✅ Complete")

    email = _state.current_email
    email_html = format_email_html(email)
    progress = f"Email {_state.step_number + 1} / {_state.total_emails}"

    try:
        action = call_openai(email, api_key)
        result = env.step(action)
        _state = result.observation
        result_html = format_result_html(action, result.reward, result.info.feedback)
        return gr.update(value=email_html), gr.update(value=result_html), gr.update(value=progress)

    except Exception as exc:
        error_html = f"<p style='color:#EF4444; font-family:monospace;'>❌ Error: {str(exc)}</p>"
        return gr.update(value=email_html), gr.update(value=error_html), gr.update(value=progress)


def reset_env():
    global _state, env
    env = EmailEnvironment()
    _state = env.reset()
    email = _state.current_email
    progress = f"Email 1 / {_state.total_emails}"
    return (
        gr.update(value=format_email_html(email)),
        gr.update(value="<p style='color:#64748b; font-family:\"Segoe UI\",sans-serif; padding:20px;'>Press <strong>Process Email</strong> to analyse this email with AI.</p>"),
        gr.update(value=progress),
    )


def show_current():
    global _state
    if _state.task_complete:
        return gr.update(value="<p style='color:#10B981;'>All done!</p>")
    return gr.update(value=format_email_html(_state.current_email))


# ─── Gradio UI Layout ──────────────────────────────────────────────────────────

CUSTOM_CSS = """
body, .gradio-container { background: #0a0f1e !important; }
.gr-panel { background: transparent !important; border: none !important; }
footer { display: none !important; }
.progress-badge {
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;
    color: #60a5fa;
    font-weight: 600;
    letter-spacing: 0.05em;
}
"""

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    title="Email-OpenEnv",
    css=CUSTOM_CSS,
) as demo:

    # Header
    gr.HTML("""
    <div style="text-align:center; padding:32px 0 24px; font-family:'Segoe UI',sans-serif;">
      <div style="font-size:13px; letter-spacing:.25em; text-transform:uppercase;
                  color:#3B82F6; font-weight:600; margin-bottom:8px;">OpenEnv Environment</div>
      <h1 style="font-size:36px; font-weight:800; color:#f1f5f9; margin:0 0 8px;">
        📬 Email Open Environment
      </h1>
      <p style="color:#64748b; font-size:15px; margin:0;">
        AI agent reads, classifies, extracts, and replies to emails · Powered by GPT-4o-mini
      </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""<div style="font-size:12px;color:#475569;font-family:'Segoe UI',sans-serif;
                       margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;">
                       🔑 OpenAI API Key</div>""")
            api_key_input = gr.Textbox(
                placeholder="sk-...",
                type="password",
                show_label=False,
                container=False,
            )
        with gr.Column(scale=1):
            progress_label = gr.HTML(
                value="<div class='progress-badge' style='padding-top:6px;'>Email 1 / 10</div>"
            )

    with gr.Row():
        process_btn = gr.Button("⚡ Process Email with AI", variant="primary", scale=3)
        reset_btn = gr.Button("🔄 Reset", variant="secondary", scale=1)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""<div style="font-size:12px;color:#475569;font-family:'Segoe UI',sans-serif;
                       margin-bottom:8px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;">
                       📩 Current Email</div>""")
            email_display = gr.HTML(
                value=format_email_html(env.current_email()) if env.current_email() else ""
            )

        with gr.Column(scale=1):
            gr.HTML("""<div style="font-size:12px;color:#475569;font-family:'Segoe UI',sans-serif;
                       margin-bottom:8px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;">
                       🤖 AI Agent Output</div>""")
            result_display = gr.HTML(
                value="<p style='color:#475569; font-family:\"Segoe UI\",sans-serif; padding:20px;'>"
                      "Press <strong>Process Email with AI</strong> to analyse.</p>"
            )

    # Footer legend
    gr.HTML("""
    <div style="display:flex; justify-content:center; gap:20px; flex-wrap:wrap;
                padding:20px 0; font-family:'Segoe UI',sans-serif; font-size:12px; color:#475569;">
        <span><span style="color:#3B82F6">●</span> Work</span>
        <span><span style="color:#10B981">●</span> Personal</span>
        <span><span style="color:#EF4444">●</span> Spam</span>
        <span><span style="color:#F59E0B">●</span> Urgent</span>
        <span style="color:#334155;">|</span>
        <span>Score is out of <strong>4.0</strong> per email (Classification + Extraction + Reply + Priority)</span>
    </div>
    """)

    # Wire up buttons
    process_btn.click(
        fn=process_email,
        inputs=[api_key_input],
        outputs=[email_display, result_display, progress_label],
    )
    reset_btn.click(
        fn=reset_env,
        inputs=[],
        outputs=[email_display, result_display, progress_label],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)