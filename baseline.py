"""
baseline.py - Baseline AI Agent for Email-OpenEnv
Uses OpenAI GPT-4o-mini to classify, extract, prioritize, and reply to emails.
"""

import os
import json
import re
from openai import OpenAI
from models import AgentAction, Email, ExtractedEntities, EmailCategory, PriorityLevel
from environment import EmailEnvironment


# ─── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email processing AI agent operating in an OpenEnv environment.

For each email you receive, you must perform 4 tasks and respond ONLY with valid JSON (no markdown, no extra text):

{
  "email_id": <integer>,
  "category": "<Work | Personal | Spam | Urgent>",
  "priority": "<Low | Medium | High | Critical>",
  "extracted_entities": {
    "name": "<sender's full name or null>",
    "date": "<relevant date mentioned or null>",
    "deadline": "<deadline date or null>",
    "request": "<main request or action needed in one sentence or null>",
    "organization": "<company or organization name or null>",
    "contact": "<phone or email contact mentioned or null>"
  },
  "reply": "<professional email reply text>",
  "reasoning": "<brief explanation of your classification decision>"
}

Guidelines:
- Work: professional emails, job applications, HR notices, invoices, meetings
- Personal: family, friends, social emails
- Spam: promotional scams, phishing, too-good-to-be-true offers
- Urgent: production issues, emergencies, angry customers demanding immediate action
- Priority Critical: systems down, angry escalation threats, immediate financial impact
- Priority High: payment due, important meetings, time-sensitive work
- Priority Medium: standard work emails, leave applications, scheduling
- Priority Low: personal messages, low-importance promotions, general inquiries
- For Spam emails: write a warning reply (do not engage with the spam)
- For Urgent emails: acknowledge urgency clearly in the reply
- Keep replies professional, concise, and actionable
"""


class BaselineAgent:
    """
    A GPT-4o-mini powered baseline agent for the Email-OpenEnv.

    Usage:
        agent = BaselineAgent(api_key="sk-...")
        env = EmailEnvironment()
        state = env.reset()
        while not state.task_complete:
            action = agent.act(state.current_email)
            result = env.step(action)
            state = result.observation
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def act(self, email: Email) -> AgentAction:
        """
        Process an email and return a structured AgentAction.

        Args:
            email: The Email object to process

        Returns:
            AgentAction with all four tasks completed
        """
        user_message = self._format_email(email)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,  # low temperature for consistent structured output
            max_tokens=800,
        )

        raw = response.choices[0].message.content.strip()
        return self._parse_response(raw, email.id)

    def _format_email(self, email: Email) -> str:
        return (
            f"EMAIL ID: {email.id}\n"
            f"FROM: {email.sender} <{email.sender_email}>\n"
            f"SUBJECT: {email.subject}\n"
            f"DATE: {email.timestamp}\n"
            f"BODY:\n{email.body}"
        )

    def _parse_response(self, raw: str, email_id: int) -> AgentAction:
        """Parse the GPT JSON response into a validated AgentAction."""
        # Strip potential markdown code fences
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Fallback: return a safe default action
            return self._fallback_action(email_id, raw)

        entities = data.get("extracted_entities", {})
        return AgentAction(
            email_id=email_id,
            category=EmailCategory(data.get("category", "Work")),
            priority=PriorityLevel(data.get("priority", "Medium")),
            extracted_entities=ExtractedEntities(
                name=entities.get("name"),
                date=entities.get("date"),
                deadline=entities.get("deadline"),
                request=entities.get("request"),
                organization=entities.get("organization"),
                contact=entities.get("contact"),
            ),
            reply=data.get("reply", "Thank you for your email. We will get back to you shortly."),
            reasoning=data.get("reasoning"),
        )

    def _fallback_action(self, email_id: int, raw_text: str) -> AgentAction:
        """Safe fallback when JSON parsing fails."""
        return AgentAction(
            email_id=email_id,
            category=EmailCategory.WORK,
            priority=PriorityLevel.MEDIUM,
            extracted_entities=ExtractedEntities(),
            reply="Thank you for your email. We have received your message and will respond shortly.",
            reasoning=f"[PARSE ERROR] Raw output: {raw_text[:100]}",
        )


# ─── Runner ────────────────────────────────────────────────────────────────────

def run_baseline(api_key: str = None, verbose: bool = True) -> dict:
    """
    Run the full baseline agent through all 10 emails.

    Returns:
        dict with total_reward, max_reward, accuracy_pct, per-email results
    """
    env = EmailEnvironment()
    agent = BaselineAgent(api_key=api_key)

    state = env.reset()
    results = []

    print("=" * 60)
    print("  EMAIL-OPENENV BASELINE AGENT RUN")
    print("=" * 60)

    while not state.task_complete:
        email = state.current_email
        print(f"\n📩 Processing Email {state.step_number + 1}/{state.total_emails}: {email.subject[:50]}...")

        action = agent.act(email)
        result = env.step(action)

        if verbose:
            print(f"   Category : {action.category}")
            print(f"   Priority : {action.priority}")
            print(f"   Reward   : {result.reward:.2f}/4.0")
            print(f"   Feedback :\n{result.info.feedback}")

        results.append({
            "email_id": email.id,
            "subject": email.subject,
            "action": action.model_dump(),
            "reward": result.reward,
            "grader": result.info.model_dump(),
        })

        state = result.observation

    total = env.total_reward
    maximum = env.max_possible_reward
    pct = (total / maximum) * 100 if maximum > 0 else 0

    print("\n" + "=" * 60)
    print(f"  FINAL SCORE: {total:.2f} / {maximum:.1f} ({pct:.1f}%)")
    print("=" * 60)

    return {
        "total_reward": total,
        "max_reward": maximum,
        "accuracy_pct": pct,
        "per_email": results,
    }


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else None
    run_baseline(api_key=key)