"""
grader.py - Evaluation and scoring logic for Email-OpenEnv
Scores agent actions against ground-truth labels using rule-based methods.
"""

from models import AgentAction, Email, GraderResult
import re


# ─── Keyword sets for reply quality assessment ─────────────────────────────────

PROFESSIONAL_KEYWORDS = {
    "thank you", "regards", "sincerely", "dear", "please", "appreciate",
    "kindly", "best wishes", "noted", "will", "ensure", "address",
    "assist", "help", "confirm", "follow up", "review", "team"
}

SPAM_REPLY_BAD_WORDS = {
    "congratulations", "prize", "lottery", "click here", "act now",
    "winner", "claim", "free money"
}

# Minimum reply length (characters) for a non-spam email
MIN_REPLY_LENGTH = 80


class Grader:
    """
    Rule-based grader that scores AgentAction across four dimensions:
      1. Classification accuracy   (0–1)
      2. Extraction correctness    (0–1)
      3. Reply quality             (0–1)
      4. Priority correctness      (0–1)
    Total reward = sum of four scores (max 4.0 per email).
    """

    def __init__(self, ground_truth: dict[int, dict]):
        self.ground_truth = ground_truth

    def grade(self, action: AgentAction, email: Email) -> GraderResult:
        gt = self.ground_truth.get(email.id, {})

        c_score = self._grade_classification(action, gt)
        e_score = self._grade_extraction(action, email, gt)
        r_score = self._grade_reply(action, gt)
        p_score = self._grade_priority(action, gt)

        total = round(c_score + e_score + r_score + p_score, 3)

        feedback = self._build_feedback(c_score, e_score, r_score, p_score, gt, action)

        return GraderResult(
            classification_score=c_score,
            extraction_score=e_score,
            reply_score=r_score,
            priority_score=p_score,
            total_reward=total,
            feedback=feedback,
        )

    # ── Individual graders ──────────────────────────────────────────────────

    def _grade_classification(self, action: AgentAction, gt: dict) -> float:
        """Exact match: 1.0 if correct, 0.0 otherwise."""
        if not gt.get("category"):
            return 0.5  # no ground truth → neutral
        predicted = action.category.value if hasattr(action.category, "value") else action.category
        expected = gt["category"]
        return 1.0 if predicted == expected else 0.0

    def _grade_extraction(self, action: AgentAction, email: Email, gt: dict) -> float:
        """
        Checks how many ground-truth key entities appear in the extracted fields.
        Score = (matched entities) / max(len(key_entities), 1)
        """
        key_entities: list[str] = gt.get("key_entities", [])
        if not key_entities:
            # Spam email – extraction should be empty/minimal
            entities = action.extracted_entities
            fields = [
                entities.name or "",
                entities.date or "",
                entities.deadline or "",
                entities.request or "",
            ]
            combined = " ".join(fields).lower()
            # Penalise if agent hallucinates entities for spam
            if len(combined.strip()) < 20:
                return 1.0
            return 0.5

        # Build combined text from all extracted fields
        entities = action.extracted_entities
        combined = " ".join(filter(None, [
            entities.name,
            entities.date,
            entities.deadline,
            entities.request,
            entities.organization,
            entities.contact,
        ])).lower()

        # Also check against the full email body as a fallback (extraction quality)
        email_text = (email.body + " " + email.subject).lower()

        hits = 0
        for entity in key_entities:
            entity_lower = entity.lower()
            # Check if entity appears in extracted output OR email (proxy for recognizing it)
            if entity_lower in combined or entity_lower in email_text:
                # Give credit only if it appears in extraction
                if entity_lower in combined:
                    hits += 1
                else:
                    hits += 0.5  # partial credit – it was in email but not extracted

        score = hits / len(key_entities)
        return min(round(score, 3), 1.0)

    def _grade_reply(self, action: AgentAction, gt: dict) -> float:
        """
        Rule-based reply quality assessment:
        - Spam → should decline or warn, not write a real reply
        - Urgent → should acknowledge urgency
        - Work/Personal → length + professional tone keywords
        """
        reply = action.reply.lower()
        category = gt.get("category", "Work")

        # ── Spam emails ────────────────────────────────────────────────────
        if category == "Spam":
            # Good reply: short warning/decline
            bad_engagement = any(w in reply for w in ["claim", "processing fee", "bank account", "send your"])
            if bad_engagement:
                return 0.0
            warn_words = ["spam", "scam", "phishing", "suspicious", "not respond", "ignore", "fraudulent", "caution"]
            if any(w in reply for w in warn_words):
                return 1.0
            return 0.5  # neutral / generic decline

        # ── Non-spam replies ───────────────────────────────────────────────
        # Length check
        if len(action.reply) < MIN_REPLY_LENGTH:
            length_score = 0.3
        elif len(action.reply) < 150:
            length_score = 0.7
        else:
            length_score = 1.0

        # Professional tone check
        prof_hits = sum(1 for kw in PROFESSIONAL_KEYWORDS if kw in reply)
        tone_score = min(prof_hits / 4, 1.0)  # 4 hits → full score

        # Urgency acknowledgement for urgent emails
        if category == "Urgent":
            urgency_words = ["urgent", "immediately", "priority", "asap", "escalate", "sorry", "apologize"]
            urgency_hit = any(w in reply for w in urgency_words)
            urgency_bonus = 0.2 if urgency_hit else -0.2
        else:
            urgency_bonus = 0.0

        raw = (length_score * 0.5) + (tone_score * 0.5) + urgency_bonus
        return round(min(max(raw, 0.0), 1.0), 3)

    def _grade_priority(self, action: AgentAction, gt: dict) -> float:
        """
        Exact match = 1.0, adjacent level = 0.5, two+ levels off = 0.0.
        Priority order: Low(0) Medium(1) High(2) Critical(3)
        """
        order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        if not gt.get("priority"):
            return 0.5
        predicted = action.priority.value if hasattr(action.priority, "value") else action.priority
        expected = gt["priority"]
        diff = abs(order.get(predicted, 1) - order.get(expected, 1))
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.5
        else:
            return 0.0

    # ── Feedback builder ────────────────────────────────────────────────────

    def _build_feedback(self, c, e, r, p, gt, action: AgentAction) -> str:
        lines = []
        lines.append(f"✅ Classification: {c:.1f}/1.0 (expected: {gt.get('category','?')}, got: {action.category})")
        lines.append(f"✅ Extraction:     {e:.1f}/1.0 (key entities matched)")
        lines.append(f"✅ Reply Quality:  {r:.1f}/1.0")
        lines.append(f"✅ Priority:       {p:.1f}/1.0 (expected: {gt.get('priority','?')}, got: {action.priority})")
        lines.append(f"━━ Total Reward:  {c+e+r+p:.2f}/4.0")
        return "\n".join(lines)