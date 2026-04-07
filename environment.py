"""
environment.py - Core OpenEnv Logic for Email-OpenEnv
Implements reset(), step(), and state() following the OpenEnv API contract.
"""

from models import (
    Email, AgentAction, EnvironmentState, StepResult,
    EmailCategory, PriorityLevel
)
from grader import Grader
from typing import Optional
import copy


# ─── Sample Email Dataset (10 realistic emails) ───────────────────────────────

SAMPLE_EMAILS: list[dict] = [
    {
        "id": 1,
        "sender": "Rahul Sharma",
        "sender_email": "rahul.sharma@techcorp.in",
        "subject": "Job Application – Senior Python Developer",
        "body": (
            "Dear Hiring Manager,\n\n"
            "I am writing to apply for the Senior Python Developer position advertised on LinkedIn. "
            "My name is Rahul Sharma and I have 6 years of experience in backend development using Python, "
            "Django, and FastAPI. I am currently available to join by January 15, 2025.\n\n"
            "Please find my resume attached. I would love to schedule an interview at your earliest convenience.\n\n"
            "Best regards,\nRahul Sharma\n+91-9876543210"
        ),
        "timestamp": "2025-01-02 09:15:00",
    },
    {
        "id": 2,
        "sender": "Priya Nair",
        "sender_email": "priya.nair@mycompany.com",
        "subject": "Leave Application – 5 Days Medical Leave",
        "body": (
            "Dear Manager,\n\n"
            "I am Priya Nair from the Finance team. I would like to request 5 days of medical leave "
            "from January 10 to January 14, 2025 due to a planned surgery. I have attached the doctor's "
            "certificate for your reference.\n\n"
            "I will ensure all pending work is completed before I leave. Please approve my request at the earliest.\n\n"
            "Thank you,\nPriya Nair"
        ),
        "timestamp": "2025-01-03 10:30:00",
    },
    {
        "id": 3,
        "sender": "Angry Customer",
        "sender_email": "angryjohn99@gmail.com",
        "subject": "URGENT: Terrible Service – Demand Immediate Refund",
        "body": (
            "To Whom It May Concern,\n\n"
            "I am absolutely furious! I placed order #ORD-4892 on December 20, 2024 and still haven't "
            "received my product. Your customer service has been completely unresponsive. "
            "I demand a FULL REFUND within 24 HOURS or I will be filing a complaint with the consumer "
            "court and escalating this on social media.\n\n"
            "This is completely unacceptable!\n\nJohn D'Souza"
        ),
        "timestamp": "2025-01-03 14:45:00",
    },
    {
        "id": 4,
        "sender": "Sneha Gupta",
        "sender_email": "sneha.gupta@partners.com",
        "subject": "Meeting Schedule – Q1 Strategy Review",
        "body": (
            "Hi Team,\n\n"
            "I'd like to schedule our Q1 Strategy Review meeting. Could everyone confirm availability "
            "for January 15, 2025 at 2:00 PM IST? The meeting will be held via Google Meet and is expected "
            "to last about 2 hours. Agenda includes budget review, OKR setting, and product roadmap.\n\n"
            "Please respond by January 8, 2025 so I can send the calendar invite.\n\n"
            "Thanks,\nSneha Gupta\nStrategy Lead"
        ),
        "timestamp": "2025-01-04 09:00:00",
    },
    {
        "id": 5,
        "sender": "no-reply@prizedraw2024.xyz",
        "sender_email": "winner@prizedraw2024.xyz",
        "subject": "🎉 CONGRATULATIONS! You've WON ₹50,00,000! Claim NOW!",
        "body": (
            "DEAR WINNER,\n\n"
            "YOU HAVE BEEN SELECTED AS THE LUCKY WINNER OF OUR INTERNATIONAL LOTTERY 2024! "
            "YOUR PRIZE AMOUNT IS ₹50,00,000 (FIFTY LAKHS). "
            "To claim your prize, send your FULL NAME, ADDRESS, BANK ACCOUNT NUMBER and "
            "a processing fee of ₹5,000 to prize.claim@prizedraw2024.xyz IMMEDIATELY!\n\n"
            "THIS OFFER EXPIRES IN 48 HOURS. ACT NOW!!!\n\n"
            "PRIZE COMMITTEE INTERNATIONAL"
        ),
        "timestamp": "2025-01-04 11:20:00",
    },
    {
        "id": 6,
        "sender": "Arjun Mehta",
        "sender_email": "arjun.mehta@clientco.com",
        "subject": "CRITICAL: Production Server Down – Immediate Action Required",
        "body": (
            "Hi Support Team,\n\n"
            "Our production server has been DOWN since 6:00 AM today (January 5, 2025). "
            "This is causing a complete outage for 5,000+ users of our payment system. "
            "Every minute of downtime is costing us approximately ₹2,00,000.\n\n"
            "We need your on-call engineer to respond IMMEDIATELY. Our technical lead Arjun Mehta "
            "is available at +91-9988776655. Please escalate this to the highest priority.\n\n"
            "Awaiting urgent response,\nArjun Mehta\nCTO, ClientCo"
        ),
        "timestamp": "2025-01-05 06:30:00",
    },
    {
        "id": 7,
        "sender": "Mom",
        "sender_email": "savitridevi1960@gmail.com",
        "subject": "Sunday Dinner – Don't forget!",
        "body": (
            "Beta,\n\n"
            "Just a reminder that we are having the whole family over for Sunday dinner on January 12. "
            "Your mausi and cousins will also be there. Please try to come by 1 PM. "
            "I am making your favourite Dal Baati Churma!\n\n"
            "Also please bring some sweets from Haldiram's.\n\n"
            "Love,\nMaa"
        ),
        "timestamp": "2025-01-05 20:00:00",
    },
    {
        "id": 8,
        "sender": "HR Department",
        "sender_email": "hr@mycompany.com",
        "subject": "Performance Review Scheduled – January 20, 2025",
        "body": (
            "Dear Employee,\n\n"
            "This is to inform you that your Annual Performance Review has been scheduled for "
            "January 20, 2025 at 11:00 AM in Conference Room B, 3rd Floor.\n\n"
            "Please prepare a self-evaluation form and a summary of your key achievements for 2024. "
            "The review will be conducted by your reporting manager. "
            "Kindly confirm your attendance by January 10, 2025.\n\n"
            "Best regards,\nHR Department\nMyCompany Pvt. Ltd."
        ),
        "timestamp": "2025-01-06 09:00:00",
    },
    {
        "id": 9,
        "sender": "Vikram Singh",
        "sender_email": "vikram.singh@ngohelp.org",
        "subject": "Partnership Proposal – Rural Education Initiative",
        "body": (
            "Dear Sir/Madam,\n\n"
            "My name is Vikram Singh, Program Director at NGO Help Foundation. "
            "We are working on a Rural Education Initiative to provide digital literacy to 10,000 children "
            "in Rajasthan by March 2025.\n\n"
            "We would love to partner with your organization for sponsorship and technical support. "
            "I would like to request a meeting at your earliest convenience to discuss the proposal in detail. "
            "Please reach out to me at vikram.singh@ngohelp.org or +91-8877665544.\n\n"
            "Looking forward to your positive response,\nVikram Singh"
        ),
        "timestamp": "2025-01-06 14:00:00",
    },
    {
        "id": 10,
        "sender": "Deepika Joshi",
        "sender_email": "deepika.joshi@freelance.com",
        "subject": "Invoice #INV-2025-007 – Payment Due January 15",
        "body": (
            "Dear Accounts Team,\n\n"
            "Please find attached Invoice #INV-2025-007 for the web development services rendered "
            "during December 2024. The total amount due is ₹85,000 + GST (18%) = ₹1,00,300.\n\n"
            "As per our agreement, payment is due by January 15, 2025. "
            "Please process the payment to:\nBank: HDFC Bank\nAccount: 5024XXXX1234\nIFSC: HDFC0001234\n\n"
            "For any queries, please contact me at deepika.joshi@freelance.com.\n\n"
            "Thank you,\nDeerika Joshi\nFreelance Web Developer"
        ),
        "timestamp": "2025-01-07 10:00:00",
    },
]

# Ground-truth labels for grading
GROUND_TRUTH: dict[int, dict] = {
    1: {"category": "Work", "priority": "Medium", "key_entities": ["Rahul Sharma", "January 15", "Python Developer", "resume"]},
    2: {"category": "Work", "priority": "Medium", "key_entities": ["Priya Nair", "January 10", "January 14", "medical leave"]},
    3: {"category": "Urgent", "priority": "Critical", "key_entities": ["John D'Souza", "ORD-4892", "refund", "24 hours"]},
    4: {"category": "Work", "priority": "Medium", "key_entities": ["Sneha Gupta", "January 15", "Q1 Strategy", "January 8"]},
    5: {"category": "Spam", "priority": "Low", "key_entities": []},
    6: {"category": "Urgent", "priority": "Critical", "key_entities": ["Arjun Mehta", "January 5", "production server", "payment system"]},
    7: {"category": "Personal", "priority": "Low", "key_entities": ["January 12", "Sunday dinner", "1 PM"]},
    8: {"category": "Work", "priority": "Medium", "key_entities": ["January 20", "Performance Review", "January 10"]},
    9: {"category": "Work", "priority": "Low", "key_entities": ["Vikram Singh", "March 2025", "Rural Education", "Rajasthan"]},
    10: {"category": "Work", "priority": "High", "key_entities": ["Deepika Joshi", "January 15", "INV-2025-007", "₹1,00,300"]},
}


# ─── Environment Class ─────────────────────────────────────────────────────────

class EmailEnvironment:
    """
    OpenEnv-compatible environment for email processing.

    API:
      env.reset()         → EnvironmentState
      env.step(action)    → StepResult
      env.state()         → EnvironmentState
    """

    def __init__(self, shuffle: bool = False):
        self.emails = [Email(**e) for e in SAMPLE_EMAILS]
        self.ground_truth = GROUND_TRUTH
        self.grader = Grader(self.ground_truth)
        self.shuffle = shuffle
        self._pointer = 0
        self._completed: list[int] = []
        self._total_reward = 0.0

    # ── Public OpenEnv API ──────────────────────────────────────────────────

    def reset(self) -> EnvironmentState:
        """Reset the environment to the first email."""
        if self.shuffle:
            import random
            random.shuffle(self.emails)
        self._pointer = 0
        self._completed = []
        self._total_reward = 0.0
        return self._build_state()

    def step(self, action: AgentAction) -> StepResult:
        """
        Process the agent's action and advance to the next email.

        Args:
            action: AgentAction with classification, extraction, priority, reply

        Returns:
            StepResult with reward, grader breakdown, and next state
        """
        current_email = self.emails[self._pointer]

        # Grade the action
        grader_result = self.grader.grade(action, current_email)
        self._total_reward += grader_result.total_reward

        # Mark as completed and advance
        self._completed.append(current_email.id)
        self._pointer += 1

        done = self._pointer >= len(self.emails)
        next_state = self._build_state(task_complete=done)

        return StepResult(
            observation=next_state,
            reward=grader_result.total_reward,
            done=done,
            info=grader_result,
        )

    def state(self) -> EnvironmentState:
        """Return the current observable state without advancing."""
        return self._build_state()

    def current_email(self) -> Optional[Email]:
        """Return the current email or None if exhausted."""
        if self._pointer < len(self.emails):
            return self.emails[self._pointer]
        return None

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def max_possible_reward(self) -> float:
        return len(self.emails) * 4.0  # 4.0 per email

    # ── Private Helpers ─────────────────────────────────────────────────────

    def _build_state(self, task_complete: bool = False) -> EnvironmentState:
        if self._pointer < len(self.emails):
            current = self.emails[self._pointer]
        else:
            current = self.emails[-1]  # hold last email when done
            task_complete = True

        return EnvironmentState(
            current_email=current,
            step_number=self._pointer,
            total_emails=len(self.emails),
            completed_ids=copy.copy(self._completed),
            task_complete=task_complete,
        )