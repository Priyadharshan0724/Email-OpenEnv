"""
models.py - Pydantic data schemas for Email-OpenEnv
Defines all structured data types used across the system.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class EmailCategory(str, Enum):
    WORK = "Work"
    PERSONAL = "Personal"
    SPAM = "Spam"
    URGENT = "Urgent"


class PriorityLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# ─── Core Email Schema ─────────────────────────────────────────────────────────

class Email(BaseModel):
    """Represents a raw incoming email."""
    id: int
    sender: str
    sender_email: str
    subject: str
    body: str
    timestamp: str

    class Config:
        use_enum_values = True


# ─── Agent Action Schema ───────────────────────────────────────────────────────

class ExtractedEntities(BaseModel):
    """Information extracted from the email body."""
    name: Optional[str] = None
    date: Optional[str] = None
    deadline: Optional[str] = None
    request: Optional[str] = None
    organization: Optional[str] = None
    contact: Optional[str] = None


class AgentAction(BaseModel):
    """
    The complete structured action submitted by the AI agent.
    Contains classification, extraction, priority, and reply.
    """
    email_id: int
    category: EmailCategory
    priority: PriorityLevel
    extracted_entities: ExtractedEntities
    reply: str
    reasoning: Optional[str] = None  # optional chain-of-thought


# ─── Environment State Schema ──────────────────────────────────────────────────

class EnvironmentState(BaseModel):
    """Current observable state of the environment."""
    current_email: Email
    step_number: int
    total_emails: int
    completed_ids: List[int] = Field(default_factory=list)
    task_complete: bool = False


# ─── Step Result Schema ────────────────────────────────────────────────────────

class GraderResult(BaseModel):
    """Detailed scoring breakdown from the grader."""
    classification_score: float   # 0.0 - 1.0
    extraction_score: float       # 0.0 - 1.0
    reply_score: float            # 0.0 - 1.0
    priority_score: float         # 0.0 - 1.0
    total_reward: float           # 0.0 - 4.0 (sum)
    feedback: str                 # human-readable explanation


class StepResult(BaseModel):
    """Result returned after the agent takes a step."""
    observation: EnvironmentState
    reward: float
    done: bool
    info: GraderResult