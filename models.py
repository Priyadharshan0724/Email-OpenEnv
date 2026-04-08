"""
Pydantic models for Email-OpenEnv environment
Defines schemas for email data, actions, and states
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class EmailCategoryEnum(str, Enum):
    """Email classification categories"""
    WORK = "work"
    PERSONAL = "personal"
    SPAM = "spam"
    URGENT = "urgent"


class UrgencyLevelEnum(str, Enum):
    """Urgency levels for emails"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Email(BaseModel):
    """Email data model"""
    id: str = Field(..., description="Unique email identifier")
    sender: str = Field(..., description="Email sender address")
    sender_name: str = Field(..., description="Sender's name")
    subject: str = Field(..., description="Email subject")
    content: str = Field(..., description="Email body content")
    timestamp: str = Field(..., description="Email received timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "email_001",
                "sender": "john.doe@company.com",
                "sender_name": "John Doe",
                "subject": "Project Update",
                "content": "Hi, please review the attached project proposal...",
                "timestamp": "2026-04-08T10:30:00"
            }
        }


class EmailState(BaseModel):
    """State representation of an email for the agent"""
    email_id: str
    sender: str
    sender_name: str
    subject: str
    content: str
    timestamp: str
    extracted_entities: Optional[Dict[str, Any]] = None
    urgency_level: Optional[UrgencyLevelEnum] = None
    category: Optional[EmailCategoryEnum] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "email_id": "email_001",
                "sender": "john.doe@company.com",
                "sender_name": "John Doe",
                "subject": "Project Update",
                "content": "Hi, please review the attached project proposal...",
                "timestamp": "2026-04-08T10:30:00",
                "extracted_entities": {
                    "names": ["John Doe"],
                    "dates": ["2026-04-15"],
                    "requests": ["review project proposal"]
                },
                "urgency_level": "high",
                "category": "work"
            }
        }


class ClassificationAction(BaseModel):
    """Action: Classify email"""
    action_type: str = "classify"
    category: EmailCategoryEnum
    confidence: float = Field(..., ge=0.0, le=1.0)


class ExtractionAction(BaseModel):
    """Action: Extract information from email"""
    action_type: str = "extract"
    sender_name: str
    requested_date: Optional[str] = None
    key_request: str
    deadline: Optional[str] = None
    entities: Dict[str, Any]


class ReplyGenerationAction(BaseModel):
    """Action: Generate email reply"""
    action_type: str = "generate_reply"
    reply_subject: str
    reply_content: str
    sender_email: str


class PriorityAction(BaseModel):
    """Action: Mark email priority"""
    action_type: str = "prioritize"
    urgency_level: UrgencyLevelEnum
    reason: str


class ActionUnion(BaseModel):
    """Union of all possible actions"""
    classification: Optional[ClassificationAction] = None
    extraction: Optional[ExtractionAction] = None
    reply: Optional[ReplyGenerationAction] = None
    priority: Optional[PriorityAction] = None


class StepResult(BaseModel):
    """Result of a step in the environment"""
    success: bool
    reward: float = Field(..., ge=-1.0, le=1.0)
    message: str
    state: Optional[EmailState] = None
    scores: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "reward": 0.85,
                "message": "Email processed successfully",
                "state": {
                    "email_id": "email_001",
                    "sender": "john.doe@company.com",
                    "sender_name": "John Doe",
                    "subject": "Project Update",
                    "content": "Hi, please review...",
                    "timestamp": "2026-04-08T10:30:00",
                    "category": "work",
                    "urgency_level": "high"
                },
                "scores": {
                    "classification_accuracy": 1.0,
                    "extraction_correctness": 0.8,
                    "reply_quality": 0.85
                }
            }
        }
}