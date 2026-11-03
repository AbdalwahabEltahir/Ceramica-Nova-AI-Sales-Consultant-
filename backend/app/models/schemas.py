"""Pydantic v2 entities (FR-1 Requirements schema, quote/session shapes)."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Budget(BaseModel):
    amount: float
    currency: str = 'USD'
    basis: str = 'total'  # total | per_m2


class Requirements(BaseModel):
    intent: str = 'product_search'
    language: str = 'en'  # en | ar | mixed
    application: Optional[str] = None
    indoor_outdoor: Optional[str] = None
    floor_or_wall: Optional[str] = None
    area_m2: Optional[float] = None
    budget: Optional[Budget] = None
    preferred_style: Optional[str] = None
    preferred_color: Optional[str] = None
    preferred_finish: Optional[str] = None
    customer_notes: List[str] = Field(default_factory=list)
    slip_requirement: Optional[str] = None
    traffic_level: Optional[str] = None
    urgency: Optional[str] = None
    signals: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    clarifying_questions: List[str] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)

    model_config = {'extra': 'ignore'}


class GenerateRequest(BaseModel):
    """POST /api/v1/generate request body (exam B1, literal field names)."""
    user_id: str
    task: str
    message: str
    language: str = 'en'

    model_config = {'extra': 'forbid'}


class CustomerRequirements(BaseModel):
    """Nested object inside GenerateResponse (exam B2, literal field names —
    note this is a narrower, exam-specific shape than models.schemas.Requirements,
    which is the richer internal FR-1 schema used by the chat orchestrator)."""
    application: str = ''
    area_m2: Optional[float] = None
    budget: Optional[float] = None
    style: str = ''
    color: str = ''
    urgency: str = ''


class GenerateResponse(BaseModel):
    """POST /api/v1/generate response body (exam B2, literal field names)."""
    intent: str = ''
    customer_requirements: CustomerRequirements = Field(default_factory=CustomerRequirements)
    missing_information: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    confidence: float = 0.0


# JSON Schema (not a Pydantic model) handed to Gemini's structured-output
# config — kept hand-written and minimal so it's obvious what the model is
# actually constrained to, independent of Pydantic's schema generation.
GENERATE_RESPONSE_JSON_SCHEMA = {
    'type': 'object',
    'properties': {
        'intent': {'type': 'string'},
        'customer_requirements': {
            'type': 'object',
            'properties': {
                'application': {'type': 'string'},
                'area_m2': {'type': ['number', 'null']},
                'budget': {'type': ['number', 'null']},
                'style': {'type': 'string'},
                'color': {'type': 'string'},
                'urgency': {'type': 'string'},
            },
            'required': ['application', 'area_m2', 'budget', 'style', 'color', 'urgency'],
        },
        'missing_information': {'type': 'array', 'items': {'type': 'string'}},
        'recommended_actions': {'type': 'array', 'items': {'type': 'string'}},
        'confidence': {'type': 'number'},
    },
    'required': ['intent', 'customer_requirements', 'missing_information',
                 'recommended_actions', 'confidence'],
}


class QuoteLineInput(BaseModel):
    sku: str
    area_m2: Optional[float] = None
    boxes: Optional[int] = None
    layout: str = 'simple'


class ToolCallRecord(BaseModel):
    tool_call_id: str
    tool_name: str
    arguments: dict
    result: dict
