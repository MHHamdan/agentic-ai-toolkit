"""Safety and compliance schemas for the dashboard API."""
from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime
from enum import Enum


class AutonomyLevel(str, Enum):
    """Autonomy levels (from paper Table IV)."""

    STATIC_WORKFLOW = "static_workflow"  # 0/4 criteria
    CONDITIONAL = "conditional"  # 1/4 criteria
    GUIDED = "guided"  # 2/4 criteria
    BOUNDED = "bounded"  # 3/4 criteria
    FULL_AGENT = "full_agent"  # 4/4 criteria


class AutonomyCriteria(BaseModel):
    """Four autonomy criteria (Section IV-A)."""

    action_selection_freedom: bool = Field(
        default=False, description="ASF: Varies actions based on state"
    )
    goal_directed_persistence: bool = Field(
        default=False, description="GDP: Continues despite obstacles"
    )
    dynamic_termination: bool = Field(
        default=False, description="DT: Stops based on goal, not fixed steps"
    )
    error_recovery: bool = Field(
        default=False, description="ER: Adapts strategy after failures"
    )

    def get_level(self) -> AutonomyLevel:
        """Determine autonomy level from criteria."""
        count = sum(
            [
                self.action_selection_freedom,
                self.goal_directed_persistence,
                self.dynamic_termination,
                self.error_recovery,
            ]
        )
        levels = {
            0: AutonomyLevel.STATIC_WORKFLOW,
            1: AutonomyLevel.CONDITIONAL,
            2: AutonomyLevel.GUIDED,
            3: AutonomyLevel.BOUNDED,
            4: AutonomyLevel.FULL_AGENT,
        }
        return levels.get(count, AutonomyLevel.STATIC_WORKFLOW)


class SafetyRequirement(BaseModel):
    """A single safety requirement check."""

    requirement_id: str
    name: str
    description: str
    passed: bool
    details: str = ""


class ComplianceStatus(BaseModel):
    """Overall compliance status (5 requirements from paper)."""

    overall_compliant: bool = False
    compliance_score: float = Field(description="0.0 to 1.0")

    # 5 Safety Requirements
    requirements: List[SafetyRequirement] = Field(default_factory=list)

    # Autonomy assessment
    autonomy_level: AutonomyLevel = AutonomyLevel.STATIC_WORKFLOW
    autonomy_criteria: AutonomyCriteria = Field(default_factory=AutonomyCriteria)

    # Metadata
    last_checked: datetime = Field(default_factory=datetime.now)
    recommendation: str = "PENDING"

    @classmethod
    def get_default_requirements(cls) -> List[SafetyRequirement]:
        """Get the 5 default safety requirements."""
        return [
            SafetyRequirement(
                requirement_id="SR1",
                name="Intent Alignment Verification",
                description="Agent actions align with user intent",
                passed=False,
            ),
            SafetyRequirement(
                requirement_id="SR2",
                name="Bounded Authority Enforcement",
                description="Agent operates within defined boundaries",
                passed=False,
            ),
            SafetyRequirement(
                requirement_id="SR3",
                name="Robust Termination Guarantee",
                description="Agent can be safely terminated",
                passed=False,
            ),
            SafetyRequirement(
                requirement_id="SR4",
                name="Full Auditability",
                description="All actions are logged and traceable",
                passed=False,
            ),
            SafetyRequirement(
                requirement_id="SR5",
                name="Graceful Degradation",
                description="Agent fails safely under stress",
                passed=False,
            ),
        ]
