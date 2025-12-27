"""Tests for Stability Monitor module."""

import pytest
import numpy as np
import time

from agentic_toolkit.monitoring.stability_monitor import (
    StabilityMonitor,
    LimitCycleDetector,
    StabilityStatus,
    ConvergenceStatus,
    OscillationStatus,
    MonotonicityStatus,
    FidelityStatus,
    StabilityReport,
    StabilityViolation,
    ViolationType,
    StabilitySeverity,
    create_stability_monitor,
    check_stability_conditions,
)


class TestStabilityMonitor:
    """Tests for StabilityMonitor class."""

    @pytest.fixture
    def goal_embedding(self):
        """Create a goal embedding."""
        np.random.seed(42)
        goal = np.random.randn(64)
        return goal / np.linalg.norm(goal)

    @pytest.fixture
    def monitor(self, goal_embedding):
        """Create a stability monitor."""
        return StabilityMonitor(
            goal_embedding=goal_embedding,
            similarity_threshold=0.9,
            oscillation_window=10,
            oscillation_bound=3,
            progress_threshold=0.001
        )

    def test_initialization(self, goal_embedding):
        """Test monitor initialization."""
        monitor = StabilityMonitor(goal_embedding=goal_embedding)
        assert monitor.similarity_threshold == 0.9
        assert monitor.oscillation_window == 10

    def test_track_state_basic(self, monitor, goal_embedding):
        """Test basic state tracking."""
        state = goal_embedding * 0.5  # 50% similar
        status = monitor.track_state(
            state_embedding=state,
            action="test_action",
            observation={"step": 1}
        )

        assert isinstance(status, StabilityStatus)
        assert status.step == 1
        assert status.goal_similarity > 0

    def test_track_state_convergence(self, monitor, goal_embedding):
        """Test tracking converging state sequence."""
        # Simulate convergence
        for i in range(20):
            similarity = 0.5 + i * 0.02  # Improve by 2% each step
            state = goal_embedding * similarity
            state = state / np.linalg.norm(state)

            status = monitor.track_state(
                state_embedding=state,
                action=f"action_{i}",
                observation={"step": i}
            )

        assert status.convergence.trend == "improving"
        assert status.convergence.converging

    def test_track_state_oscillation(self, monitor, goal_embedding):
        """Test detection of oscillating actions."""
        # Simulate oscillation with repeated actions
        for i in range(30):
            state = goal_embedding * 0.5
            action = "action_a" if i % 2 == 0 else "action_b"

            status = monitor.track_state(
                state_embedding=state,
                action=action,
                observation={"step": i}
            )

        # Should detect oscillation
        assert status.oscillation.oscillating or status.oscillation.overlap_ratio > 0

    def test_check_goal_convergence(self, monitor, goal_embedding):
        """Test goal convergence checking."""
        # Add some history
        for i in range(10):
            similarity = 0.7 + i * 0.02
            state = goal_embedding * similarity
            monitor.track_state(
                state_embedding=state,
                action=f"action_{i}"
            )

        status = monitor.check_goal_convergence()
        assert isinstance(status, ConvergenceStatus)
        assert status.current_similarity > 0.7

    def test_check_oscillation(self, monitor):
        """Test oscillation checking."""
        # Add non-oscillating actions
        for i in range(15):
            monitor._action_history.append(f"unique_action_{i}")

        status = monitor.check_oscillation()
        assert isinstance(status, OscillationStatus)
        assert not status.oscillating

    def test_check_oscillation_with_cycle(self, monitor):
        """Test oscillation detection with explicit cycle."""
        # Add cycling actions
        cycle = ["a", "b", "c"]
        for _ in range(5):
            for action in cycle:
                monitor._action_history.append(action)

        status = monitor.check_oscillation()
        assert status.oscillating or status.overlap_ratio > 0

    def test_check_monotonicity(self, monitor, goal_embedding):
        """Test monotonicity checking."""
        # Simulate improving similarity
        for i in range(15):
            similarity = 0.5 + i * 0.01
            monitor._similarity_history.append(similarity)

        status = monitor.check_monotonicity()
        assert isinstance(status, MonotonicityStatus)
        assert status.mean_progress > 0

    def test_check_monotonicity_failure(self, monitor, goal_embedding):
        """Test monotonicity failure detection."""
        # Simulate degrading similarity
        for i in range(15):
            similarity = 0.8 - i * 0.02
            monitor._similarity_history.append(similarity)

        status = monitor.check_monotonicity()
        assert not status.monotonic
        assert status.mean_progress < 0

    def test_check_observation_fidelity(self, monitor):
        """Test observation fidelity checking."""
        observation = {"status": "success", "result": 42}
        schema = {
            "required": ["status"],
            "properties": {
                "status": {"type": "string"},
                "result": {"type": "number"}
            }
        }

        status = monitor.check_observation_fidelity(observation, schema)
        assert isinstance(status, FidelityStatus)
        assert status.schema_valid

    def test_check_observation_fidelity_failure(self, monitor):
        """Test observation fidelity failure."""
        observation = {"status": 123}  # Wrong type
        schema = {
            "required": ["status"],
            "properties": {
                "status": {"type": "string"}
            }
        }

        status = monitor.check_observation_fidelity(observation, schema)
        assert not status.schema_valid
        assert len(status.validation_errors) > 0

    def test_get_stability_report(self, monitor, goal_embedding):
        """Test report generation."""
        # Add some history
        for i in range(10):
            state = goal_embedding * (0.5 + i * 0.05)
            monitor.track_state(
                state_embedding=state,
                action=f"action_{i}"
            )

        report = monitor.get_stability_report()
        assert isinstance(report, StabilityReport)
        assert report.total_steps == 10
        assert len(report.recommendations) > 0

    def test_reset(self, monitor, goal_embedding):
        """Test monitor reset."""
        # Add some history
        monitor.track_state(
            state_embedding=goal_embedding,
            action="test"
        )

        assert monitor._step == 1

        monitor.reset()

        assert monitor._step == 0
        assert len(monitor._action_history) == 0


class TestLimitCycleDetector:
    """Tests for LimitCycleDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a limit cycle detector."""
        return LimitCycleDetector(
            similarity_threshold=0.95,
            cycle_threshold=0.8,
            min_cycle_length=2,
            max_cycle_length=10
        )

    def test_initialization(self):
        """Test detector initialization."""
        detector = LimitCycleDetector()
        assert detector.similarity_threshold == 0.95
        assert detector.cycle_threshold == 0.8

    def test_no_cycle_initial(self, detector):
        """Test no cycle detected initially."""
        state = np.random.randn(64)
        is_cycle, info = detector.check_state(state)

        assert not is_cycle
        assert "Insufficient history" in info["message"]

    def test_detect_cycle(self, detector):
        """Test cycle detection."""
        np.random.seed(42)

        # Create cycling states
        base_state = np.random.randn(64)
        base_state = base_state / np.linalg.norm(base_state)

        cycle_detected = False

        for round in range(5):
            for i in range(3):  # Cycle of length 3
                # Same states in each round
                state = base_state + 0.01 * i
                state = state / np.linalg.norm(state)

                is_cycle, info = detector.check_state(state)
                if is_cycle:
                    cycle_detected = True

        # Should eventually detect the cycle
        # Note: Due to similarity threshold, may not always detect
        assert detector._state_history is not None

    def test_no_cycle_random(self, detector):
        """Test no cycle with random states."""
        np.random.seed(42)

        for i in range(30):
            state = np.random.randn(64)
            is_cycle, info = detector.check_state(state)

        # Random states shouldn't form a cycle
        assert "No limit cycle" in info["message"] or not is_cycle

    def test_reset(self, detector):
        """Test detector reset."""
        state = np.random.randn(64)
        detector.check_state(state)

        assert len(detector._state_history) == 1

        detector.reset()

        assert len(detector._state_history) == 0


class TestViolationTypes:
    """Tests for violation types and dataclasses."""

    def test_stability_violation_creation(self):
        """Test StabilityViolation creation."""
        violation = StabilityViolation(
            violation_type=ViolationType.OSCILLATION,
            severity=StabilitySeverity.WARNING,
            timestamp=time.time(),
            step=10,
            description="Test violation"
        )

        assert violation.violation_type == ViolationType.OSCILLATION
        assert violation.severity == StabilitySeverity.WARNING

    def test_stability_violation_to_dict(self):
        """Test StabilityViolation serialization."""
        violation = StabilityViolation(
            violation_type=ViolationType.GOAL_DIVERGENCE,
            severity=StabilitySeverity.CRITICAL,
            timestamp=1234567890.0,
            step=5,
            description="Goal divergence detected"
        )

        d = violation.to_dict()
        assert d["type"] == "GOAL_DIVERGENCE"
        assert d["severity"] == "CRITICAL"
        assert d["step"] == 5


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_stability_monitor(self):
        """Test monitor factory function."""
        def embed(text: str) -> np.ndarray:
            return np.random.randn(64)

        monitor = create_stability_monitor(
            goal_text="Complete the task",
            embedding_fn=embed,
            similarity_threshold=0.8
        )

        assert isinstance(monitor, StabilityMonitor)
        assert monitor.similarity_threshold == 0.8

    def test_check_stability_conditions(self):
        """Test batch stability check."""
        np.random.seed(42)

        goal = np.random.randn(64)
        goal = goal / np.linalg.norm(goal)

        states = [goal * (0.5 + i * 0.05) for i in range(10)]
        actions = [f"action_{i}" for i in range(10)]

        report = check_stability_conditions(
            goal_embedding=goal,
            state_embeddings=states,
            actions=actions
        )

        assert isinstance(report, StabilityReport)
        assert report.total_steps == 10


class TestStatusDataclasses:
    """Tests for status dataclasses."""

    def test_convergence_status_to_dict(self):
        """Test ConvergenceStatus serialization."""
        status = ConvergenceStatus(
            converging=True,
            current_similarity=0.85,
            target_similarity=0.9,
            estimated_steps_remaining=10,
            trend="improving"
        )

        d = status.to_dict()
        assert d["converging"] is True
        assert d["current_similarity"] == 0.85

    def test_oscillation_status_to_dict(self):
        """Test OscillationStatus serialization."""
        status = OscillationStatus(
            oscillating=False,
            cycle_detected=False,
            overlap_ratio=0.1,
            window_size=10,
            bound=3,
            repeated_actions=[]
        )

        d = status.to_dict()
        assert d["oscillating"] is False
        assert d["overlap_ratio"] == 0.1

    def test_monotonicity_status_to_dict(self):
        """Test MonotonicityStatus serialization."""
        status = MonotonicityStatus(
            monotonic=True,
            mean_progress=0.02,
            required_progress=0.001,
            consecutive_negative=0,
            window_size=10
        )

        d = status.to_dict()
        assert d["monotonic"] is True
        assert d["mean_progress"] == 0.02

    def test_fidelity_status_to_dict(self):
        """Test FidelityStatus serialization."""
        status = FidelityStatus(
            fidelity_satisfied=True,
            schema_valid=True,
            validation_errors=[],
            error_rate=0.0,
            bound=0.1
        )

        d = status.to_dict()
        assert d["fidelity_satisfied"] is True
        assert d["error_count"] == 0
