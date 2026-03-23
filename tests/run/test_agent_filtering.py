"""Tests for agent filtering via GEAK_ALLOWED_AGENTS / GEAK_EXCLUDED_AGENTS."""

from __future__ import annotations

import logging

from minisweagent.agents.agent_spec import (
    ALL_AGENT_TYPES,
    filter_agent_type,
    get_allowed_agent_types,
)

# ---------------------------------------------------------------------------
# get_allowed_agent_types()
# ---------------------------------------------------------------------------


class TestGetAllowedAgentTypes:
    def test_no_env_vars_returns_none(self, monkeypatch):
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        assert get_allowed_agent_types() is None

    def test_allowed_set(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "strategy_agent")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        result = get_allowed_agent_types()
        assert result == {"strategy_agent"}

    def test_allowed_set_filters_unknown_types(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "strategy_agent,bogus_agent")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        result = get_allowed_agent_types()
        assert result == {"strategy_agent"}

    def test_excluded_set(self, monkeypatch):
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.setenv("GEAK_EXCLUDED_AGENTS", "bogus_type")
        result = get_allowed_agent_types()
        assert result == ALL_AGENT_TYPES - {"bogus_type"}
        assert result == {"strategy_agent"}

    def test_both_set_allowed_takes_precedence(self, monkeypatch, caplog):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "strategy_agent")
        monkeypatch.setenv("GEAK_EXCLUDED_AGENTS", "strategy_agent")
        with caplog.at_level(logging.WARNING):
            result = get_allowed_agent_types()
        assert result == {"strategy_agent"}
        assert "takes precedence" in caplog.text

    def test_invalid_agent_name_filtered(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "bogus_agent,another_bogus")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        result = get_allowed_agent_types()
        assert result == set()

    def test_whitespace_handling(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", " strategy_agent ")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        result = get_allowed_agent_types()
        assert result == {"strategy_agent"}


# ---------------------------------------------------------------------------
# filter_agent_type()
# ---------------------------------------------------------------------------


class TestFilterAgentType:
    def test_no_env_vars_passthrough(self, monkeypatch):
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        for agent in ("strategy_agent", "some_unknown_type"):
            assert filter_agent_type(agent) == agent

    def test_allowed_agents_pass_through(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "strategy_agent")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_FALLBACK_AGENT", raising=False)
        assert filter_agent_type("strategy_agent") == "strategy_agent"

    def test_disallowed_agent_remapped_to_fallback(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "strategy_agent")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_FALLBACK_AGENT", raising=False)
        assert filter_agent_type("unknown_type") == "strategy_agent"

    def test_excluded_agent_remapped(self, monkeypatch):
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.setenv("GEAK_EXCLUDED_AGENTS", "strategy_agent")
        monkeypatch.delenv("GEAK_FALLBACK_AGENT", raising=False)
        result = filter_agent_type("strategy_agent")
        # strategy_agent is excluded, but it's the only type; fallback logic applies
        assert isinstance(result, str)

    def test_custom_fallback_agent(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "strategy_agent")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        monkeypatch.setenv("GEAK_FALLBACK_AGENT", "strategy_agent")
        assert filter_agent_type("unknown_type") == "strategy_agent"

    def test_remap_logs_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "strategy_agent")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_FALLBACK_AGENT", raising=False)
        with caplog.at_level(logging.WARNING):
            result = filter_agent_type("unknown_type")
        assert result == "strategy_agent"
        assert "not allowed" in caplog.text
        assert "unknown_type" in caplog.text


# ---------------------------------------------------------------------------
# _parse_llm_response integration (safety net)
# ---------------------------------------------------------------------------


class TestParseResponseSafetyNet:
    """Verify that _parse_llm_response applies filter_agent_type."""

    TASK_JSON_WITH_UNKNOWN_TYPE = (
        '[{"label": "oe-task", "priority": 0, "agent_type": "unknown_type", '
        '"kernel_language": "python", "task_prompt": "Run optimization"}]'
    )

    def test_unknown_agent_type_remapped_in_parse(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "strategy_agent")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_FALLBACK_AGENT", raising=False)

        from minisweagent.agents.heterogeneous.task_generator import _parse_llm_response
        from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent

        class _FakeDefault:
            pass

        tasks = _parse_llm_response(self.TASK_JSON_WITH_UNKNOWN_TYPE, _FakeDefault)
        assert len(tasks) == 1
        # unknown_type remapped to strategy_agent via filter_agent_type,
        # then resolved to StrategyInteractiveAgent via _agent_type_to_class()
        assert tasks[0].agent_class is StrategyInteractiveAgent

    def test_fallback_not_in_allowed_set(self, monkeypatch):
        """Test that fallback agent is validated against allowed set (Issue #20)."""
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "strategy_agent")
        monkeypatch.setenv("GEAK_FALLBACK_AGENT", "bogus_type")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)

        result = filter_agent_type("unknown_type")
        assert result != "bogus_type"
        assert result in get_allowed_agent_types()
        assert result == "strategy_agent"


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------


class TestPromptInjection:
    def test_allowed_agents_adds_restriction(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "strategy_agent")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)

        from minisweagent.agents.heterogeneous.task_generator import _build_agent_restriction_addendum

        addendum = _build_agent_restriction_addendum()
        assert "strategy_agent" in addendum
        assert "MUST NOT" in addendum
        assert "only" in addendum.lower() or "Only" in addendum

    def test_excluded_agents_adds_restriction(self, monkeypatch):
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.setenv("GEAK_EXCLUDED_AGENTS", "strategy_agent")

        from minisweagent.agents.heterogeneous.task_generator import _build_agent_restriction_addendum

        addendum = _build_agent_restriction_addendum()
        assert "strategy_agent" in addendum
        assert "NOT available" in addendum

    def test_no_env_vars_returns_empty(self, monkeypatch):
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)

        from minisweagent.agents.heterogeneous.task_generator import _build_agent_restriction_addendum

        assert _build_agent_restriction_addendum() == ""
