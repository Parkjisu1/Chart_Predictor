"""Base agent class and Claude CLI runner."""

from __future__ import annotations

import json
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)


@dataclass
class AgentResult:
    """Standard result from any agent."""
    agent_name: str
    signal_value: float = 0.0      # -1.0 to 1.0
    confidence: float = 0.0        # 0.0 to 1.0
    reasoning: str = ""
    details: dict = field(default_factory=dict)
    error: str | None = None
    execution_time: float = 0.0


class ClaudeCLIRunner:
    """Wrapper for calling Claude CLI as a subprocess."""

    def __init__(self):
        settings = get_settings()
        self.cli_path = settings.claude.cli_path
        self.timeout = settings.claude.timeout

    def run(self, prompt: str) -> dict | None:
        """Run Claude CLI with a prompt and parse JSON response."""
        start = time.time()
        try:
            result = subprocess.run(
                [self.cli_path, "-p", prompt, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = time.time() - start
            logger.info("claude_cli_call", elapsed=round(elapsed, 2),
                        returncode=result.returncode)

            if result.returncode != 0:
                logger.warning("claude_cli_error", stderr=result.stderr[:200])
                return None

            output = result.stdout.strip()
            return self._parse_json(output)

        except subprocess.TimeoutExpired:
            logger.warning("claude_cli_timeout", timeout=self.timeout)
            return None
        except FileNotFoundError:
            logger.error("claude_cli_not_found", path=self.cli_path)
            return None
        except Exception as e:
            logger.error("claude_cli_exception", error=str(e))
            return None

    def _parse_json(self, text: str) -> dict | None:
        """Extract JSON from Claude CLI output (may include markdown)."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code blocks
        for marker in ["```json", "```"]:
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start)
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass

        # Try finding JSON object in text
        for i, ch in enumerate(text):
            if ch == "{":
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == "{":
                        depth += 1
                    elif text[j] == "}":
                        depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i:j + 1])
                        except json.JSONDecodeError:
                            break
                break

        logger.warning("claude_cli_parse_failed", text=text[:200])
        return None


class AgentBase(ABC):
    """Base class for all agents."""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"agent.{name}")

    @abstractmethod
    def analyze(self, **kwargs) -> AgentResult:
        """Run the agent's analysis. Must be implemented by subclasses."""
        ...

    def _make_result(self, **kwargs) -> AgentResult:
        return AgentResult(agent_name=self.name, **kwargs)
