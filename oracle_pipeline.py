"""
Oracle-X Pipeline Runner

Unified pipeline execution supporting multiple modes with clean architecture.
"""

import inspect
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple

# Core imports
from oracle_engine.agent import oracle_agent_pipeline

# Optional imports with fallbacks
try:
    from data_feeds.data_feed_orchestrator import DataFeedOrchestrator

    orchestrator_available = True
except ImportError:
    DataFeedOrchestrator = None
    orchestrator_available = False

try:
    from oracle_engine.agent_optimized import get_optimized_agent

    optimization_available = True
except ImportError:
    optimization_available = False

try:
    from learning_system.unified_learning_orchestrator import (
        UnifiedLearningOrchestrator,
        UnifiedLearningConfig,
    )

    advanced_learning_available = True
except ImportError:
    UnifiedLearningOrchestrator = None
    UnifiedLearningConfig = None
    advanced_learning_available = False


DEFAULT_PROMPT = (
    "Generate comprehensive trading scenarios based on current market conditions"
)


class OracleXPipeline:
    """Unified Oracle-X pipeline supporting multiple execution modes."""

    def __init__(self, mode: str = "standard", config: Optional[Dict] = None):
        self.mode = mode
        self.config = config or {}
        self._optimized_agent = None
        self.orchestrator = self._init_orchestrator()
        self.learning_orchestrator = self._init_learning_orchestrator()
        self.mode_handlers = self._build_mode_handlers()

    def _build_mode_handlers(self) -> Dict[str, Callable[[], Optional[str]]]:
        """Map modes to runner callables for consistent dispatch."""
        return {
            "standard": self.run_standard_pipeline,
            "optimized": self.run_optimized_pipeline,
            "advanced": self.run_advanced_pipeline,
        }

    def _prompt_for_mode(self, mode: str) -> str:
        """Resolve prompt with optional per-mode overrides."""
        prompts = self.config.get("prompts") or {}
        if isinstance(prompts, dict) and mode in prompts and prompts[mode]:
            return prompts[mode]
        if self.config.get("prompt"):
            return str(self.config["prompt"])
        return DEFAULT_PROMPT

    def _build_playbook_path(self, mode: str) -> str:
        """Consistent playbook filename generation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"playbooks/{mode}_playbook_{timestamp}.json"

    def _init_orchestrator(self) -> Optional[DataFeedOrchestrator]:
        """Initialize data feed orchestrator."""
        if not orchestrator_available:
            print("⚠️  Data feed orchestrator not available")
            return None

        try:
            return DataFeedOrchestrator()
        except Exception as e:
            print(f"⚠️  Data feed orchestrator failed: {e}")
            return None

    def _init_learning_orchestrator(self) -> Optional[UnifiedLearningOrchestrator]:
        """Initialize advanced learning orchestrator."""
        if not advanced_learning_available or not self.config.get(
            "enable_advanced_learning", True
        ):
            print("⚠️  Advanced learning orchestrator not available")
            return None

        try:
            config = UnifiedLearningConfig()
            if "learning_config" in self.config:
                for key, value in self.config["learning_config"].items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            return UnifiedLearningOrchestrator(config)
        except Exception as e:
            print(f"⚠️  Learning orchestrator failed: {e}")
            return None

    def _run_pipeline(
        self,
        mode: str,
        generator: Callable[[], Tuple[str, Optional[Dict[str, Any]]]],
        prompt: str,
        chart_image_b64: Optional[str] = None,
    ) -> Optional[str]:
        """Shared pipeline execution harness."""
        start_time = time.time()
        playbook, metadata = generator()

        if not playbook:
            print("❌ Pipeline returned empty playbook")
            return None

        execution_time = time.time() - start_time
        filename = self._build_playbook_path(mode)

        output = {
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": execution_time,
            "pipeline_mode": mode,
            "prompt": prompt,
            "chart_attached": bool(chart_image_b64),
            "playbook": playbook,
        }
        if metadata:
            output["metadata"] = metadata

        return self._save_results(filename, output)

    def _save_results(self, filename: str, data: Dict[str, Any]) -> Optional[str]:
        """Save pipeline results."""
        try:
            Path("playbooks").mkdir(exist_ok=True)
            with open(filename, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"📁 Results saved to: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            return None

    def _run_agent_mode(
        self,
        mode: str,
        generator: Callable[[str], Tuple[str, Optional[Dict[str, Any]]]],
    ) -> Optional[str]:
        """Execute an agent-driven mode with consistent prompt handling."""
        prompt = self._prompt_for_mode(mode)
        return self._run_pipeline(mode, lambda: generator(prompt), prompt)

    def _generate_standard(self, prompt: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Standard pipeline generator wrapper."""
        return oracle_agent_pipeline(prompt, None), None

    def _generate_optimized(
        self, prompt: str, agent: Any
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Optimized pipeline generator wrapper."""
        return agent.oracle_agent_pipeline_optimized(
            prompt, None, enable_experiments=True
        )

    def _get_optimized_agent(self) -> Optional[Any]:
        """Load and cache optimized agent if available."""
        if not optimization_available:
            return None
        if self._optimized_agent is not None:
            return self._optimized_agent
        try:
            self._optimized_agent = get_optimized_agent()
        except Exception as e:
            print(f"❌ Failed to load optimized agent: {e}")
            self._optimized_agent = None
        return self._optimized_agent

    def run_standard_pipeline(self) -> Optional[str]:
        """Run the standard Oracle-X pipeline."""
        print("🚀 Starting Oracle-X Standard Pipeline...")

        return self._run_agent_mode("standard", self._generate_standard)

    def run_optimized_pipeline(self) -> Optional[str]:
        """Run the optimized Oracle-X pipeline."""
        agent = self._get_optimized_agent()
        if agent is None:
            print("❌ Optimization system not available, falling back to standard mode")
            return self.run_standard_pipeline()

        print("🚀 Starting Oracle-X Optimized Pipeline...")

        try:
            return self._run_agent_mode(
                "optimized", lambda prompt: self._generate_optimized(prompt, agent)
            )
        except Exception as e:
            print(f"❌ Optimized pipeline failed: {e}")
            return self.run_standard_pipeline()

    async def run_advanced_pipeline(self) -> Optional[str]:
        """Run advanced pipeline with learning orchestrator."""
        if not self.learning_orchestrator:
            print("❌ Advanced learning system not available")
            return None

        print("🚀 Starting Oracle-X Advanced Learning Pipeline...")

        try:
            start_time = time.time()

            # Initialize and run learning pipeline
            await self.learning_orchestrator.initialize_systems()
            self.learning_orchestrator.start_real_time_learning()

            market_data = {}
            if self.orchestrator:
                try:
                    signals = self.orchestrator.get_signals_from_scrapers(
                        ["AAPL", "TSLA", "NVDA"]
                    )
                    market_data = signals
                except Exception as e:
                    print(f"⚠️  Market data collection failed: {e}")

            processing_results = self.learning_orchestrator.process_market_data(
                market_data
            )
            trading_decision = self.learning_orchestrator.make_trading_decision(
                market_data
            )
            performance_report = (
                self.learning_orchestrator.generate_performance_report()
            )
            explanations = self.learning_orchestrator.get_model_explanations(
                market_data
            )
            system_status = self.learning_orchestrator.get_system_status()

            execution_time = time.time() - start_time
            filename = self._build_playbook_path("advanced")

            output = {
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": execution_time,
                "pipeline_mode": "advanced",
                "market_data": market_data,
                "processing_results": processing_results,
                "trading_decision": trading_decision,
                "performance_report": performance_report,
                "model_explanations": explanations,
                "system_status": system_status,
            }

            return self._save_results(filename, output)

        except Exception as e:
            print(f"❌ Advanced pipeline failed: {e}")
            return None
        finally:
            if self.learning_orchestrator:
                self.learning_orchestrator.shutdown()

    def run(self) -> Optional[str]:
        """Run pipeline based on selected mode."""
        runner = self.mode_handlers.get(self.mode)
        if not runner:
            print(f"❌ Unknown mode: {self.mode}")
            return None

        if inspect.iscoroutinefunction(runner):
            import asyncio

            return asyncio.run(runner())

        return runner()


def run_oracle_pipeline(prompt_text: str) -> Dict:
    """Dashboard integration function."""
    pipeline = OracleXPipeline(mode="standard", config={"prompt": prompt_text})
    error_message = "Pipeline execution failed"
    try:
        result_file = pipeline.run()
        if result_file and os.path.exists(result_file):
            with open(result_file, "r") as f:
                results = json.load(f)
            results["date"] = datetime.now().strftime("%Y-%m-%d")
            results["logs"] = f"Pipeline executed at {datetime.now().isoformat()}"
            return results
    except Exception as e:
        error_message = str(e)
        print(f"Dashboard pipeline failed: {error_message}")

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "playbook": {"trades": [], "tomorrows_tape": "Pipeline execution failed"},
        "logs": f"Error: {error_message}",
    }
