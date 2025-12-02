#!/usr/bin/env python3
"""
Test Coverage Analysis and Enhancement Tool for ORACLE-X

This script analyzes current test coverage, identifies gaps, and generates
test stubs for uncovered critical functions and classes.
"""

import ast
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class CoverageAnalysis:
    """Analysis result for a module's test coverage"""

    module_path: str
    coverage_percentage: float
    total_statements: int
    covered_statements: int
    missing_lines: List[int]
    uncovered_functions: List[str]
    uncovered_classes: List[str]
    has_tests: bool
    test_file_path: Optional[str]
    priority: str  # HIGH, MEDIUM, LOW


class TestCoverageAnalyzer:
    """Analyzes test coverage and suggests improvements"""

    # Critical modules that need high test coverage
    CRITICAL_MODULES = [
        "oracle_options_pipeline.py",
        "data_feeds/data_feed_orchestrator.py",
        "data_feeds/options_valuation_engine.py",
        "oracle_engine/ensemble_ml_engine.py",
        "oracle_engine/ml_prediction_engine.py",
    ]

    # Test file patterns to look for
    TEST_PATTERNS = [
        "test_{module_name}.py",
        "test_{module_name}_*.py",
        "{module_name}_test.py",
        "tests/test_{module_name}.py",
        "tests/unit/test_{module_name}.py",
        "tests/integration/test_{module_name}.py",
    ]

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.coverage_data: Dict[str, CoverageAnalysis] = {}

    def run_coverage_analysis(self) -> Dict[str, CoverageAnalysis]:
        """Run comprehensive coverage analysis"""
        print("🔍 Running test coverage analysis...")

        # Run pytest with coverage
        try:
            cmd = [
                "python",
                "-m",
                "pytest",
                "--cov=.",
                "--cov-report=json:coverage.json",
                "--cov-report=term-missing",
                "tests/",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.root_path
            )

            if result.returncode != 0:
                print(f"⚠️  Pytest had issues: {result.stderr}")

            # Parse coverage.json if it exists
            coverage_json_path = self.root_path / "coverage.json"
            if coverage_json_path.exists():
                with open(coverage_json_path, "r") as f:
                    coverage_data = json.load(f)
                self._parse_coverage_data(coverage_data)

        except Exception as e:
            print(f"❌ Error running coverage analysis: {e}")
            return {}

        return self.coverage_data

    def _parse_coverage_data(self, coverage_data: Dict[str, Any]) -> None:
        """Parse coverage.json data"""
        files = coverage_data.get("files", {})

        for file_path, file_data in files.items():
            # Skip test files themselves
            if "test" in file_path.lower():
                continue

            # Get coverage metrics
            summary = file_data.get("summary", {})
            coverage_percentage = summary.get("percent_covered", 0.0)
            total_statements = summary.get("num_statements", 0)
            covered_statements = summary.get("covered_lines", 0)
            missing_lines = file_data.get("missing_lines", [])

            # Analyze the source file for functions and classes
            uncovered_functions, uncovered_classes = self._analyze_source_file(
                file_path, missing_lines
            )

            # Check if tests exist
            has_tests, test_file_path = self._find_test_file(file_path)

            # Determine priority
            priority = self._determine_priority(file_path, coverage_percentage)

            analysis = CoverageAnalysis(
                module_path=file_path,
                coverage_percentage=coverage_percentage,
                total_statements=total_statements,
                covered_statements=covered_statements,
                missing_lines=missing_lines,
                uncovered_functions=uncovered_functions,
                uncovered_classes=uncovered_classes,
                has_tests=has_tests,
                test_file_path=test_file_path,
                priority=priority,
            )

            self.coverage_data[file_path] = analysis

    def _analyze_source_file(
        self, file_path: str, missing_lines: List[int]
    ) -> Tuple[List[str], List[str]]:
        """Analyze source file to find uncovered functions and classes"""
        uncovered_functions = []
        uncovered_classes = []

        try:
            full_path = self.root_path / file_path
            if not full_path.exists():
                return uncovered_functions, uncovered_classes

            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if function definition line is in missing lines
                    if node.lineno in missing_lines:
                        uncovered_functions.append(node.name)

                elif isinstance(node, ast.ClassDef):
                    # Check if class definition line is in missing lines
                    if node.lineno in missing_lines:
                        uncovered_classes.append(node.name)

        except Exception as e:
            print(f"Warning: Could not analyze {file_path}: {e}")

        return uncovered_functions, uncovered_classes

    def _find_test_file(self, module_path: str) -> Tuple[bool, Optional[str]]:
        """Find corresponding test file for a module"""
        module_name = Path(module_path).stem

        for pattern in self.TEST_PATTERNS:
            test_file_pattern = pattern.format(module_name=module_name)

            # Check in various locations
            for test_dir in [".", "tests", "tests/unit", "tests/integration"]:
                test_path = self.root_path / test_dir / test_file_pattern
                if test_path.exists():
                    return True, str(test_path.relative_to(self.root_path))

        return False, None

    def _determine_priority(self, file_path: str, coverage_percentage: float) -> str:
        """Determine priority for improving test coverage"""
        # Critical modules get high priority regardless of coverage
        if any(critical in file_path for critical in self.CRITICAL_MODULES):
            return "HIGH"

        # Low coverage gets higher priority
        if coverage_percentage < 50:
            return "HIGH"
        elif coverage_percentage < 70:
            return "MEDIUM"
        else:
            return "LOW"

    def generate_test_stubs(self, module_path: str) -> str:
        """Generate test stub code for a module"""
        try:
            full_path = self.root_path / module_path
            if not full_path.exists():
                return ""

            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)

            # Extract functions and classes
            functions = []
            classes = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    classes.append(node.name)

            # Generate test stub
            module_name = Path(module_path).stem
            test_code = self._generate_test_template(
                module_name, module_path, functions, classes
            )

            return test_code

        except Exception as e:
            print(f"Error generating test stub for {module_path}: {e}")
            return ""

    def _generate_test_template(
        self,
        module_name: str,
        module_path: str,
        functions: List[str],
        classes: List[str],
    ) -> str:
        """Generate test template code"""
        import_path = module_path.replace("/", ".").replace(".py", "")

        template = f'''#!/usr/bin/env python3
"""
Test suite for {module_name}

Auto-generated test stub - implement the actual test logic.
"""

import unittest
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from {import_path} import (
'''

        # Add imports for classes and functions
        all_imports = classes + functions
        if all_imports:
            for i, item in enumerate(all_imports):
                template += f"        {item}"
                if i < len(all_imports) - 1:
                    template += ","
                template += "\n"

        template += '''    )
except ImportError as e:
    pytest.skip("Could not import module", allow_module_level=True)


class Test{module_name}(unittest.TestCase):
    """Test suite for {module_name}"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass
    
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        pass

'''.format(
            module_name=module_name.replace("_", "").title()
        )

        # Add test methods for functions
        for func in functions:
            template += f'''    def test_{func}(self):
        """Test {func} function"""
        # TODO: Implement test for {func}
        self.skipTest("Test not implemented yet")
    
'''

        # Add test classes for classes
        for cls in classes:
            template += f'''    def test_{cls.lower()}_creation(self):
        """Test {cls} instantiation"""
        # TODO: Implement test for {cls} creation
        self.skipTest("Test not implemented yet")
    
    def test_{cls.lower()}_methods(self):
        """Test {cls} methods"""
        # TODO: Implement test for {cls} methods
        self.skipTest("Test not implemented yet")
    
'''

        template += """
if __name__ == '__main__':
    unittest.main()
"""

        return template

    def generate_coverage_report(self) -> str:
        """Generate comprehensive coverage report"""
        if not self.coverage_data:
            return "No coverage data available. Run run_coverage_analysis() first."

        report = []
        report.append("# Test Coverage Analysis Report")
        report.append("=" * 50)

        # Overall statistics
        total_files = len(self.coverage_data)
        high_priority = len(
            [a for a in self.coverage_data.values() if a.priority == "HIGH"]
        )
        low_coverage = len(
            [a for a in self.coverage_data.values() if a.coverage_percentage < 50]
        )
        no_tests = len([a for a in self.coverage_data.values() if not a.has_tests])

        report.append(f"**Total Files Analyzed:** {total_files}")
        report.append(f"**High Priority Files:** {high_priority}")
        report.append(f"**Files with <50% Coverage:** {low_coverage}")
        report.append(f"**Files without Tests:** {no_tests}")
        report.append("")

        # Files by priority
        for priority in ["HIGH", "MEDIUM", "LOW"]:
            priority_files = [
                a for a in self.coverage_data.values() if a.priority == priority
            ]
            if priority_files:
                report.append(f"## {priority} Priority Files ({len(priority_files)})")
                report.append("")

                for analysis in sorted(
                    priority_files, key=lambda x: x.coverage_percentage
                ):
                    report.append(f"### {analysis.module_path}")
                    report.append(
                        f"- **Coverage:** {analysis.coverage_percentage:.1f}%"
                    )
                    report.append(
                        f"- **Statements:** {analysis.covered_statements}/{analysis.total_statements}"
                    )
                    report.append(
                        f"- **Has Tests:** {'✅' if analysis.has_tests else '❌'}"
                    )

                    if analysis.test_file_path:
                        report.append(f"- **Test File:** `{analysis.test_file_path}`")

                    if analysis.uncovered_functions:
                        report.append(
                            f"- **Uncovered Functions:** {', '.join(analysis.uncovered_functions[:5])}"
                        )
                        if len(analysis.uncovered_functions) > 5:
                            report.append(
                                f"  (and {len(analysis.uncovered_functions) - 5} more)"
                            )

                    if analysis.uncovered_classes:
                        report.append(
                            f"- **Uncovered Classes:** {', '.join(analysis.uncovered_classes)}"
                        )

                    report.append("")

        return "\n".join(report)

    def generate_improvement_recommendations(self) -> str:
        """Generate actionable recommendations for improving test coverage"""
        recommendations = []
        recommendations.append("# Test Coverage Improvement Recommendations")
        recommendations.append("=" * 55)
        recommendations.append("")

        # Immediate actions (High priority, no tests)
        no_tests_high = [
            a
            for a in self.coverage_data.values()
            if a.priority == "HIGH" and not a.has_tests
        ]

        if no_tests_high:
            recommendations.append("## 🚨 Immediate Actions Required")
            recommendations.append("These critical files have no tests:")
            recommendations.append("")

            for analysis in no_tests_high:
                recommendations.append(
                    f"- [ ] **{analysis.module_path}** - Create test file"
                )
                recommendations.append(
                    f"  - Suggested path: `tests/test_{Path(analysis.module_path).stem}.py`"
                )
                recommendations.append(
                    f"  - Functions to test: {len(analysis.uncovered_functions)} functions"
                )
                recommendations.append(
                    f"  - Classes to test: {len(analysis.uncovered_classes)} classes"
                )
                recommendations.append("")

        # Coverage improvements
        low_coverage = [
            a
            for a in self.coverage_data.values()
            if a.coverage_percentage < 70 and a.has_tests
        ]

        if low_coverage:
            recommendations.append("## 📈 Coverage Improvements")
            recommendations.append("These files have tests but need more coverage:")
            recommendations.append("")

            for analysis in sorted(low_coverage, key=lambda x: x.coverage_percentage):
                recommendations.append(
                    f"- [ ] **{analysis.module_path}** ({analysis.coverage_percentage:.1f}% coverage)"
                )
                if analysis.uncovered_functions:
                    recommendations.append(
                        f"  - Add tests for: {', '.join(analysis.uncovered_functions[:3])}"
                    )
                recommendations.append("")

        # Integration test suggestions
        recommendations.append("## 🔗 Integration Test Suggestions")
        recommendations.append("")
        recommendations.append("- [ ] Create end-to-end pipeline tests")
        recommendations.append("- [ ] Add error handling and edge case tests")
        recommendations.append("- [ ] Test API integrations with mocked services")
        recommendations.append(
            "- [ ] Add performance and load tests for critical paths"
        )
        recommendations.append("")

        return "\n".join(recommendations)


def main():
    """CLI entry point for test coverage analysis"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze and improve test coverage for ORACLE-X"
    )
    parser.add_argument("--analyze", action="store_true", help="Run coverage analysis")
    parser.add_argument(
        "--report", action="store_true", help="Generate coverage report"
    )
    parser.add_argument(
        "--recommendations",
        action="store_true",
        help="Generate improvement recommendations",
    )
    parser.add_argument(
        "--generate-stub", help="Generate test stub for specific module"
    )
    parser.add_argument("--output", help="Output file for reports")

    args = parser.parse_args()

    analyzer = TestCoverageAnalyzer()

    if args.analyze or args.report or args.recommendations:
        print("Running coverage analysis...")
        analyzer.run_coverage_analysis()

    if args.report:
        report = analyzer.generate_coverage_report()
        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Coverage report saved to {args.output}")
        else:
            print(report)

    if args.recommendations:
        recommendations = analyzer.generate_improvement_recommendations()
        output_file = args.output or "COVERAGE_RECOMMENDATIONS.md"
        with open(output_file, "w") as f:
            f.write(recommendations)
        print(f"Recommendations saved to {output_file}")

    if args.generate_stub:
        stub_code = analyzer.generate_test_stubs(args.generate_stub)
        if stub_code:
            module_name = Path(args.generate_stub).stem
            output_file = f"test_{module_name}.py"
            with open(output_file, "w") as f:
                f.write(stub_code)
            print(f"Test stub generated: {output_file}")
        else:
            print(f"Could not generate stub for {args.generate_stub}")


if __name__ == "__main__":
    main()
