#!/usr/bin/env python3
"""
TODO/FIXME Analysis and Prioritization Tool for ORACLE-X

This script scans the codebase for TODO, FIXME, HACK, and similar markers,
categorizes them by priority, and generates actionable remediation plans.
"""

import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum


class TodoPriority(Enum):
    """Priority levels for TODO items"""
    CRITICAL = 1    # Security issues, data integrity, blocking features
    HIGH = 2        # Missing functionality, performance issues
    MEDIUM = 3      # Optimizations, enhancements
    LOW = 4         # Code quality, documentation improvements


@dataclass
class TodoItem:
    """Represents a TODO/FIXME item found in code"""
    file_path: str
    line_number: int
    marker_type: str  # TODO, FIXME, HACK, etc.
    description: str
    context: str  # Surrounding code context
    priority: TodoPriority
    category: str
    estimated_effort: str  # S, M, L, XL


class TodoAnalyzer:
    """Analyzes and categorizes TODO items in the codebase"""
    
    # Pattern to match TODO markers
    TODO_PATTERNS = [
        r'#\s*(TODO|FIXME|HACK|XXX|BUG|OPTIMIZE|REFACTOR|NOTE)\s*:?\s*(.+)',
        r'//\s*(TODO|FIXME|HACK|XXX|BUG|OPTIMIZE|REFACTOR|NOTE)\s*:?\s*(.+)',
        r'/\*\s*(TODO|FIXME|HACK|XXX|BUG|OPTIMIZE|REFACTOR|NOTE)\s*:?\s*(.+?)\*/',
    ]
    
    # Priority keywords for automatic classification
    PRIORITY_KEYWORDS = {
        TodoPriority.CRITICAL: [
            'security', 'vulnerability', 'inject', 'auth', 'password', 'token',
            'critical', 'urgent', 'blocking', 'broken', 'data loss', 'corrupt'
        ],
        TodoPriority.HIGH: [
            'implement', 'missing', 'required', 'essential', 'integrate',
            'performance', 'slow', 'timeout', 'memory', 'leak', 'api'
        ],
        TodoPriority.MEDIUM: [
            'optimize', 'improve', 'enhance', 'refactor', 'cleanup',
            'better', 'efficiency', 'cache', 'feature', 'upgrade'
        ],
        TodoPriority.LOW: [
            'documentation', 'comment', 'rename', 'style', 'format',
            'logging', 'debug', 'test', 'example', 'cosmetic'
        ]
    }
    
    # Categories for grouping related TODOs
    CATEGORIES = {
        'data_sources': ['data_feeds/', 'adapter', 'api', 'feed', 'source'],
        'ml_pipeline': ['ml', 'model', 'prediction', 'training', 'ensemble'],
        'options_trading': ['option', 'valuation', 'strike', 'expiry', 'greeks'],
        'infrastructure': ['cache', 'database', 'config', 'deploy', 'monitor'],
        'testing': ['test', 'mock', 'validate', 'verify', 'coverage'],
        'documentation': ['doc', 'readme', 'comment', 'explain', 'guide'],
        'security': ['auth', 'secure', 'encrypt', 'validate', 'sanitize'],
        'performance': ['optimize', 'speed', 'memory', 'cache', 'parallel'],
        'ui_dashboard': ['dashboard', 'ui', 'frontend', 'display', 'chart'],
        'general': []  # catch-all
    }
    
    def __init__(self, root_path: str = '.'):
        self.root_path = Path(root_path)
        self.todos: List[TodoItem] = []
    
    def scan_codebase(self, 
                     file_patterns: Optional[List[str]] = None,
                     exclude_patterns: Optional[List[str]] = None) -> List[TodoItem]:
        """
        Scan the codebase for TODO markers
        
        Args:
            file_patterns: File patterns to include (default: Python files)
            exclude_patterns: Patterns to exclude
            
        Returns:
            List of TodoItem objects
        """
        if file_patterns is None:
            file_patterns = ['**/*.py', '**/*.js', '**/*.ts', '**/*.md', '**/*.yaml', '**/*.yml']
        
        if exclude_patterns is None:
            exclude_patterns = [
                '**/.*',  # Hidden files
                '**/__pycache__/**',
                '**/node_modules/**',
                '**/venv/**',
                '**/.venv/**',
                '**/env/**',
                '**/build/**',
                '**/dist/**'
            ]
        
        todos = []
        
        for pattern in file_patterns:
            for file_path in self.root_path.glob(pattern):
                # Skip if matches exclude pattern
                if any(file_path.match(exclude) for exclude in exclude_patterns):
                    continue
                
                if file_path.is_file():
                    todos.extend(self._scan_file(file_path))
        
        self.todos = todos
        return todos
    
    def _scan_file(self, file_path: Path) -> List[TodoItem]:
        """Scan a single file for TODO markers"""
        todos = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                for pattern in self.TODO_PATTERNS:
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        marker_type = match.group(1).upper()
                        description = match.group(2).strip()
                        
                        # Get context (2 lines before and after)
                        start_line = max(0, line_num - 3)
                        end_line = min(len(lines), line_num + 2)
                        context = ''.join(lines[start_line:end_line])
                        
                        todo = TodoItem(
                            file_path=str(file_path.relative_to(self.root_path)),
                            line_number=line_num,
                            marker_type=marker_type,
                            description=description,
                            context=context,
                            priority=self._classify_priority(description, str(file_path)),
                            category=self._classify_category(description, str(file_path)),
                            estimated_effort=self._estimate_effort(description)
                        )
                        todos.append(todo)
        
        except Exception as e:
            print(f"Warning: Could not scan {file_path}: {e}")
        
        return todos
    
    def _classify_priority(self, description: str, file_path: str) -> TodoPriority:
        """Classify TODO priority based on keywords and context"""
        description_lower = description.lower()
        file_path_lower = file_path.lower()
        
        # Check for priority keywords
        for priority, keywords in self.PRIORITY_KEYWORDS.items():
            if any(keyword in description_lower or keyword in file_path_lower 
                   for keyword in keywords):
                return priority
        
        # Default to medium priority
        return TodoPriority.MEDIUM
    
    def _classify_category(self, description: str, file_path: str) -> str:
        """Classify TODO category based on content and file location"""
        description_lower = description.lower()
        file_path_lower = file_path.lower()
        
        for category, keywords in self.CATEGORIES.items():
            if category == 'general':
                continue
            if any(keyword in description_lower or keyword in file_path_lower 
                   for keyword in keywords):
                return category
        
        return 'general'
    
    def _estimate_effort(self, description: str) -> str:
        """Estimate implementation effort"""
        description_lower = description.lower()
        
        # Large effort indicators
        if any(word in description_lower for word in [
            'implement', 'integrate', 'redesign', 'refactor', 'complete'
        ]):
            return 'L'
        
        # Small effort indicators
        if any(word in description_lower for word in [
            'rename', 'comment', 'document', 'fix typo', 'update'
        ]):
            return 'S'
        
        # Medium effort (default)
        return 'M'
    
    def generate_report(self) -> str:
        """Generate a comprehensive TODO analysis report"""
        if not self.todos:
            return "No TODO items found in the codebase."
        
        report = []
        report.append("# TODO/FIXME Analysis Report")
        report.append("=" * 50)
        report.append(f"**Total Items Found:** {len(self.todos)}")
        report.append("")
        
        # Priority breakdown
        priority_counts = {}
        for todo in self.todos:
            priority_counts[todo.priority] = priority_counts.get(todo.priority, 0) + 1
        
        report.append("## Priority Breakdown")
        for priority in TodoPriority:
            count = priority_counts.get(priority, 0)
            report.append(f"- **{priority.name}**: {count} items")
        report.append("")
        
        # Category breakdown
        category_counts = {}
        for todo in self.todos:
            category_counts[todo.category] = category_counts.get(todo.category, 0) + 1
        
        report.append("## Category Breakdown")
        for category, count in sorted(category_counts.items()):
            report.append(f"- **{category}**: {count} items")
        report.append("")
        
        # Detailed items by priority
        for priority in TodoPriority:
            priority_todos = [t for t in self.todos if t.priority == priority]
            if priority_todos:
                report.append(f"## {priority.name} Priority Items ({len(priority_todos)})")
                report.append("")
                
                for todo in sorted(priority_todos, key=lambda x: (x.category, x.file_path)):
                    report.append(f"### {todo.marker_type}: {todo.description}")
                    report.append(f"- **File:** `{todo.file_path}:{todo.line_number}`")
                    report.append(f"- **Category:** {todo.category}")
                    report.append(f"- **Effort:** {todo.estimated_effort}")
                    report.append("")
        
        return "\n".join(report)
    
    def get_prioritized_action_plan(self) -> str:
        """Generate an actionable prioritized plan"""
        if not self.todos:
            return "No TODO items to prioritize."
        
        plan = []
        plan.append("# TODO Remediation Action Plan")
        plan.append("=" * 40)
        plan.append("")
        
        # Group by priority and effort
        for priority in TodoPriority:
            priority_todos = [t for t in self.todos if t.priority == priority]
            if not priority_todos:
                continue
            
            plan.append(f"## {priority.name} Priority ({len(priority_todos)} items)")
            
            # Group by effort within priority
            effort_groups = {'S': [], 'M': [], 'L': []}
            for todo in priority_todos:
                effort_groups[todo.estimated_effort].append(todo)
            
            for effort in ['S', 'M', 'L']:
                if effort_groups[effort]:
                    effort_name = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}[effort]
                    plan.append(f"### {effort_name} Effort ({len(effort_groups[effort])} items)")
                    
                    for todo in effort_groups[effort]:
                        plan.append(f"- [ ] **{todo.file_path}:{todo.line_number}** - {todo.description}")
            
            plan.append("")
        
        return "\n".join(plan)


def main():
    """CLI entry point for TODO analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze TODO/FIXME items in ORACLE-X codebase')
    parser.add_argument('--path', default='.', help='Root path to scan')
    parser.add_argument('--report', action='store_true', help='Generate detailed report')
    parser.add_argument('--action-plan', action='store_true', help='Generate action plan')
    parser.add_argument('--output', help='Output file for report')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    
    args = parser.parse_args()
    
    analyzer = TodoAnalyzer(args.path)
    todos = analyzer.scan_codebase()
    
    if args.summary or (not args.report and not args.action_plan):
        # Show summary
        print("📋 TODO Analysis Summary")
        print(f"Total items found: {len(todos)}")
        
        if todos:
            priority_counts = {}
            category_counts = {}
            
            for todo in todos:
                priority_counts[todo.priority] = priority_counts.get(todo.priority, 0) + 1
                category_counts[todo.category] = category_counts.get(todo.category, 0) + 1
            
            print("\nBy Priority:")
            for priority in TodoPriority:
                count = priority_counts.get(priority, 0)
                if count > 0:
                    print(f"  {priority.name}: {count}")
            
            print("\nTop Categories:")
            sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            for category, count in sorted_categories[:5]:
                print(f"  {category}: {count}")
    
    if args.report:
        report = analyzer.generate_report()
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)
    
    if args.action_plan:
        plan = analyzer.get_prioritized_action_plan()
        output_file = args.output or "TODO_ACTION_PLAN.md"
        with open(output_file, 'w') as f:
            f.write(plan)
        print(f"Action plan saved to {output_file}")


if __name__ == '__main__':
    main()
