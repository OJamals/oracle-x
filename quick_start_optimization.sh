#!/bin/bash
"""
Oracle-X Optimization Quick Start

Run this script to quickly set up and test the optimization system.
"""

echo "🚀 Oracle-X Optimization Quick Start"
echo "===================================="

# Set up environment
echo "🔧 Setting up environment..."
source optimization.env 2>/dev/null || echo "⚠️  optimization.env not found, using defaults"

# Test system
echo "🧪 Testing optimization system..."
python oracle_optimize_cli.py test --prompt "Quick test of optimization system"

if [ $? -eq 0 ]; then
    echo "✅ Optimization system test passed!"
else
    echo "❌ Optimization system test failed!"
    exit 1
fi

# Show analytics
echo "📊 Current system analytics..."
python oracle_optimize_cli.py analytics --days 1

# List templates
echo "📝 Available templates..."
python oracle_optimize_cli.py templates list

echo ""
echo "🎉 Quick start completed!"
echo "Next steps:"
echo "  1. Run optimized pipeline: python main_optimized.py"
echo "  2. Monitor performance: python oracle_optimize_cli.py analytics"
echo "  3. Start experiments: python oracle_optimize_cli.py experiment start [template_a] [template_b] [condition]"
echo ""
