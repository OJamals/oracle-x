# Test Optimization Guide

The Oracle-X test suite has been optimized for better performance and reliability.

## Quick Start

### Fast Tests (Default - Recommended for Development)
```bash
# Run only fast unit tests (excludes slow integration tests)
python run_tests.py

# Alternative: direct pytest command
python -m pytest -m "not integration and not ml and not slow"
```

### All Tests
```bash
# Run all tests including slow integration tests  
python run_tests.py --all

# Run with custom timeout
python run_tests.py --all --timeout 60
```

### Test Categories

#### By Speed
- `--fast` - Unit tests only (default)
- `--all` - All tests including slow ones

#### By Type  
- `--unit` - Unit tests only
- `--integration` - Integration tests (make real API calls)
- `--ml` - Machine learning tests (very slow)

#### Examples
```bash
# Unit tests only
python run_tests.py --unit

# Integration tests only (slow)
python run_tests.py --integration

# ML tests only (very slow)
python run_tests.py --ml

# Specific test file
python run_tests.py tests/test_financial_calculator.py

# Verbose output
python run_tests.py --verbose
```

## Performance Optimizations

### Enabled Features
- ✅ **30-second timeouts** - Prevents hanging tests
- ✅ **Parallel execution** - Tests run in parallel automatically
- ✅ **Test categorization** - Fast vs slow test separation
- ✅ **Warning suppression** - Reduces noise
- ✅ **Fail fast** - Stops after 5 failures

### Test Markers
Tests are now marked with categories:
- `@pytest.mark.unit` - Fast unit tests  
- `@pytest.mark.integration` - Integration tests (external APIs)
- `@pytest.mark.ml` - Machine learning tests
- `@pytest.mark.network` - Tests requiring network access
- `@pytest.mark.slow` - Tests taking > 5 seconds
- `@pytest.mark.api` - Tests calling external APIs

## Troubleshooting

### Tests Taking Too Long
```bash
# Run only fast tests
python run_tests.py --fast

# Disable parallel execution if causing issues
python run_tests.py --no-parallel
```

### Network/API Issues
```bash
# Skip network-dependent tests
python -m pytest -m "not network and not api"
```

### Memory Issues
```bash
# Run tests serially instead of parallel
python run_tests.py --no-parallel
```

## Configuration Files

- `pytest.ini` - Main pytest configuration with timeouts and markers
- `run_tests.py` - Optimized test runner script
- `conftest.py` - Test fixtures and setup (if exists)

## Migration Notes

The test suite has been optimized to separate:
1. **Fast unit tests** - Run by default for quick feedback
2. **Slow integration tests** - Run explicitly when needed
3. **ML training tests** - Run only when developing ML features

This provides better developer experience with faster test cycles.
