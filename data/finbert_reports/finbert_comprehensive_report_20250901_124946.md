# Comprehensive FinBERT Model Accuracy Report
Generated: 2025-09-01 12:49:46

## Executive Summary

### Key Findings
- **Best Performing Model**: finbert_tone
.3f
- **Total Models Tested**: 6 FinBERT variants
- **Test Datasets**: Twitter, News, Reddit, Mixed sources
- **Calibration Issues Identified**: 24 significant issues

### Performance Highlights
- **Twitter**: Good (0.696 accuracy)
- **News**: Poor (0.219 accuracy)
- **Reddit**: Poor (0.153 accuracy)
- **Mixed**: Fair (0.321 accuracy)

## Detailed Performance Analysis

### Model Rankings
1. **finbert_tone**: 0.347
2. **finbert_pretrain**: 0.347
3. **finbert_news**: 0.347
4. **finbert_earnings**: 0.347
5. **finbert_prosus**: 0.347
6. **fintwitbert_sentiment**: 0.347

## Source-Specific Analysis

### Best Models by Source
#### Twitter
- **Recommended Model**: finbert_tone
.3f
- **Performance Level**: Good

#### News
- **Recommended Model**: finbert_tone
.3f
- **Performance Level**: Poor

#### Reddit
- **Recommended Model**: finbert_tone
.3f
- **Performance Level**: Poor

#### Mixed
- **Recommended Model**: finbert_tone
.3f
- **Performance Level**: Fair

## Calibration Analysis

### Calibration Issues Found: 24

#### Top Calibration Issues
| Model | Dataset | Calibration Error | Severity |
|-------|---------|-------------------|----------|
| finbert_tone | news | 0.770 | High |
| finbert_pretrain | news | 0.770 | High |
| finbert_news | news | 0.770 | High |
| finbert_earnings | news | 0.770 | High |
| finbert_prosus | news | 0.770 | High |

## Technical Insights

### Model Architecture Analysis
- **Fallback Behavior**: All models fall back to FinTwitBERT for Twitter and finbert-tone for other sources
- **Performance Consistency**: All models show identical performance due to shared fallback models
- **Source-Specific Optimization**: Twitter performs best, followed by mixed, news, and Reddit

## Future Improvements

### Key Priorities
1. **Calibration Fixes**: Address the 24 identified calibration issues
2. **Source Optimization**: Improve performance on news and Reddit data
3. **Model Differentiation**: Better distinguish between FinBERT variants
4. **Real-time Adaptation**: Implement dynamic model selection
