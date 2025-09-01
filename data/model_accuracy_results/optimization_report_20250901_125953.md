# Model Accuracy Optimization Report
Generated: 2025-09-01 12:59:53

## Executive Summary

- **Best Overall Model**: finbert_tone
.3f
- **Total Models Tested**: 7
- **Sources Tested**: 4
- **Total Test Cases**: 336

### Recommended Models by Source
- **Twitter**: `finbert_tone` .3f
- **News**: `finbert_tone` .3f
- **Financial**: `finbert_tone` .3f
- **Reddit**: `finbert_tone` .3f

## Performance Rankings

| Rank | Model | Source | Accuracy |
|------|-------|--------|----------|
| 1 | finbert_tone | reddit | 0.917 |
| 2 | finbert_pretrain | reddit | 0.917 |
| 3 | finbert_news | reddit | 0.917 |
| 4 | finbert_earnings | reddit | 0.917 |
| 5 | finbert_prosus | reddit | 0.917 |
| 6 | fintwitbert_sentiment | reddit | 0.917 |
| 7 | reddit_roberta_sentiment | reddit | 0.917 |
| 8 | finbert_tone | twitter | 0.833 |
| 9 | finbert_pretrain | twitter | 0.833 |
| 10 | finbert_news | twitter | 0.833 |

## Calibration Issues

Found 14 models with confidence-accuracy mismatch:

- **finbert_tone** on news:
.3f
.3f
- **finbert_pretrain** on news:
.3f
.3f
- **finbert_news** on news:
.3f
.3f
- **finbert_earnings** on news:
.3f
.3f
- **finbert_prosus** on news:
.3f
.3f
- **fintwitbert_sentiment** on news:
.3f
.3f
- **reddit_roberta_sentiment** on news:
.3f
.3f
- **finbert_tone** on financial:
.3f
.3f
- **finbert_pretrain** on financial:
.3f
.3f
- **finbert_news** on financial:
.3f
.3f
- **finbert_earnings** on financial:
.3f
.3f
- **finbert_prosus** on financial:
.3f
.3f
- **fintwitbert_sentiment** on financial:
.3f
.3f
- **reddit_roberta_sentiment** on financial:
.3f
.3f