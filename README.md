# YouTube Trending Videos - What Actually Makes a Video Go Viral?

I wanted to go beyond basic EDA and actually try to *predict* virality, so I took the YouTube US trending dataset and built a full ML pipeline on top of it. This covers everything from feature engineering to SHAP interpretability.

---

## What I was trying to answer

Can you predict whether a YouTube video will go viral just from its metadata - things like the category, when it was published, title length, number of tags? No engagement data like likes or comments, since you wouldn't have that before a video trends.

---

## The dataset

**YouTube Trending Videos (US)** from Kaggle - about 40,000 rows of trending video snapshots collected daily across 2017-2018. Each row has the video's metadata and engagement stats at the time it was trending.

After deduplication (keeping one row per video at peak engagement), the working dataset was ~6,350 unique videos.

---

## Project structure

```
youtube-trending-analysis/
│
├── README.md
├── youtube_video_analysis.ipynb
├── requirements.txt
```

---

## What I built

### 1. Exploratory Data Analysis
Started with the basics - category distributions, engagement metric distributions, correlation between views/likes/comments, and how trending volume changed over time. Entertainment and Music dominate the trending list by a wide margin.

### 2. Feature Engineering
Built 15+ features from raw metadata:
- Title features: length, word count, whether it has caps/numbers/exclamation marks
- Tag count
- Publish timing: hour, day of week, month, weekend flag
- Days between publish date and trending date
- Category label encoding

### 3. Regression — Predicting View Count
Framed this as a regression problem on log(views) to handle the heavy skew. Compared three models with 5-fold cross-validation:

| Model | CV RMSE | Test R² |
|---|---|---|
| Ridge Regression | 1.78 | 0.07 |
| Random Forest | 1.46 | 0.35 |
| XGBoost | 1.43 | **0.38** |

XGBoost came out on top. The R² of 0.38 is honestly what you'd expect here - view counts on YouTube are genuinely noisy and influenced by a lot of things the metadata doesn't capture (thumbnails, the algorithm, promotion spend). Getting to 0.38 with just metadata is reasonable.


### 4. Classification — Viral vs Non-Viral
Turned it into a binary classification problem: videos in the top 25% of views (above ~1.47M) are labelled viral. Used RandomizedSearchCV to tune the Random Forest and compared against XGBoost.

| Model | ROC-AUC |
|---|---|
| Random Forest (tuned) | 0.784 |
| XGBoost | **0.808** |

XGBoost again wins. At ROC-AUC of 0.81, it's correctly ranking a viral video above a non-viral one 81% of the time - not bad for metadata alone.

### 5. NLP — Adding Title and Tag Text
Vectorized video titles and tags using TF-IDF (500 features, unigrams + bigrams) and combined with the structured features. This pushed the ROC-AUC from **0.808 -> 0.840** - a meaningful improvement just from reading the words.

Top predictive terms: "official", "lyric video", "bros", "buzzfeed", "stephen colbert" - mostly signals of established channels and music content, which trend more reliably.


### 6. SHAP Interpretability
Used SHAP to understand *why* the model makes each prediction, not just *what* it predicts.

**Key findings from SHAP:**
- `days_to_trend` is the strongest signal - videos that stay trending longer accumulate more views, and the model picked this up
- `category_encoded` is second - category genuinely matters
- `publish_month` and `tag_count` follow - when you post and how many tags you use both have real impact
- `is_weekend` and `ratings_disabled` had almost no effect


---

## Key takeaways

- Simple metadata does have *some* predictive power, but YouTube virality is hard to predict without content-level signals (thumbnail, actual video quality, algorithm push)
- Category and timing are the biggest controllable factors
- Adding text features (TF-IDF) gave a meaningful boost over structured features alone
- Tree-based models (Random Forest, XGBoost) significantly outperform linear models on this type of tabular data

---

## How to run

```bash
pip install -r requirements.txt
jupyter notebook youtube_trending_eda_ml.ipynb
```

The notebook downloads the dataset automatically via `kagglehub`. You'll need a Kaggle account and your API key set up.

---

## Requirements

See `requirements.txt`. Main dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `kagglehub`.
