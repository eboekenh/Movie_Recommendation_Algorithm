# 🎬 Hybrid Movie Recommendation System

A hybrid recommender system combining User-Based and Item-Based Collaborative Filtering on the MovieLens 20M dataset to generate personalized movie recommendations.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-green)
![License](https://img.shields.io/badge/License-MIT-green)

## Problem Statement

Given a user's movie rating history, recommend new movies they are likely to enjoy using two complementary approaches:
- **User-Based CF** — Find similar users by taste and recommend what they liked
- **Item-Based CF** — Find movies similar to what the user already rated highly

## Dataset

- **Source:** [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/)
- **Scale:** 20 million ratings from 138,493 users across 27,278 movies
- **After filtering:** Movies with 1,000+ ratings retained (3,134 movies)

## Algorithm

### User-Based Collaborative Filtering
1. Build user-movie rating matrix (138K × 3K)
2. Identify users who watched ≥60% of the same movies as the target user
3. Compute Pearson correlation between target user and candidates
4. Filter to users with correlation ≥ 0.65
5. Calculate weighted ratings (correlation × rating)
6. Rank movies by weighted average score → **Top 5 recommendations**

### Item-Based Collaborative Filtering
1. Find the target user's highest-rated recent movie
2. Compute pairwise Pearson correlation between that movie and all others
3. Rank by correlation strength → **Top 5 recommendations**

### Hybrid Output
Combines both approaches: **5 user-based + 5 item-based = 10 recommendations**

## Example Output

**User-Based Recommendations:**
| Movie | Weighted Score |
|-------|---------------|
| The Shawshank Redemption | 4.52 |
| Pulp Fiction | 4.38 |
| Forrest Gump | 4.31 |
| The Matrix | 4.27 |
| Fight Club | 4.15 |

*Actual results vary by target user.*

## Tech Stack

- **Python 3.8+** — Core language
- **Pandas** — Data manipulation, pivot tables, correlation computation
- **NumPy** — Numerical operations

## Getting Started

```bash
git clone https://github.com/eboekenh/Movie_Recommendation_Algorithm.git
cd Movie_Recommendation_Algorithm
pip install -r requirements.txt
```

Download the [MovieLens 20M dataset](https://grouplens.org/datasets/movielens/20m/) and place `movies.csv` and `ratings.csv` in the project directory.

```bash
python Hybrid_movie_recommendation.py
```

## Key Design Decisions

- **Minimum 1,000 ratings per movie** — Ensures correlation calculations are statistically meaningful
- **60% overlap threshold** — Users must share substantial viewing history for reliable similarity
- **Pearson correlation ≥ 0.65** — Only strongly similar users influence recommendations
- **Weighted ratings** — Higher-correlated users have more influence on recommendations

## License

MIT
