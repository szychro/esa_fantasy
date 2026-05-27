# esa_fantasy

## Running the scraper

- `python3 main.py` refreshes only the current season and merges new rows into `ekstraklasa_players.csv`.
- `python3 main.py --all-seasons` refreshes all configured seasons and merges them into the existing file.
- `python3 main.py --full-refresh` rebuilds the dataset from scratch for all configured seasons.

## Data preparation
- `python3 data_prep.py` calculates the fantasy points and adds the context regarding the opponents 

## Training
- `python3 train.py` 

## Prediction
- `python3 predict.py` predicts the top5 players for each position