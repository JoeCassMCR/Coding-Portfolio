# Sentiment Analysis 

This project classifies political tweet sentiment during the 2020 US Presidential election using traditional and transformer-based models. The pipeline evaluates:
- A custom neural network trained on TF-IDF features
- Pretrained DistilBERT and BERTweet transformer models
- Rule-based VADER sentiment scoring

## Dataset
- Source: [`munawwarsultan2017/US_Presidential_Election_2020_Dem_Rep`](https://huggingface.co/datasets/munawwarsultan2017/US_Presidential_Election_2020_Dem_Rep)

## Models Evaluated
| Model               | Description                        |
|--------------------|------------------------------------|
| Custom Neural Net  | TF-IDF + 2-layer dense net          |
| DistilBERT         | Pretrained on SST-2 sentiment data |
| BERTweet           | Pretrained on Twitter corpus        |
| VADER              | Rule-based lexicon                  |

## Results

| Model        | F1 Score | MCC     |
|--------------|----------|---------|
| Neural Net   | *0.8623024194756197*   | *0.7934810552923607*  |
| DistilBERT   | *0.3086590072119339*   | *0.20482974434122708*  |
| BERTweet     | *0.1718068255204697*   | *0.00*  |
| VADER        | *0.2446533754716227*   | *0.05564685113045962*  |

See visual comparisons in [`assets/`](./assets/)

## Highlights

- Benchmarks transformer models vs traditional approaches
- Includes full preprocessing and training pipeline
- Stores evaluation in `results.json`
- Clean code with logging and error handling

## Running the Code

```bash
pip install -r requirements.txt
python sentiment_analysis.py
```
## Report

You can read the full research write-up in [Sentiment Analysis Report](./Sentiment%20Analysis/Sentiment%20Analysis.pdf)



