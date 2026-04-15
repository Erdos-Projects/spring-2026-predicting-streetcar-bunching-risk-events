# Predicting streetcar bunching risk-events from risk-patterns:
## An ML approach to congestion on the TTC


 Heuristically, streetcars are "bunched" when a pair of streetcars arrive at a stop in quick succession. The scheduled headway between streetcars is the time gap between sucessive streetcars, according to a fixed schedule. A "bunching incident" happens when the observed headway between two streetcars in reality ends up being a time gap that is much lower than the scheduled headway. While there is not a strict definition of how much sooner than the scheduled headway counts as a "bunching incident", it should be calculated as the scheduled headway multiplied by some scale factor <<  1. Here, we take this scale factor to be 1/2, so that a bunching incident occurs when a streetcar arrives at a stop sooner than 50% of the scheduled headway.

### <u> Bunching episodes </u> 

As it is defined, a bunching incident will always involve a pair of streetcars. A pair of streetcars become bunched when they arrive at a stop in sufficiently quick succession from each other. When this happens, a ``bunching episode" begins for this pair of streetcars. Such a bunching episode may persist to the next stop if they also arrive at the immediate next stop bunched together. 

In this project, we will produce and train a collection of models that address the matter of predicting when ``bunching episodes" will persist for multiple stops. 



## <u> Project overview </u>

This project develops a flexible prediction tool for predicting bunching risk-events, from confirmed bunching risk-patterns. In this project a bunching risk-event can be specified by 4 integers (b,s,n,m). Then, the bunching risk-event prediction task is given by:

> Given a number of bunching incidents  b, observed by a pair of streetcars in the last s stops, predict if the same pair of streetcars will encounter another number of bunching incidents  n within the next m stops 

 For any risk-event, we call a collection of bunching incidents satisfying the conditional statement of the associated prediction task, a risk-pattern. Then, the prediction task for a type of risk-event is a binary classification problem: for each associated risk-pattern, assign a 1 if the risk-pattern extends to a risk-event, assign a 0 if it does not.


## Feature engineering


### <u> Summary of features to be used</u> 


Our prediction model will essentially use multiple "layers" of features. 

- First-order features: these features are directly calculated from the cleaned stop-level dataset, ```ev``` 

- Second-order features: from ```ev``` we will derive another dataframe ```contacts```, which essentially extracts bunched pairs of streetcars from ```ev``` (across all stops and headways). This dataframe will include an entry for a pairs of streetcars whenever a new bunching episode has begun. That is, a pair of vehicle IDs may correspond to distinct rows of this dataframe, if the pair of vehicles have multiple non-contiguous bunching episodes. This dataframe also contains features relating to these bunched pairs (for example, the absolute difference between the schedule adherence features of the streetcars in the pair).

- Third-order features: from ```contacts```, we will derive a third dataframe ```episodes```, built from the ```contacts``` dataframe. Each bunching incident (at a given stop) is either the start of a new bunching episode, or is the continuation of an existing bunching episode. These episodes are assigned an episode ID, and this dataframe tracks features such as how many stops the episode spans, how many minutes the episode has gone on for, and things of this nature.



- External features are also included. These include features related to distance to traffic signal and pedestrian crossings. We also include weather data, and data about active road permits near the streetcar stops. For distance based external features, we use the Haversine distance formula in order to accurately calculate distance between two geographic positions.

- An incident dataframe ```incident_df``` is built from episode-level rows, then enriched with external features. For each configured risk-task (b,s,n,m), the pipeline adds task-specific target, eligibility (i.e. flagging episodes that are part of the specified risk-pattern). The dataframe ```incident_df``` is also used to calculate targets (primarily for training purposes). 

- The main feature dataframe that our models will be fit with is called ```incident_X```, which is a sub-dataframe of ```incident_df```, consisting of valid bunching episodes satisfying the current definition of "risk-pattern". 


# Modeling overview

For each risk-event type,  we use XGBoost as our model for our binary classification prediction task. For training and testing purposes, we use a time-sensitive train/validation/test split our of data set. The splits are given by: 

- Training set: 2024/07/01 to 2025/09/30

   - A single buffer day 

- Validation set: 2025/10/01 to 2025/11/15

  - A single buffer day

- Test set: 2025/11/16 to 2026/03/11

We do a 4-fold cross-validation scheme for model training.  A hyperparameter random search is done for tuning XGboost hyperparameters (grid search and bayesian optimization hyperparameter tuning options are also available). Depending on the skewed-ness of the specific type of risk-event, we may choose different metrics for assessing performance on validation sets. Generally speaking, as we are leaning towards recall rate, we consider F2-score, with possibly a cap on false positive rate.  


# Data-loading

## Chunked Event Data

The data_files folder contains chunked versions of the large event files used by the pipeline.
All chunk files are kept below `45 MB` so they can be pushed to GitHub.

## Contents

- `manifest.json`: mapping from original filename to chunk files + SHA-256 hash
- `chunks/*.partNNNN.csv`: chunked CSV parts with repeated header rows
<!-- 
## Reassemble Files

From repo root:

```powershell
python scripts/data_file_chunks.py assemble --in-dir data_files --out-dir . --force
```

That reconstructs the original files in repo root with names expected by the pipeline:

- `df2025_all.csv` (primary)
- `506-2024-1.csv`, `df_2024-2`, `df_2026` (extras)
- `all506_df-new-1.csv`, `all506_df-new-2.csv` (fallback pair)

The script validates each reconstructed file against the source SHA-256 stored in `manifest.json`. -->

## Data-loading in the pipeline

The pipeline reads chunked event files directly from `data_files/manifest.json` by default.







