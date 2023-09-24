This repository contains code, data and output for Task 2.

## Data:

The dataset used in this task is [goEmotion](https://github.com/google-research/google-research/tree/master/goemotions), it's a text based emotion classification dataset. 

According to the requirements of the task, I used two different version of dataset: 1) contains every individuals' vote, here called dataset_raw. 2) contain samples with selected label by taking the majority vote from annotators, here called dataset_majority_vote.

for dataset_raw, I used the raw data provided in the repo, and random split train, val and test according to the ratio of 7:2:1.

for dataset_majority_vote, since they have already done the splitting, I just take it as it is.

originally the dataset has 27+1 emotion labels (27 + neutral), and the author further broken down these 28 into (6 + neutral). Since we only take care of six emotions: anger, disgust, fear, joy, sadness, and surprise, the samples with only neutral as label are removed.

### Statistics of data:

Below are the sample counts for each category of two datasets.

|           -           | anger | disgust | fear |  joy  | sadness | surprise |
| :-------------------: | :---: | :-----: | :--: | :---: | :-----: | :------: |
|      dataset_raw      | 28802 |  3420   | 3460 | 79436 |  14494  |  22904   |
| dataset_majority_vote | 6724  |   738   | 790  | 21171 |  3234   |   5584   |

## Model:

Since it's a text based classification task, pretrained transformer based model will be one of the good options, because it excels at understanding contextual relationships in text, benefits from large-scale text pretraining.

In the repo of goEmotion, Bert-base is implemented as one of the base line method, here I chose another transformer based model GPT2.

I first tried customized GPT2 (cuz I'm facing hardware resource problem), I reduced the size of the network by altering the configs, but it turn out that if it's not pretrained, it's really difficult to train a new transformer model with acceptable performance.

Finally I used pretrained version of GPT2. Code and setting of these models can be found in `./models`.

## Performance

Below are the performance in terms of F1 score on the **test set** of two datasets and the performance of Bert-base reported in the paper:

|           -           | anger | disgust | fear | joy  | sadness | surprise | overall |
| :-------------------: | :---: | :-----: | :--: | :--: | :-----: | :------: | :-----: |
|      dataset_raw      | 0.73  |  0.00   | 0.55 | 0.92 |  0.60   |   0.72   |  0.47   |
| dataset_majority_vote | 0.81  |  0.55   | 0.83 | 0.96 |  0.71   |   0.77   |  0.66   |
|       Bert-base       | 0.57  |  0.53   | 0.68 | 0.82 |  0.66   |   0.59   |  0.64   |

## Discuss and further improvement

Observing the performance on dataset_raw and dataset_majority vote we can see that the model perform much better on dataset_majority than on dataset_raw in terms of both on individual emotions and overall score, this indicates that the outliers in dataset_raw actually have a significant impact on performance, substantial improvement will be achieved when we remove them.

Besides, from the sample counts we know that the number of 'disgust' and 'fear' are the lowest two among all emotions, however, the performance of these two models differs significantly. For 'disgust', non of them are predicted correctly on dataset_raw (I trained three models with different random seed, but they all failed to predict 'disgust'), but for 'fear' we can observe acceptable score. This suggests that the failure in predicting 'disgust' may not be due to a lack of data but rather due to significant variations in how annotators assess this emotion. 

When comparing to the Bert-base model reported in the paper, we can see that GPT2 performs a bit better overall, and also in terms of individual emotions, also the patterns are also similar. Thus, we can conclude that bidirectional and unidirectional transformers should exhibit relatively similar performance in sequence classification tasks (at least in terms of emotion classification), however, the overall differences in sequence classification between these two models still need to be validated through different tasks.

One remaining question is that the individual emotion performances of GPT2 are all much better than of Bert-base, but the overall performance is just a bit better, this might be related to the metric calculation, here I used f1_score implemented in sklear (average="binary" for each emotions, average="macro" for overall), not sure what they used in the paper.

About further improvement, I have two directions based on the training of the model:

- The author also provided another labels for these text, they are **positive**, **negative** and **ambiguous**, hence I think maybe we can try Coarse-to-fine strategy, first train to model to identify lower level of emotions, after the model gets the understanding of these, then train it on real emotions.
- Since GPT2 was not pretrained particularly for emotion recognition, in order to get better performance on this, maybe we can try to first train it on relative large scale of emotion related task, for example: identifying if two sentences are spoken in the same mood.



Corresponding tensorboard files are included in this repo, please check `./log`.