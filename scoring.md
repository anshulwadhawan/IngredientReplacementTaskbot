# Evaluation Metric

## Overall Precision and Recall 
### Description
Outputs the **precision** (number of correct substitutes / number of total substitutes predicted) and the **recall** (number of substitutes in the ground truth that are predicted / number of total substitutes in the ground truth).

### Script
```
python3 evaluate.py -p [path_to_prediction] -g [path_to_ground_truth]
```

### Sample output
```
Precision: 0.998468606431853, Recall:0.998468606431853
```

## Precision and Recall for each ingredient
### Description
Outputs the precision and recall for each individual ingredient in the file `evaluations/metrics_by_ingredient.json`.

### Script
```
python3 evaluate.py -p [path_to_prediction] -g [path_to_ground_truth] -each True
```

### Sample output
```
{
  "american cheese": {
    "precision": 0.9705882352941176,
    "recall": 0.9705882352941176
  },
  "apple juice": {
    "precision": 1.0,
    "recall": 1.0
  },
}
```

## Get the missing substitutes in prediction
### Description
Outputs the missing substitutes for each ingredient in the file `evaluations/missing_ingredients.json`

### Script
```
python3 evaluate.py -p [path_to_prediction] -g [path_to_ground_truth] -gm True
```

### Sample output
```
{
  "american cheese": {
    "cheddar cheese"
  },
  "apple juice": {
    "apple cider",
    "pear juice"
  },
}
```

## Get the wrong substitutes in prediction
### Description
Outputs the wrong substitutes for each ingredient in the file `evaluations/wrong_ingredients.json`

### Script
```
python3 evaluate.py -p [path_to_prediction] -g [path_to_ground_truth] -gw True
```

### Sample output
```
{
  "american cheese": {
    "apple cider",
    "pear juice"
  },
  "apple juice": {
    "cheddar cheese"
  },
}
```
