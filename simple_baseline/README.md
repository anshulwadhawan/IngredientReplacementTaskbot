## This is the code for our simple baseline. 

An interactive analysis of the FoodBERT embedding subspace, after reducing the embedding dimensions from 78600 to 3 using t-SNE, has been showcased in the below animation. The analysis shows different clusters like {onion, green onion, white onion}, {oil, cooking oil}, {salt, seasoning, garlic salt}, and {cream, heavy cream, ice cream}, each containing closely placed ingredients. This shows that similar ingredients tend to occupy similar spots in the embedding space even when dimensionality is reduced significantly.

![alt text](https://github.com/annypan/CIS-530-final-project/blob/main/simple_baseline/space.gif)

KNN model: Discussed is a KNN model that finds K nearest neighbors for a particular ingredient in the FoodBERT embedding space, and shortlists them on the basis of a thresholding criteria. The final approach is separated into two parts: The first part calculates text-based embeddings for up to 100 occurrences of every ingredient and optionally concatenates them with image-based embeddings. The second part employs these embeddings together with KNN and a further scoring and filtering step to predict substitutes.

Sample substitutes for four ingredients (salt, sugar, honey, pepperoni) are produced by the provided code.

On our test set, the model has a Precision of 0.784 and Recall of 0.139.

## To run the code, run the following set of commands:

### 1. Cloning the repository for FoodBERT embeddings
```
git clone https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution.git repo
```
### 2. Change current directory to get inside repo folder
```
cd repo
```

### 3. Commands to download data
```
    wget https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution/releases/download/0.1/foodbert_data.zip
    wget https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution/releases/download/0.1/foodbert_embeddings_data.zip
    wget https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution/releases/download/0.1/multimodal_data.zip
```

### 4. Unzipping zip files and creating required directory structure
```
    unzip -qq foodbert_data.zip
    unzip -qq foodbert_embeddings_data.zip
    unzip -qq multimodal_data.zip
    mv foodbert_data/* foodbert/data/
    mv foodbert_embeddings_data/* foodbert_embeddings/data/
    mv multimodal_data/* multimodal/data/
    rm -r foodbert_data
    rm -r foodbert_embeddings_data
    rm -r multimodal_data
```

### 5. Installing required libraries
```
pip install -r requirements.txt
```
### 6. Installing appropriate versions for torch and torchvision
```
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

### 7. Make sure to be inside repo to run simple-baseline.py. 
### test_file_path.json denotes path of the test file; output_predicitons_file_path.json denotes path of the output file.

```
python simple-baseline.py --test-file-path test_file_path.json --test-preds-path output_predicitons_file_path.json
```

## To evaluate the produced predictions, use the output predicitons file produced above.
