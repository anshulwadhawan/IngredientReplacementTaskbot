import os
import sys

#sys.path.append(os.path.join(os.path.dirname(__file__),'repo/'))

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from foodbert_embeddings.helpers.approx_knn_classifier import ApproxKNNClassifier
from foodbert_embeddings.helpers.generate_ingredient_embeddings import generate_food_embedding_dict
from foodbert_embeddings.helpers.utils import clean_ingredient_name, clean_substitutes

from collections import OrderedDict
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-t", "--test-file-path", dest="test_file_path",
                    required=False, help="path to the test file")
parser.add_argument("-p", "--test-preds-path", dest="test_preds_path",
                    default='predicitons.json',
                    required=False, help="path to the model's prediction file")
args = parser.parse_args()

threshold = 0.48

def avg(values):
    summed = sum(values)
    length = len(values)
    return summed / length


def custom_potential_neighbors_sort(potential_neighbors):
    # First sort by how often something was nearby, if this is equal, use the smaller distance
    sorted_neighbors = sorted(potential_neighbors.items(), key=lambda x: (len(x[1]), -avg(x[1])), reverse=True)
    return sorted_neighbors


def filter_out_forbidden_neigbours(ingredient_name, potential_neighbors):
    '''
    Neigbors that are the same as the ingredient are to be removed, additional rules such as mozeralla & mozeralla_cheese, penne & penne_pasta can be added here
    '''
    banned_keys = {ingredient_name}

    # Ban ingredients that contain ingredient_name
    for ingredient in potential_neighbors.keys():
        if ingredient_name in ingredient.split('_'):
            banned_keys.add(ingredient)

    filtered_potential_neighbors = {key: value for key, value in potential_neighbors.items() if key not in banned_keys}

    return filtered_potential_neighbors


def get_nearest_N_neigbours(ingredient_name, ingredients_to_embeddings, all_ingredient_labels,
                            knn_classifier: ApproxKNNClassifier):
    if ingredient_name not in ingredients_to_embeddings:
        return None
    ingredient_embeddings = ingredients_to_embeddings[ingredient_name]
    all_distances, all_indices = knn_classifier.k_nearest_neighbors(ingredient_embeddings)

    potential_neighbors = defaultdict(list)

    for i in range(len(ingredient_embeddings)):
        labels = all_ingredient_labels[all_indices[i]]
        distances = all_distances[i]

        for label, distance in zip(labels, distances):
            potential_neighbors[label].append(distance)

    potential_neighbors = filter_out_forbidden_neigbours(ingredient_name, potential_neighbors)
    sorted_neighbors = custom_potential_neighbors_sort(potential_neighbors)
    sorted_neighbors = [(key, value) for key, value in sorted_neighbors if len(value) >= len(ingredient_embeddings)]  # remove too rare ones
    # further removal
    relative_lengths = [len(elem[1]) / (len(sorted_neighbors[0][1])) for elem in sorted_neighbors]
    final_neighbors = []
    for idx in range(len(relative_lengths)):
        if relative_lengths[idx] >= threshold:  # Currently doesn't sort anything out # TODO tune this
            final_neighbors.append(sorted_neighbors[idx])

    try:
        return list(zip(*final_neighbors))[0]

    except Exception as e:
        return None

name = 'foodbert' # or multimodal
substitute_pairs_path = Path(f'foodbert_embeddings/data/substitute_pairs_foodbert_{"text" if name == "foodbert" else "multimodal"}.json')
normalization_fixes_path = Path('foodbert_embeddings/data/normalization_correction.json')
max_embedding_count = 100
image_embedding_dim = 768

if normalization_fixes_path.exists():
    with normalization_fixes_path.open() as f:
        normalization_fixes = json.load(f)
else:
    normalization_fixes = {}

ingredients_to_embeddings = generate_food_embedding_dict(max_sentence_count=max_embedding_count)

if name == 'multimodal':
    with open("multimodal/data/embedding_dict.pth", "rb") as f:
        ingredients_to_image_embeddings = torch.load(f, map_location='cpu')

    # PCA for image embeddings
    X = [elem.cpu().numpy() for elem in ingredients_to_image_embeddings.values()]
    pca = PCA(n_components=image_embedding_dim)
    pca.fit(X)
    for key, image_embedding in ingredients_to_image_embeddings.items():
        if key not in ingredients_to_embeddings:
            continue
        pca_image_embedding = pca.transform(image_embedding.reshape(1, -1)).squeeze()
        original_embedding = ingredients_to_embeddings[key]
        pca_image_embedding = np.expand_dims(pca_image_embedding, axis=0).repeat(axis=0, repeats=len(original_embedding))
        pca_image_embedding = pca_image_embedding.astype(np.float32)
        ingredients_to_embeddings[key] = np.concatenate([original_embedding, pca_image_embedding / 2], axis=1)

all_ingredient_embeddings = []
all_ingredient_labels = []

for key, value in ingredients_to_embeddings.items():
    all_ingredient_embeddings.append(value)
    all_ingredient_labels.extend([key] * len(value))

all_ingredient_embeddings = np.concatenate(all_ingredient_embeddings)
all_ingredient_labels = np.stack(all_ingredient_labels)

knn_classifier: ApproxKNNClassifier = ApproxKNNClassifier(all_ingredient_embeddings=all_ingredient_embeddings,
                                                                                max_embedding_count=max_embedding_count)

print("SAMPLE OUTPUT")
 
print("Substitutes for salt: ")
print(get_nearest_N_neigbours(ingredient_name='salt', ingredients_to_embeddings=ingredients_to_embeddings,
                                          all_ingredient_labels=all_ingredient_labels, knn_classifier=knn_classifier))

print("Substitutes for sugar: ")
print(get_nearest_N_neigbours(ingredient_name='sugar', ingredients_to_embeddings=ingredients_to_embeddings,
                                          all_ingredient_labels=all_ingredient_labels, knn_classifier=knn_classifier))

print("Substitutes for honey: ")
print(get_nearest_N_neigbours(ingredient_name='honey', ingredients_to_embeddings=ingredients_to_embeddings,
                                          all_ingredient_labels=all_ingredient_labels, knn_classifier=knn_classifier))

print("Substitutes for pepperoni: ")
print(get_nearest_N_neigbours(ingredient_name='pepperoni', ingredients_to_embeddings=ingredients_to_embeddings,
                                          all_ingredient_labels=all_ingredient_labels, knn_classifier=knn_classifier))


if args.test_file_path:
    test_file_path = Path(args.test_file_path)
    test_preds_path = Path(args.test_preds_path)
    with test_file_path.open() as f:
            test = json.load(f)

    subtitute_pairs = OrderedDict()
    for ingredient_name in tqdm(test.keys(), total=len(test)):
        substitutes = get_nearest_N_neigbours(ingredient_name=ingredient_name.replace(" ", "_"), ingredients_to_embeddings=ingredients_to_embeddings,
                                              all_ingredient_labels=all_ingredient_labels, knn_classifier=knn_classifier)

        if substitutes is not None:
            substitutes = clean_substitutes(substitutes, normalization_fixes)
        
        subtitute_pairs[ingredient_name]= substitutes

    with test_preds_path.open('w') as f:
        json.dump(subtitute_pairs, f)

    print("Predicitons saved.")
    print(subtitute_pairs)