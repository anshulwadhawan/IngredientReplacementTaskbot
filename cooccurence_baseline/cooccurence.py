import numpy as np
import json


def predict(cooc_mat, ingredients):
    targets = ['american cheese', 'apple juice',
               'bacon', 'ground beef', "blueberry", "bulgur", "carrot", "cheddar", "cherry", "chicken", "duck", "hazelnut",
               "jack cheese", "kale", "lamb", "lobster", "parsley", "peach", "peanut", "pineapple", "pork", "prosciutto", "raisin", "raspberry",
               "rice milk", "roquefort", "salmon", "strawberry", "thyme", "tuna", "turkey", "veal", "sugar", "mushroom", "eggplant", "chocolate", "tofu",
               "paneer"]
    res = {}
    for ingredient in targets:
        ingredient_idx = np.where(ingredients == ingredient)[0][0]
        cooc = cooc_mat[ingredient_idx]
        indices = np.argpartition(cooc, -5)[-5:]
        substitutes = []
        for substitute_index in indices:
            substitutes.append(ingredients[substitute_index])
        res[ingredient] = substitutes
    with open('predictions/cooc_predictions.json', 'w') as f_out:
        json.dump(res, f_out, ensure_ascii=False)
        print('predictions are ready')


if __name__ == '__main__':
    with np.load('data/aggregate_data.npz', allow_pickle=True) as data:
        ingredients = data['ingredients']
        dev = data['dev']
        test = data['test']
        train = data['train']
        cooc_mat = [[0 for _ in range(len(ingredients))]
                    for _ in range(len(ingredients))]
        for recipe in train:
            for fst in range(len(recipe)):
                fst_ing = recipe[fst]
                for snd in range(fst):
                    snd_ing = recipe[snd]
                    cooc_mat[fst_ing][snd_ing] += 1
                    cooc_mat[snd_ing][fst_ing] += 1
        for recipe in dev:
            for fst in range(len(recipe)):
                fst_ing = recipe[fst]
                for snd in range(fst):
                    snd_ing = recipe[snd]
                    cooc_mat[fst_ing][snd_ing] += 1
                    cooc_mat[snd_ing][fst_ing] += 1
        for recipe in test:
            for fst in range(len(recipe)):
                fst_ing = recipe[fst]
                for snd in range(fst):
                    snd_ing = recipe[snd]
                    cooc_mat[fst_ing][snd_ing] += 1
                    cooc_mat[snd_ing][fst_ing] += 1
        predict(cooc_mat / np.array(cooc_mat).sum(axis=1), ingredients)
