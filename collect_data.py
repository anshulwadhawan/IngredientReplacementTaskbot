from train_test_split import read_data
import json
from collections import Counter

if __name__ == '__main__':
    ingredients, train, dev, test = read_data()
    ingredients_set = set(ingredients)

    """
    Only include the ingredients that are both mentioned in the book
    and in the ingredients set
    """
    # with open('data/replacement_suggestions.json') as f:
    #     ground_truth = {}
    #     replacement_suggestions = json.load(f)
    #     for key in replacement_suggestions:
    #         if key not in ground_truth:
    #             ground_truth[key] = []
    #         for value in replacement_suggestions[key]:
    #             if value in ingredients_set:
    #                 ground_truth[key].append(value)
    #         if len(ground_truth[key]) == 0:
    #             ground_truth.pop(key)
    #     print(ground_truth)
    # with open('data/ground_truth.json', "w") as f_out:
    #     json.dump(ground_truth, f_out, ensure_ascii=False)
    with open('data/ground_truth_substitutes_dict.json') as f:
        ground_truth = json.load(f)
        ground_truth_cleaned = {}
        for key in ground_truth:
            if key not in ingredients_set:
                continue
            ground_truth_cleaned[key] = ground_truth[key]
        with open('data/ground_truth_cleaned.json', 'w') as f_out:
            json.dump(ground_truth_cleaned, f_out, ensure_ascii=False)

    # See the most common ingredients in the training set
    # Pick the ingredients for evaluation
    # counter = Counter()
    # for recipe in train:
    #     for ingredient in recipe:
    #         counter[ingredient] += 1
    # for most_common_index in counter.most_common(35):
    #     print(ingredients[most_common_index[0]] + ", ")
    # for _ in range(25):
    #     random_index = random.choice(list(counter.keys()))
    #     print(ingredients[random_index] + ", ")
