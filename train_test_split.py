import numpy as np


def read_data():
    with np.load('data/aggregate_data.npz', allow_pickle=True) as data:
        ingredients = data['ingredients']
        dev = data['dev']
        test = data['test']
        train = data['train']
    return ingredients, train, dev, test


if __name__ == '__main__':
    # data processing and train-test split: already done
    # with np.load('data/simplified-recipes-1M.npz', allow_pickle=True) as data:
    #     recipes = data['recipes']
    #     ingredients = data['ingredients']
    #     np.random.shuffle(recipes)
    #     train_size = math.floor(0.8 * len(recipes))
    #     dev_test_size = math.floor(0.1 * len(recipes))
    #     train, dev, test = recipes[:train_size], recipes[train_size:train_size +
    #                                                      dev_test_size], recipes[train_size + dev_test_size:]
    #     np.savez_compressed('data', ingredients=ingredients,
    #                         train=train, dev=dev, test=test)
    pass
