import random
import json
import math

if __name__ == '__main__':
    with open('data/recipes_with_nutritional_info.json') as f:
        # with open('data/instructions.txt', 'w') as f_out:
        #     recipes = json.load(f)
        #     for recipe in recipes:
        #         for instruction in recipe['instructions']:
        #             f_out.write(instruction['text'] + '\n')
        #     print('finished')
        recipes = []
        recipes_json = json.load(f)
        for recipe_json in recipes_json:
            lines = []
            for instruction in recipe_json['instructions']:
                lines.append(instruction['text'])
            recipes.append(lines)
        random.shuffle(recipes)
        size = len(recipes)
        # train_recipes = recipes[:math.floor(size * 0.8)]
        train_recipes = recipes[:1000]
        # val_recipes = recipes[math.floor(size * 0.8):math.floor(size * 0.9)]
        val_recipes = recipes[1000:1100]
        # test_recipes = recipes[math.floor(size * 0.9):]
        test_recipes = recipes[1100:1200]
        with open('data/train.txt', 'w') as f_out:
            for lines in train_recipes:
                for line in lines:
                    f_out.write(line + '\n')
                f_out.write('\n\n\n')
        with open('data/valid.txt', 'w') as f_out:
            for lines in val_recipes:
                for line in lines:
                    f_out.write(line + '\n')
                f_out.write('\n\n\n')
        with open('data/test.txt', 'w') as f_out:
            for lines in test_recipes:
                for line in lines:
                    f_out.write(line + '\n')
                f_out.write('\n\n\n')
        print('finished')
