1.To access GPT3 api, create a .env file and enter the following (replace your-api-secret-key):
        OPENAI_API_KEY='your-api-secret-key'
2.pip install -r requirement.txt (in the same directory)
3.after running the jyputer notebook "strong_baseline.ipynb", 
   run "python3 evaluate.py -p [path_to_prediction] -g [path_to_ground_truth] -gm True" to obtain precision and recall on testset.