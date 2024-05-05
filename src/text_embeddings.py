"""
This script generates the embeddings for the selector options in the front-end.
"""

import itertools
from transformers import AutoTokenizer, CLIPModel
import pandas as pd

# define all options to be considered by the recomender, and their combinations
options = {
    'color' : ['white', 'black', 'brown', 'blue', 'green', 'red', 'orange', 'pink', 'gray', 'yellow', 'multicolor'],
    'person' : ['woman', 'man', 'kid'],
    'season' : ['winter', 'spring', 'summer', 'autum'],
    'type' : ['dress', 'trouser', 'short', 'shirt', 'jacket', 'jumper', 'skirt'],
    'pattern' : ['spotted', 'striped', 'plain', 'checkered'],
}
all_combinations = [' '.join(tup) for tup in itertools.product(*options.values())]

# define the model that will embed all combinations
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# use the model to get the embeddings
inputs = tokenizer(all_combinations, padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)

# prepare the csv with the result, through pd dataframes
embed_df = pd.DataFrame(text_features.detach().numpy(), columns = [f'embedding_{i}' for i in range(512)])
options_df = pd.DataFrame(itertools.product(*options.values()), columns = options.keys())
pd.concat([options_df, embed_df], axis=1).to_csv('embed_options.csv')