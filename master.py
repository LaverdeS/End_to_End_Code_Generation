import json
from pprint import pprint

with open('c:/Users/SEBASTIAN LAVERDE/Documents/GitHub/End_to_End_Code_Generation/conala-train.json') as f:
    data = json.load(f)

intents = []
snippets = []
j = 1

for i in data:
    intents.append(i['intent'])
    snippets.append(i['snippet'])

pprint(intents)
pprint(snippets)