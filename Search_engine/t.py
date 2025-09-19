import json


with open('carbonValue.json', 'r') as file:
    data = json.load(file)

print("Number of items:", len(data))

print("First item:", data[0])
print("Item keys:", data[0].keys(), "\nItem types:", [type(value) for value in data[0].values()])

for item in data:
    # print("Item keys:", item.keys(), "\nItem types:", [type(value) for value in item.values()])
    item['pattern'] = ' '.join([item['name'], item['brand'], item['category'], ' '.join(item['features']), item['color'], ' '.join(item['materials']), item['size'], item['description']])

print("First item after modification:", data[0])

with open('carbonValue.json', 'w') as file:
    json.dump(data, file)