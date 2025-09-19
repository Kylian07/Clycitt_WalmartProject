# from Search_engine.model import NeuralNet
# import torch
# from Search_engine.nltk_utils import bag_of_words, tokenize
# import random
# import json
# import os

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# # script_dir = os.path.dirname(os.path.abspath(__file__))
# # file_path = os.path.join(script_dir, 'carbonValue.json')
# with open('Search_engine/carbonValue.json', 'r') as json_data:
#     intents = json.load(json_data)


# FILE = "Search_engine/data.pth"
# data = torch.load(FILE)


# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# # print(input_size, hidden_size, output_size, tags)

# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()


# def search_engine(query):
#     sentence = tokenize(query)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)

#     _, predicted = torch.sort(output, dim=1, descending=True)

#     tag_list = [tags[i.item()] for i in predicted[0]]

#     results = []

#     for intent in intents:
#         for tag in tag_list:
#             if tag == intent["id"]:
#                 prob = torch.softmax(output, dim=1)[0][predicted[0][tag_list.index(tag)]]
#                 if prob > 0.005:
#                     results.append(intent)

#     results = sorted(results, key=lambda x: x['carbon_footprint_kg'], reverse=False)

#     return results




import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import torch
from transformers import AutoTokenizer, AutoModel
import re
import json

import nltk
nltk.download('punkt')
nltk.download('stopwords')


class NLP_Search_Engine:
    def __init__(self, data:pd.DataFrame, /, top_k:int = 10, model_name = 'distilbert-base-uncased', device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        self.model = model.to(device)
        self.device = device
        self.top_k = top_k
        # Load and preprocess data
        # file_path = 'carbonValue.json'
        self.processed_texts, self.original_texts, self.df = self.load_data(data)

        # Encode texts
        self.embeddings = self.encode_texts(self.processed_texts, self.tokenizer, self.model, self.device)

    def preprocess_text(self, text:str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Join tokens back to string
        return ' '.join(tokens)
    
    # Load and preprocess dataset
    def load_data(self, df:pd.DataFrame):
        texts = df['description'].tolist()
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        return processed_texts, texts, df
    
    
    # Encode texts into embeddings using PyTorch transformer
    def encode_texts(self, texts, tokenizer, model, device='cpu'):
        model.eval()
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {key: val.to(device) for key, val in inputs.items()}
                outputs = model(**inputs)
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])
        return np.array(embeddings)

    # Compute cosine similarity and search
    def _search(self, query, processed_texts, original_texts, embeddings, tokenizer, model, top_k=5, device='cpu'):
        # Preprocess and encode query
        processed_query = self.preprocess_text(query)
        inputs = tokenizer(processed_query, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            query_embedding = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        # Compute cosine similarity
        # Normalize embeddings for cosine similarity
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(norm_embeddings, norm_query)
        
        # Get top-k results
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(original_texts[i], similarities[i]) for i in top_k_indices]
        return results
    
    def search(self, query:str):
        results = self._search(query, self.processed_texts, self.original_texts, self.embeddings, self.tokenizer, self.model, top_k=self.top_k, device=self.device)
        return pd.concat([self.df.loc[self.df['description'].str.contains(text)] for text, distance in results], axis=0)

if __name__ == "__main__":
    data = pd.read_json('Search_engine\carbonValue.json')
    searchEngine = NLP_Search_Engine(data, top_k=5)
    r = searchEngine.search("Organic yellow cotton top with a comfortable fit.")
    records = r.to_dict('records')
    print(records)
    # print(json.dumps(records, indent=2))


    
# class NLP_Search_Engine:
#     def __init__(self, data:pd.DataFrame, /, top_k:int = 10, model_name = 'distilbert-base-uncased', device = 'cuda' if torch.cuda.is_available() else 'cpu'):
#         self.data = [
#     {
#         "id": 1,
#         "name": "White Piñatex Activewear",
#         "category": "Activewear",
#         "brand": "Girlfriend Collective",
#         "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff",
#         "carbon_footprint_kg": 7.9,
#         "price": 135.63,
#         "materials": [
#             "Cork",
#             "Tencel Lyocell"
#         ],
#         "features": [
#             "Biodegradable",
#             "Carbon Neutral",
#             "Recycled Materials"
#         ],
#         "color": "Blue",
#         "size": "M",
#         "description": "White Piñatex Activewear Girlfriend Collective Activewear Biodegradable Carbon Neutral Recycled Materials Blue Cork Tencel Lyocell M"
#     },
#     {
#         "id": 2,
#         "name": "Green Cork Bottoms",
#         "category": "Bottoms",
#         "brand": "Levi's",
#         "image_url": "https://images.unsplash.com/photo-1575428652377-a2d80e2277fc",
#         "carbon_footprint_kg": 29.3,
#         "price": 26.38,
#         "materials": [
#             "Apple Leather"
#         ],
#         "features": [
#             "Carbon Neutral",
#             "Water Saving",
#             "Recycled Materials"
#         ],
#         "color": "Yellow",
#         "size": "XS",
#         "description": "Green Cork Bottoms Levi's Bottoms Carbon Neutral Water Saving Recycled Materials Yellow Apple Leather XS"
#     },
#     {
#         "id": 3,
#         "name": "Yellow Hemp Activewear",
#         "category": "Activewear",
#         "brand": "Girlfriend Collective",
#         "image_url": "https://images.unsplash.com/photo-1575428652377-a2d80e2277fc",
#         "carbon_footprint_kg": 4.8,
#         "price": 85.88,
#         "materials": [
#             "Organic Cotton",
#             "Piñatex"
#         ],
#         "features": [
#             "Biodegradable",
#             "Vegan",
#             "Recycled Materials"
#         ],
#         "color": "Yellow",
#         "size": "S",
#         "description": "Yellow Hemp Activewear Girlfriend Collective Activewear Biodegradable Vegan Recycled Materials Yellow Organic Cotton Piñatex S"
#     },
#     {
#         "id": 4,
#         "name": "Beige Recycled Wool Footwear",
#         "category": "Footwear",
#         "brand": "Veja",
#         "image_url": "https://images.unsplash.com/photo-1611312449408-fcece27cdbb7",
#         "carbon_footprint_kg": 19.6,
#         "price": 113.94,
#         "materials": [
#             "Organic Cotton",
#             "Recycled Wool"
#         ],
#         "features": [
#             "Vegan",
#             "Carbon Neutral"
#         ],
#         "color": "Yellow",
#         "size": "L",
#         "description": "Beige Recycled Wool Footwear Veja Footwear Vegan Carbon Neutral Yellow Organic Cotton Recycled Wool L"
#     },
#     {
#         "id": 5,
#         "name": "Red Recycled Wool Footwear",
#         "category": "Footwear",
#         "brand": "Allbirds",
#         "image_url": "https://images.unsplash.com/photo-1551488831-00ddcb6c6bd3",
#         "carbon_footprint_kg": 17.1,
#         "price": 108.26,
#         "materials": [
#             "Organic Cotton",
#             "Recycled Wool"
#         ],
#         "features": [
#             "Vegan",
#             "Recycled Materials"
#         ],
#         "color": "Navy",
#         "size": "S",
#         "description": "Red Recycled Wool Footwear Allbirds Footwear Vegan Recycled Materials Navy Organic Cotton Recycled Wool S"
#     },
#     {
#         "id": 6,
#         "name": "Blue Tencel Lyocell Footwear",
#         "category": "Footwear",
#         "brand": "Veja",
#         "image_url": "https://images.unsplash.com/photo-1584917865442-de89df76afd3",
#         "carbon_footprint_kg": 11.3,
#         "price": 34.72,
#         "materials": [
#             "Apple Leather"
#         ],
#         "features": [
#             "Vegan",
#             "Biodegradable"
#         ],
#         "color": "Navy",
#         "size": "S",
#         "description": "Blue Tencel Lyocell Footwear Veja Footwear Vegan Biodegradable Navy Apple Leather S"
#     },
#     {
#         "id": 7,
#         "name": "Blue Apple Leather Outerwear",
#         "category": "Outerwear",
#         "brand": "The North Face",
#         "image_url": "https://images.unsplash.com/photo-1551488831-00ddcb6c6bd3",
#         "carbon_footprint_kg": 16.2,
#         "price": 114.52,
#         "materials": [
#             "Cork"
#         ],
#         "features": [
#             "Biodegradable"
#         ],
#         "color": "Blue",
#         "size": "XL",
#         "description": "Blue Apple Leather Outerwear The North Face Outerwear Biodegradable Blue Cork XL"
#     },
#     {
#         "id": 8,
#         "name": "Blue Tencel Lyocell Accessories",
#         "category": "Accessories",
#         "brand": "Pandora",
#         "image_url": "https://images.unsplash.com/photo-1581044777550-4cfa60707c03",
#         "carbon_footprint_kg": 0.7000000000000001,
#         "price": 92.36,
#         "materials": [
#             "Hemp"
#         ],
#         "features": [
#             "Biodegradable",
#             "Recycled Materials"
#         ],
#         "color": "Black",
#         "size": "L",
#         "description": "Blue Tencel Lyocell Accessories Pandora Accessories Biodegradable Recycled Materials Black Hemp L"
#     },
#     {
#         "id": 9,
#         "name": "Beige Bamboo Viscose Outerwear",
#         "category": "Outerwear",
#         "brand": "The North Face",
#         "image_url": "https://images.unsplash.com/photo-1589674781759-1ddc97ddfd5c",
#         "carbon_footprint_kg": 20.2,
#         "price": 252.89,
#         "materials": [
#             "Piñatex",
#             "Hemp"
#         ],
#         "features": [
#             "Vegan"
#         ],
#         "color": "Green",
#         "size": "L",
#         "description": "Beige Bamboo Viscose Outerwear The North Face Outerwear Vegan Green Piñatex Hemp L"
#     },
#     {
#         "id": 10,
#         "name": "Gray Recycled PET Dresses",
#         "category": "Dresses",
#         "brand": "Reformation",
#         "image_url": "https://images.unsplash.com/photo-1597700254310-3a5379c9ec45",
#         "carbon_footprint_kg": 9.2,
#         "price": 247.33,
#         "materials": [
#             "Recycled PET"
#         ],
#         "features": [
#             "Water Saving",
#             "Vegan"
#         ],
#         "color": "Gray",
#         "size": "L",
#         "description": "Gray Recycled PET Dresses Reformation Dresses Water Saving Vegan Gray Recycled PET L"
#     },
#     {
#         "id": 11,
#         "name": "Navy Apple Leather Activewear",
#         "category": "Activewear",
#         "brand": "Lululemon",
#         "image_url": "https://images.unsplash.com/photo-1589674781759-1ddc97ddfd5c",
#         "carbon_footprint_kg": 10.9,
#         "price": 283.72,
#         "materials": [
#             "Apple Leather",
#             "Cork"
#         ],
#         "features": [
#             "Recycled Materials",
#             "Vegan",
#             "Fair Trade Certified"
#         ],
#         "color": "Black",
#         "size": "XS",
#         "description": "Navy Apple Leather Activewear Lululemon Activewear Recycled Materials Vegan Fair Trade Certified Black Apple Leather Cork XS"
#     },
#     {
#         "id": 12,
#         "name": "White Recycled PET Activewear",
#         "category": "Activewear",
#         "brand": "Lululemon",
#         "image_url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab",
#         "carbon_footprint_kg": 7.1,
#         "price": 180.26,
#         "materials": [
#             "Cork",
#             "Hemp"
#         ],
#         "features": [
#             "Fair Trade Certified",
#             "Recycled Materials"
#         ],
#         "color": "Black",
#         "size": "XL",
#         "description": "White Recycled PET Activewear Lululemon Activewear Fair Trade Certified Recycled Materials Black Cork Hemp XL"
#     },
#     {
#         "id": 13,
#         "name": "Green Cork Activewear",
#         "category": "Activewear",
#         "brand": "Lululemon",
#         "image_url": "https://images.unsplash.com/photo-1539003947677-9b44fb7d2a8c",
#         "carbon_footprint_kg": 8.4,
#         "price": 80.13,
#         "materials": [
#             "Piñatex",
#             "Recycled PET"
#         ],
#         "features": [
#             "Carbon Neutral"
#         ],
#         "color": "White",
#         "size": "M",
#         "description": "Green Cork Activewear Lululemon Activewear Carbon Neutral White Piñatex Recycled PET M"
#     },
#     {
#         "id": 14,
#         "name": "Black Apple Leather Accessories",
#         "category": "Accessories",
#         "brand": "Stella McCartney",
#         "image_url": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea",
#         "carbon_footprint_kg": 5.4,
#         "price": 200.7,
#         "materials": [
#             "Organic Cotton",
#             "Hemp"
#         ],
#         "features": [
#             "Biodegradable",
#             "Vegan",
#             "Fair Trade Certified"
#         ],
#         "color": "White",
#         "size": "XS",
#         "description": "Black Apple Leather Accessories Stella McCartney Accessories Biodegradable Vegan Fair Trade Certified White Organic Cotton Hemp XS"
#     },
#     {
#         "id": 15,
#         "name": "White Hemp Activewear",
#         "category": "Activewear",
#         "brand": "Girlfriend Collective",
#         "image_url": "https://images.unsplash.com/photo-1575428652377-a2d80e2277fc",
#         "carbon_footprint_kg": 8.5,
#         "price": 88.4,
#         "materials": [
#             "Hemp",
#             "Recycled PET"
#         ],
#         "features": [
#             "Biodegradable",
#             "Carbon Neutral"
#         ],
#         "color": "Navy",
#         "size": "S",
#         "description": "White Hemp Activewear Girlfriend Collective Activewear Biodegradable Carbon Neutral Navy Hemp Recycled PET S"
#     },
#     {
#         "id": 16,
#         "name": "Black Cork Outerwear",
#         "category": "Outerwear",
#         "brand": "Patagonia",
#         "image_url": "https://images.unsplash.com/photo-1575428652377-a2d80e2277fc",
#         "carbon_footprint_kg": 11.0,
#         "price": 263.55,
#         "materials": [
#             "Recycled PET",
#             "Tencel Lyocell"
#         ],
#         "features": [
#             "Recycled Materials",
#             "Vegan"
#         ],
#         "color": "Red",
#         "size": "S",
#         "description": "Black Cork Outerwear Patagonia Outerwear Recycled Materials Vegan Red Recycled PET Tencel Lyocell S"
#     },
#     {
#         "id": 17,
#         "name": "White Bamboo Viscose Activewear",
#         "category": "Activewear",
#         "brand": "Lululemon",
#         "image_url": "https://images.unsplash.com/photo-1584917865442-de89df76afd3",
#         "carbon_footprint_kg": 10.8,
#         "price": 164.75,
#         "materials": [
#             "Recycled PET",
#             "Piñatex"
#         ],
#         "features": [
#             "Carbon Neutral",
#             "Recycled Materials",
#             "Water Saving"
#         ],
#         "color": "Beige",
#         "size": "XS",
#         "description": "White Bamboo Viscose Activewear Lululemon Activewear Carbon Neutral Recycled Materials Water Saving Beige Recycled PET Piñatex XS"
#     },
#     {
#         "id": 18,
#         "name": "White Recycled Wool Accessories",
#         "category": "Accessories",
#         "brand": "Stella McCartney",
#         "image_url": "https://images.unsplash.com/photo-1591561954557-26941169b49e",
#         "carbon_footprint_kg": 10.3,
#         "price": 32.99,
#         "materials": [
#             "Recycled Wool"
#         ],
#         "features": [
#             "Water Saving",
#             "Vegan"
#         ],
#         "color": "Gray",
#         "size": "M",
#         "description": "White Recycled Wool Accessories Stella McCartney Accessories Water Saving Vegan Gray Recycled Wool M"
#     },
#     {
#         "id": 19,
#         "name": "Gray Hemp Activewear",
#         "category": "Activewear",
#         "brand": "Lululemon",
#         "image_url": "https://images.unsplash.com/photo-1589674781759-1ddc97ddfd5c",
#         "carbon_footprint_kg": 7.4,
#         "price": 110.9,
#         "materials": [
#             "Apple Leather",
#             "Recycled PET"
#         ],
#         "features": [
#             "Biodegradable",
#             "Carbon Neutral"
#         ],
#         "color": "Beige",
#         "size": "M",
#         "description": "Gray Hemp Activewear Lululemon Activewear Biodegradable Carbon Neutral Beige Apple Leather Recycled PET M"
#     },
#     {
#         "id": 20,
#         "name": "Beige Bamboo Viscose Bottoms",
#         "category": "Bottoms",
#         "brand": "Thought",
#         "image_url": "https://images.unsplash.com/photo-1539003947677-9b44fb7d2a8c",
#         "carbon_footprint_kg": 9.0,
#         "price": 211.2,
#         "materials": [
#             "Recycled PET",
#             "Cork"
#         ],
#         "features": [
#             "Carbon Neutral"
#         ],
#         "color": "Black",
#         "size": "XL",
#         "description": "Beige Bamboo Viscose Bottoms Thought Bottoms Carbon Neutral Black Recycled PET Cork XL"
#     }]
#         self.data = pd.DataFrame.from_dict(data)
#     def search(self, query):
#         return self.data