from django.shortcuts import render, redirect
from walmartApp.search_engine import NLP_Search_Engine  # Adjust the import path as necessary
from django.db import models

import pandas as pd
import json
import random
# from Search_engine.chat import chat  # Adjust the import path as necessary

point_bar = 50

class Query_Queue:
    def __init__(self, size=10):
        self.queue: list[dict] = []
        self.size = size

    def add_query(self, query):
        if len(self.queue) >= self.size:
            self.queue.pop(0)  # Remove the oldest query
        for item in self.queue:
            if item['query'] == query['query']:
                return None
        self.queue.append(query)

    def get_queries(self):
        return self.queue
    
    def clear_queries(self):
        self.queue.clear()

    def search(self, query, csrf_token=None):
        # self.add_query(query)
        # return search_engine(query)
        for item in self.queue:
            if item['query'] == query and item['csrf_token'] == csrf_token:
                return item['results']
        return None
    
# query_queue = Query_Queue()

data = pd.read_json('Search_engine\carbonValue.json')
print(data.head())
search_engine = NLP_Search_Engine(data)

with open('Search_engine\carbonValue.json', 'r') as json_data:
    intents = json.load(json_data)

print(f"Loaded intents: {len(intents)} items")

# Create your views here.
def index(request):
    return render(request, 'index.html', { 'product_list' : sorted(random.sample(intents, 7), key=lambda x: x['carbon_footprint_kg'], reverse=False), 'point':point_bar})


def product_detail(request, product_id):
    # product = intents.find(lambda x: x['id'] == product_id, None)
    print(f"Product ID: {product_id}")
    product = None
    for product in intents:
        if str(product['id']) == str(product_id):
            product = product
            break

    print(f"Product found: {product}")
    if not product:
        return render(request, 'product_details.html', { 'error': 'Product not found'})
    
    product_list = search_engine.search(' '.join([product['name'], product['category'], product['brand'], product['color']]))
    product_list = product_list.to_dict('records')
    product_list = list(filter(lambda d: d.get('id') != product['id'], product_list))
    product_list = sorted(product_list, key=lambda x: x['carbon_footprint_kg'], reverse=False)

    return render(request, 'product_details.html', {'product_id': product_id, "product": product, 'product_list': product_list})
    # try:
    #     product = Product.objects.get(id=product_id)

def cart(request):
    return render(request, 'cart.html') 

def checkout(request):
    return render(request, 'checkout.html')

def search(request):
    query = request.GET.get('search', '')
    print(f"Search query: {query}")

    results = search_engine.search(query)
    return render(request, 'search.html', {'query': query, 'products': sorted(results.to_dict('records'), key=lambda x: x['carbon_footprint_kg'], reverse=False)})

def login(request):
    if request.method == 'POST':
        # Handle login logic here
        pass
    return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        # Handle registration logic here
        pass
    return render(request, 'register.html')

def logout(request):
    # Handle logout logic here
    return render(request, 'logout.html')

def profile(request):
    return render(request, 'profile.html')

def order_history(request):
    return render(request, 'order_history.html')

def wishlist(request):
    return render(request, 'wishlist.html')

def add_to_cart(request, product_id):
    # Handle adding to cart logic here
    return render(request, 'cart.html')

def remove_from_cart(request, product_id):
    # Handle removing from cart logic here
    return render(request, 'cart.html')

def add_to_wishlist(request, product_id):
    # Handle adding to wishlist logic here
    return render(request, 'wishlist.html')

def buy(request, product_id):
    product = None
    for product in intents:
        if str(product['id']) == str(product_id):
            product = product
            break

    global point_bar
    if point_bar < round(product['carbon_footprint_kg']):
        product_list = search_engine.search(' '.join([product['name'], product['category'], product['brand'], product['color']]))
        product_list = product_list.to_dict('records')
        product_list = list(filter(lambda d: d.get('id') != product['id'], product_list))
        product_list = sorted(product_list, key=lambda x: x['carbon_footprint_kg'], reverse=False)

        return render(request, 'product_details.html', {'product_id': product_id, 'error': 'Your Recycle point is low, You cant use recycle point for this product', "product": product, 'product_list': product_list})

    point_bar-= round(product['carbon_footprint_kg'])
    # return render(request, 'index.html', { 'product_list' : random.sample(intents, 7), 'point':point_bar})
    return redirect('/home/')

def sell(request):
    if request.method == 'POST':
        amount=request.POST['password']
        print(amount)
        global point_bar
        point_bar+=int(amount)
        # return render(request, 'index.html', { 'product_list' : random.sample(intents, 7), 'point':point_bar})
        # return redirect('/')
        return redirect('/home/')
    else:
        return render(request, 'login_page.html')