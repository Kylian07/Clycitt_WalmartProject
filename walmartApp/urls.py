from walmartApp import views
from django.urls import path


urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.index, name='home'),
    path('product/<str:product_id>/', views.product_detail, name='product_detail'),
    path('buy/<str:product_id>/', views.buy, name='buy'),
    # path('cart/', views.index, name='cart'),
    # path('checkout/', views.checkout, name='checkout'),
    path('sell/',views.sell, name='sell'),
    path('search/', views.search, name='search'),
    # path('login/', views.login, name='login'),
    # path('register/', views.register, name='register'),
    # path('logout/', views.logout, name='logout'),
    # path('profile/', views.profile, name='profile'),
    # path('order_history/', views.order_history, name='order_history'),
    # path('wishlist/', views.wishlist, name='wishlist'),
    # path('add_to_cart/<int:product_id>/', views.add_to_cart, name='add_to_cart'),
    # path('remove_from_cart/<int:product_id>/', views.remove_from_cart, name='remove_from_cart'),
    # path('add_to_wishlist/<int:product_id>/', views.add_to_wishlist, name='add_to_wishlist'),
]