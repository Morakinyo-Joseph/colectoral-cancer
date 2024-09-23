from django.urls import path
from . import views


urlpatterns = [
    path("", views.home, name="home"),
    path("imaging", views.predict_segmentation, name="imaging"),
    path("clinical", views.clinical_prediction, name="clinical"),    
    path("gene", views.gene_expression_prediction, name="gene"),
]