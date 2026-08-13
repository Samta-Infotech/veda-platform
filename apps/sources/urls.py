from django.urls import path
from apps.sources.views import DataSourceListView

urlpatterns = [
    path("data-sources/list", DataSourceListView.as_view(), name="data-source-list"),
]
