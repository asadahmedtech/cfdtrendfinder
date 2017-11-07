from django.conf.urls import url, include
from . import views

print 'okays'
urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^gettrend/', views.gettrend, name='gettrend')
]
