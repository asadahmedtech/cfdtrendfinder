# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from . import twitterAnalysis
# Create your views here.
def get_trend_from_twitter(keyword):
	tweets,pos,neg = twitterAnalysis.getRes(keyword)
	data = [tweets[0],tweets[1],tweets[2],pos,neg]
	return data
def index(request):
	return render(request, 'trendfinder/index.html')

def gettrend(request):
	if request.method == 'POST':
		tweet = request.POST.get('keyword')
		trend_data = get_trend_from_twitter(tweet)
		return render(request, 'trendfinder/show.html', {'data':trend_data})
	return render(request, 'trendfinder/index.html')