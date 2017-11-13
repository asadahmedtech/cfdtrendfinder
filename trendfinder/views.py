# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from . import twitterAnalysis
# Create your views here.
def get_trend_from_twitter(keyword):
	tweets,pos,neg = twitterAnalysis.getRes(keyword)
	data,sentiment = [tweets[0],tweets[1],tweets[2]],['Negative Sentiment is '+str(neg),'Positive Sentiment is '+str(pos)]
	return (data,sentiment)
def index(request):
	return render(request, 'trendfinder/index.html')

def gettrend(request):
	if request.method == 'POST':
		tweet = request.POST.get('keyword')
		trend_data,sentiment = get_trend_from_twitter(tweet)
		return render(request, 'trendfinder/show.html', {'data':trend_data,'sentiment':sentiment,'keyword':tweet})
	return render(request, 'trendfinder/index.html')