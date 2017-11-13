import sentiment_mod as s

def foo(text):
	modtext = ''
	for i in text.split():
		if "\u" not in i:
			modtext+=i

	return i
text = foo("\u2342 idk thor ws awsme bleh \u2131")
print(s.sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
print(s.sentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))
print(s.sentiment(text))
 
print((s.sentiment('the movie was good on one side bad on other side depends on the side you chose to be on')))