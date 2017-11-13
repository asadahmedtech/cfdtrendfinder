from tweepy import Stream
from tweepy import API
from tweepy import Cursor
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s



#consumer key, consumer secret, access token, access secret.
ckey="WX7vpbOZwgXUvLQuYjKhvteWK"
csecret="k8zVSlPFosjSHocQbJ3EaEyHQOeW8BOB6HPb9UXsIVkRHT81MP"
atoken="2835133730-kS6Q2Wu5e2j1147LtfJvS9FbE30KkHiIvSgNHKD"
asecret="bR4yDcdy75iG0ifFxgTKJeVMHOIGd9siHkMtHvP5tYrQh"

tweets = []
class listener(StreamListener):
    
    def on_data(self, data):
        try:
            all_data = json.loads(data)

            tweet = all_data["text"]
            sentiment_value, confidence = s.sentiment(tweet)
            print(tweet, sentiment_value, confidence)
            tweets.append([tweet,sentiment_value])
            i+=1;
            if confidence*100 >= 80:
                output = open("twitter-out.txt","a")
                output.write(sentiment_value)
                output.write('\n')
                output.close()

            return True
        except:
            return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
api = API(auth)

def getRes(keyword):
    items = 100
    pos,neg=0,0
    tweets = Cursor(api.search, q=keyword).items(items)
    t = []
    i=0
    for tweet in tweets:
        if (i<4):
            if('\u'not in tweet.text):

                t.append(tweet.text)
                i+=1
        a = s.sentiment(tweet.text)
        print a
        if a[0] =='pos':
            pos+=1
        else:
            neg+=1
    data = tuple()
    data += (t,)
    data += (pos,neg)
    return data
    # while(i<100):
    #     try:
    #         print "Getting Tweet #",i
            
    #         if(s.sentiment(tweets[i])=='pos'):
    #             pos += 1

    #         else:
    #             neg +=1
    #         i+=1;
    #     except Exception as e:
    #         print str(e)
    #         break
    # return(tweets,pos,neg)
