
import matplotlib.pyplot as plt
import re 
import pandas as pd
from sklearn.metrics import label_ranking_loss
from textblob import TextBlob 
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
n_words= set(stopwords.words('english'))
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from wordcloud import WordCloud,STOPWORDS
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer() 
nltk.download('punkt')
nltk.download('wordnet')
import numpy as np
from datetime import date
from wordcloud import WordCloud

pathe="/Users/sebastienbrown/Desktop/Job Search Summer 2023/Projects/Sentiment Analysis/Sentiment-Analysis-Bitcoin/Data/Bitcoin_tweets.csv"

path2="/Users/sebastienbrown/Desktop/Job Search Summer 2023/Projects/Sentiment Analysis/Sentiment-Analysis-Bitcoin/Data/Bitcoin.csv"


def timestamp_importer(path):
  dataset=pd.read_csv(path)
  subset=dataset[0:]
  subset['Date']=pd.to_datetime(subset['Date'])
  subset=subset.fillna(method="bfill")
  subset=subset.fillna(method="ffill")
  subset=subset.fillna(method="pad")
  return subset

def plotBitcoin(btc,ax1):
  ax1.plot(btc['Date'],btc['Close'])
  ax1.set_xlabel("Date")
  ax1.set_ylabel("Bitcoin price in USD")
  ax1.set_xlim([date.fromisoformat('2021-02-05'),btc['Date'][len(btc)-1]])
  ax1.set_ylim([0,max(btc['Close'])])
  #plt.show()    
  return ax1

def importer(path):
  dataset=pd.read_csv(path,delimiter=",")
  dataset=dataset.sort_values(by=['date'])
  dataset=dataset.reset_index()
  subset=dataset[0:20000]
  print(subset['date'])

  subset=subset[1:]
  subset['date']=pd.to_datetime(subset['date'])
  subset=subset.fillna(method="bfill")
  subset=subset.fillna(method="ffill")
  subset=subset.fillna(method="pad")
  subset = subset.reset_index()
  subset['nPOS']=np.nan
  subset['nNEG']=np.nan
  subset['nNEU']=np.nan
  return subset
  

def clean(text):

  # removing @ tags and links from the text
  text= ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text).split()) 
  # converting all letters to lower case and relacing '-' with spaces.
  text= text.lower().replace('-', ' ')
  # removing stowards and numbers
  table= str.maketrans('', '', string.punctuation+string.digits)
  text= text.translate(table)
  # tokenizing words 
  tokens = word_tokenize(text)
  # stemming the words 
  stemmed = [porter.stem(word) for word in tokens]
  words = [w for w in stemmed if not w in n_words]

  text = ' '.join(words)
  return text


def analyze(text):
  analysis=TextBlob(text)
  senti=analysis.sentiment.polarity

  if senti<0:
    emotion="NEG"
  elif senti>0:
    emotion="POS"
  else:
    emotion="NEU"

  return emotion


def resampledata(df,ax2):

  row=0
  for sentiment in df['sentiment']:
    
    if sentiment=="POS":
      df['nPOS'][row]=1
    
    if sentiment=="NEG":
      df['nNEG'][row]=1
    
    if sentiment=="NEU":
      df['nNEU'][row]=1
    
    row+=1
  
  print(df[['date','sentiment','nPOS','nNEG','nNEU']])

  core_data=df.drop(columns=['user_name','user_location','user_description','user_created','user_followers','user_friends',
            'user_favourites','user_verified','text','hashtags','source','is_retweet','cleaned','sentiment'])

  core_data=core_data.fillna(0)
  del core_data['index']
  core_data=core_data.resample("d",on='date').sum()
  core_data=core_data.reset_index()
  print(core_data)
  core_data['tweetvol']=core_data['nPOS']+core_data['nNEG']+core_data['nNEU']
  print(core_data)

  ax2.plot(core_data['date'],core_data['nPOS']/core_data['tweetvol'],label="%positive",color='g')
  ax2.plot(core_data['date'],core_data['nNEG']/core_data['tweetvol'],label="%negative",color='r')
  ax2.plot(core_data['date'],core_data['nNEU']/core_data['tweetvol'],label="%neutral",color='tab:orange')
  ax2.set_ylabel("Evolving percentage of tweets for each sentiment")
  ax2.set_title("Bitcoin price and tweet percentage per sentiment vs time")
  plt.legend()
  plt.tight_layout()
  #plt.savefig('relative%_sentiments.png')
  #plt.show()

  #return core_data,ax2,df
  ##ax2.plot(core_data['date'],core_data['nPOS'],label="positive",color='g')
  ##ax2.plot(core_data['date'],core_data['nNEG'],label='negative',color='r')
  ##ax2.plot(core_data['date'],core_data['nNEU'],label='neutral',color='b')
  #plt.xlabel("Datetime")
  ##ax2.set_ylabel("Number of observations")
  #plt.legend()
  #plt.show()

  return core_data,ax2,df


def make_wordcloud(df):

  text=""
  for x in df['cleaned']:
    text+=x
  
  wordcloud=WordCloud(collocations=False,background_color="white",
                          width=1200, height=1000,stopwords=STOPWORDS).generate(text)
  plt.figure()
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.axis("off")
  plt.title('General wordcloud')
  plt.savefig("General Wordcloud.png")
  plt.show() 


def sentiment_wordclouds(df):

  POStext=""
  NEGtext=""
  NEUtext=""

  j=0
  for x in df['sentiment']:

    if x=="POS":
      POStext+=df['cleaned'][j]
    
    if x=="NEG":
      NEGtext+=df['cleaned'][j]

    if x=="NEU":
      NEUtext+=df['cleaned'][j]

    j+=1
  
  print(POStext)
  print(NEGtext)
  print(NEUtext)

  textlist=[POStext,NEGtext,NEUtext]
  titlelist=["positive","negative","neutral"]
  stop_words=["crypto","bitcoin","btc","binanc","blockchain","airdrop","price","cryptocurr",
    "eth","ethereum","altcoin","project","thi","bsc","nft","shib","doge","bnb","nex","shibarmi",'usd']+list(STOPWORDS)

  for i in range(len(textlist)):

    wordcloud=WordCloud(collocations=False,background_color="white",
                          width=1200, height=1000,stopwords=stop_words).generate(textlist[i])
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud for "+titlelist[i])
    plt.savefig(titlelist[i]+" wordcloud.png")
    plt.show() 


def plotSentiments(df):

  positive=df['sentiment'].str.contains("POS").sum()
  print(positive)
  negative=df['sentiment'].str.contains("NEG").sum()
  print(negative)
  neutral=df['sentiment'].str.contains("NEU").sum()
  print(neutral)

  fig=plt.figure(figsize=(10,10))
  plt.bar(["positive","negative","neutral"],[positive,negative,neutral])
  plt.title("Total tweets per sentiment")
  plt.savefig("totalTweets.png")
  plt.show()



  
def main(path,path2):

    #start=time()
    fig, ax1=plt.subplots()
    plt.gcf().set_size_inches(14,8)
    ax2=ax1.twinx()
    df2=timestamp_importer(path2)
    btc_clean=df2.dropna()
    ax1=plotBitcoin(btc_clean,ax1)

    df=importer(path)
    df['cleaned']=df['text'].apply(lambda x: clean(x))
    df['sentiment']=df['cleaned'].apply(lambda x: analyze(x))
    #end=time()

    newdf,ax2,df=resampledata(df,ax2)
    fig.tight_layout()
    plt.savefig('sentiments.png')
    plt.show()
    print('RESAMPLE DONE')

    print("FIRSTCLOUD")
    make_wordcloud(df)
    
    print("NEXTCLOUDS")
    sentiment_wordclouds(df)

    plotSentiments(df)


main(pathe,path2)
