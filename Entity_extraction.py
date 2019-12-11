#%% If needed, download the packages
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

#%% Import packages
import requests       #Allows you to obtain the source code of a webpage
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import warcio as warc                # Read WARC-files
from nltk import word_tokenize       # Tokenize
from nltk import TweetTokenizer      # Tokenize
from nltk import pos_tag             # Speech tagging
from nltk.corpus import stopwords    # Remove stopwords
from nltk.stem import PorterStemmer  # Stemming
from nltk.stem import WordNetLemmatizer # Lemmatization
from spellchecker import SpellChecker   # Spelling checker
from nltk.chunk import tree2conlltags
from pprint import pprint
from nltk import RegexpParser
from nltk.tag.stanford import StanfordNERTagger

#%% Read WARC-file
# Initialize list of DocumentIDs and HTML-codes
url        = []
recordID   = []
rawHTML    = []

# Define location of file
file  = ("/Users/reneehaegens/Desktop/Web Data Processing Systems/Assignment" + 
         "/files github/sample.warc")

with open(file, 'rb') as stream:
    for record in warc.ArchiveIterator(stream):
        # Other option is 'warcinfo'. Does not contain text
        if record.rec_type == 'response':
            if record.http_headers.get_header('Content-Type') == 'text/html':    
                url.append(record.rec_headers.get_header('WARC-Target-URI'))
                recordID.append(record.rec_headers.get_header('WARC-Record-ID'))
                parsed_html = BeautifulSoup(record.content_stream().read(), 'html.parser')
                for script in parsed_html(['script', 'style']):
                    script.decompose()
                rawHTML.append(parsed_html.get_text())
        

#%% Create dataframe
df = pd.DataFrame(data=list(zip(recordID, url, rawHTML))
                  ,columns=['RecordID', 'URL', 'HTMLtext']
                  )                  

#%% Extract raw text from HTML
def extract_raw_text_from_html(html_content):
    """
    Strips away all the html tags including the scripts and styles
    """
    lines    = (line.strip() for line in html_content.splitlines())
    chunks   = (phrase.strip() for line in lines for phrase in line.split("  "))
    raw_text = ' '.join(chunk for chunk in chunks if chunk)

    return raw_text

def extract_raw_text_from_url(url):
    """
    Downloads the html page and strips the page of html elements
    """
    response = requests.get(url)
    if response.status_code == 200:
        return extract_raw_text_from_html(response.text)
    else:
        print("Could not get webpage, status code: %d" % response.status_code)
        return ""
    
df['RawText'] = df['HTMLtext'].apply(extract_raw_text_from_html)
text          = df['RawText']

#%% Set starting time
t0 = time.time()

#%% Tokenize and POS tagging
"""
Split the raw text into separate tokens. Words like like "can't" are splitted 
into "ca" and "n't". 
Every token is lowercased, so two similar words having different cases are seen
as equal.
"""
# OPTION 1
df['TokensTagged'] = text.apply(word_tokenize).apply(pos_tag)
# OPTION 2
#df['TokensTagged'] = text.apply(TweetTokenizer().tokenize).apply(pos_tag)

## Additional tagging (TODO: GIVES ERRORS)
#pattern            = 'NP: {<DT>?<JJ>*<NN>}'
#cp                 = RegexpParser(pattern)
#cs                 = cp.parse(df['TokensTagged'])
#iob_tagged         = tree2conlltags(cs)

# Expand dataframe: a row for every token in the text per recordID
df_token           = pd.DataFrame(df['TokensTagged'].tolist(), index=df['RecordID']) \
                     .stack()                                                        \
                     .reset_index(name='TokensTagged')[['RecordID','TokensTagged']]
df_token['Token']  = [token[0].lower() for token in df_token['TokensTagged']]
df_token['Type']   = [token[1] for token in df_token['TokensTagged']]

## Filter dataframe on nouns only
df_token              = df_token[df_token['Type'].str.startswith("NN")]
df_token              = df_token.drop('TokensTagged', axis=1)
    
#%% Stop word removal
"""
Words with very low discrimination values (the most common words across the 
corpus) take much space, don't contain entities and are therefore deleted
"""
stopWords = stopwords.words("english")
df_stop   = df_token[df_token['Token'].isin([token for token in df_token['Token'] if token not in stopWords])]

#%% Spell correction (TODO: ADJUST TO DF)
"""
Misspelled words don’t match with the actual term. Therefore, all words that 
are unknown are corrected.
"""
#misspelled = SpellChecker().unknown(tokens)
#for word in misspelled:
#    print(word, " becomes", SpellChecker().correction(word))
#    tokens[tokens.index(word)] = SpellChecker().correction(word)

#%% Stemming/lemmatization
"""
Use lemmatization to make each term unique by meaning and save space
Some terms which have the same meaning are classified as distinct terms and 
hurt the similarity. We use lemmatization to make each term unique by 
meaning and save space. As, unlike stemming, lemmatization depends on 
correctly identifying the intended meaning of the word in a sentence.
"""
df_stop['Lemma'] = [WordNetLemmatizer().lemmatize(token) for token in df_stop['Token']]

#### CHECK FOR THE DIFFERENCE BETWEEN STEMMING AND LEMMATIZATION
#a = pd.DataFrame()
#a['Token'] = df_stop['Token']
#a['Stem'] = [PorterStemmer().stem(token) for token in df_stop['Token']]
#a['Lemma'] = [WordNetLemmatizer().lemmatize(token) for token in df_stop['Token']]
#a['Equal_YN'] = (a.Stem == a.Lemma)

#%% Group similar words per recordID
"""
A recordID might include words multiple times. Therefore, the dataframe is 
grouped on recordID and lemma. The number of times a word appears in the text 
is saved in the column "N"
"""
df_def = df_stop.groupby(['RecordID', 'Lemma'], as_index=False).count()[['RecordID','Lemma','Token']]
df_def = df_def.rename({"Token": "N"}, axis='columns')


#%% Stanford NER
"""
Assign the type to a named entity
3 class: Location, Person, Organization
4 class:	 Location, Person, Organization, Misc
7 class:	 Location, Person, Organization, Money, Percent, Date, Time
"""

path = "/Users/reneehaegens/Desktop/Web Data Processing Systems/Assignment/stanford-ner-2018-10-16/"
classifier = path + "classifiers/english.all.3class.distsim.crf.ser.gz"
#classifier = path + "classifiers/english.conll.4class.distsim.crf.ser.gz"
#classifier = path + "/classifiers/english.muc.7class.distsim.crf.ser.gz"
jar        = path + "stanford-ner.jar"
st         = StanfordNERTagger(classifier, jar)
ner_tokens = st.tag(df_def['Lemma'])
df_def['NER']= [token[1] for token in ner_tokens]

## Obtain all the tagged_tokens
#ner_tagged_tokens = []
#for token in ner_tokens:
#    if token[1] != 'O':
#        ner_tagged_tokens.append(token)


#%% Obtain total run time of entity extraction
print(time.time() - t0)
