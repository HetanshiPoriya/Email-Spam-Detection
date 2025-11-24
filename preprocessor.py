# preprocessor.py

import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# Initialize lemmatizer (lm) and stopwords outside the function for efficiency
lm = WordNetLemmatizer() 
# You might need to ensure 'stopwords' and 'wordnet' are downloaded in the environment
# nltk.download('stopwords') 
# nltk.download('wordnet')

def transform_text(text):
    # Your full text transformation logic goes here
    # 1. Convert to lower case
    text = text.lower()
    
    # 2. Tokenization
    text = nltk.word_tokenize(text)
    
    # 3. Removing special characters and filtering out non-alphanumeric tokens
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    
    # 4. Removing stopwords and punctuations
    for i in text:
        # Assuming you defined stopwords and string.punctuation somewhere
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()

    # 5. Lemmatizing
    for word in text:
         y.append(lm.lemmatize(word))
         
    # Return as a single string
    return " ".join(y)