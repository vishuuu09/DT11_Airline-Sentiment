import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")  # Fixed typo
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer

class TextToNum:
    def __init__(self, text):
        self.text = text  # Ensure self.text is updated

    def cleaner(self):
        text = re.sub(r',', '', self.text)
        cleaned_text = re.sub(r'[^\w\s]', '', text)  # Removes special characters
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra spaces
        self.cleaned = cleaned_text.strip()  # Removes leading/trailing whitespace
        return self.cleaned  # Return cleaned text for debugging

    def token(self):
        self.tkns = word_tokenize(self.cleaned)
        return self.tkns  # Return tokens

    def removeStop(self):
        stop = set(stopwords.words('english'))  # Use set for faster lookup
        self.cl = [i for i in self.tkns if i.lower() not in stop]  # Convert to lowercase before checking
        return self.cl  # Return cleaned list

    def stemme(self):
        ps = PorterStemmer()
        self.st = [ps.stem(word) for word in self.cl]
        return self.st  # Return stemmed words
