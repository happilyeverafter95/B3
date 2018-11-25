import pandas as pd
import numpy as np

from keras.models import Model, load_model
from keras.layers import Dense, Embedding, Input, Activation, LSTM, Bidirectional, Dropout, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation


# Bot class takes a dataset with col "Message sent" -> "Message Received" 
# TODO: support dataset with AB dialogue 

# <----------------------- Global Config -------------------------------> 

epochs = 10
embedding_dim = 300 
max_len = 100 # max message length
batch_size = 32
glove_embeddings_index = None

# <----------------------- Helper Function -----------------------------> 
def preprocess(text):
    text = text.split()
    text = [t.lower() for t in text if t.isalpha()]
    return " ".join(text)
    
def onehot(arr, num_class):
    return np.eye(num_class)[np.array(arr.astype(int)).reshape(-1)]
    
def load_embeddings(embedding_dir="C://Users//mandy//Desktop//B3 chatbot//B3//glove.6b.50d.txt"):
    embeddings_index = {}
    f  = open(embedding_dir, encoding = "utf8")
        
    for line in f:
        values = line.split()
        word = ''.join(values[:-embedding_dim])
        coefs = np.asarray(values[-embedding_dim:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    global glove_embeddings_index
    glove_embeddings_index = embeddings_index

# <----------------------- Prediction Class -----------------------------> 

class bot(): 
        
    def __init__(self):
        self.data = None
        self.tokenizer = None
        self.response_class = None
        self.num_class = None
        self.model = None

    def load_corpus(self, data):
        # check corpus criteria
        self.data = data
                
    def preprocess_data(self):
        self.data['preproc_sent'] = self.data['sent'].apply(preprocess)
        self.data['preproc_received'] = self.data['received'].apply(preprocess) 
    
    def train_tokenizer(self):
        tokenizer = Tokenizer(num_words = None)
        tokenizer.fit_on_texts(self.data['preproc_received'])
        self.tokenizer = tokenizer
              
    def cluster_responses(self):
        responses = self.data['received']
        vectorizer = TfidfVectorizer()
        response_vectors = vectorizer.fit_transform(responses)
        AP = AffinityPropagation()
        clustering = AP.fit(response_vectors)
        response_classes = list(clustering.predict(response_vectors))
        self.data['response_class'] = pd.Series(response_classes)
        self.num_class = len(set(response_classes))
        
    def train_model(self, directory = "C://Users//mandy//Desktop//B3 chatbot//B3//"):
        
        tokenizer = self.tokenizer
        X = self.data['preproc_received']
        sequences = tokenizer.texts_to_sequences(X)
        X = pad_sequences(sequences, maxlen=max_len)
        
        y = self.data['response_class']
        y = onehot(y, self.num_class)
        
        word_index = tokenizer.word_index
                
        # create embedding layer 
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        
        for word, i in word_index.items():
            embedding_vector = glove_embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        
        embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights = [embedding_matrix], input_length = max_len, trainable = False) 
        input= Input(shape=(max_len, ), dtype = 'int32')
        embedded_sequences = embedding_layer(input) 
        x = Bidirectional(LSTM(50, return_sequences=True))(embedded_sequences)
        x = Bidirectional(LSTM(10, return_sequences=True))(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(50, activation = 'relu')(x)
        x = Dropout(0.1)(x)
        output = Dense(self.num_class, activation='softmax')(x)
        model = Model(inputs=input, outputs=output)
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        
        checkpoint = ModelCheckpoint(directory + "model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early = EarlyStopping(monitor='val_loss', mode='min', patience=3)
        callback = [checkpoint, early]
        
        model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callback)
        
        self.model = model 
        
    def get_class(self, message): 
        message = preprocess(message)
        message = self.tokenizer.texts_to_sequences([message])
        message = pad_sequences(message, maxlen = max_len)
        
        pred_class = self.model.predict(message)
        pred_class = np.argmax(pred_class) 
        
        return pred_class
        
    def predict(self, message): 
        pred_class = self.get_class(message)
        
        data = self.data
        response_class = data[data['response_class'] == pred_class]['received']
        
        # randomly chosen message from response class 
        random = response_class.sample(1)
        
        return random
        
        
# <----------------------- Init test example ------------------> 

test = bot() 
data = pd.DataFrame(columns = ['sent', 'received'])
data['sent'] = pd.Series(['hi', 'hello', 'helllllooooo', 'bye', 'good bye'])
data['received'] = pd.Series(['hi', 'herro', 'hiya', 'bybye', 'bye'])
test.load_corpus(data)
test.preprocess_data()
test.train_tokenizer()
test.cluster_responses()
test.train_model()
test.predict("this is test")