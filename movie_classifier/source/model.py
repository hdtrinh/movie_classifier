from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense,  GlobalMaxPool1D,Dropout, LSTM,Bidirectional
from keras.layers.embeddings import Embedding
import pickle

class model_classifier():
    def __init__(self, model_path = '../model/'):
        self.model_path = model_path
        self.vocabulary_size = 0
        self.embedding_dim = 0
        self.input_len = 0
        self.num_dense_1 = 0
        self.num_classes = 0
        self.n_epochs = 0
        self.batch_dim = 0
        self.params = {}
        
        
    def load_params(self, params_dict):
        self.params = params_dict
      
        self.vocabulary_size = params_dict['VOCABULARY_SIZE']
        self.embedding_dim = params_dict['EMBEDDING_DIM']
        self.input_len = params_dict['INPUT_LEN']
        self.num_dense_1 = params_dict['NUM_DENSE_1']
        self.num_classes = params_dict['NUM_CLASSES']
        self.n_epochs = params_dict['NUM_EPOCHS']
        self.batch_dim = params_dict['BATCH_DIM']
        return self.params
        
    def define_model(self, params_dict):
        
        params = self.load_params(params_dict)
        '''
        model = Sequential()
        model.add(Embedding(input_dim=self.vocabulary_size, 
                            output_dim=self.embedding_dim, 
                            input_length=self.input_len))
        model.add(GlobalMaxPool1D())
        model.add(Dense(self.num_dense_1, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        '''             
        model = Sequential()
        model.add(Embedding(input_dim=self.vocabulary_size, 
                               output_dim=self.embedding_dim, 
                               input_length=self.input_len))
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(GlobalMaxPool1D())
        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train_model(self,X_train, X_test, y_train, y_test):
        history = self.model.fit(X_train, y_train, 
                  validation_data=(X_test,y_test), 
                  epochs=self.n_epochs, batch_size=self.batch_dim)
        
        return history
    
    def save_model(self, model_path = '../model/'):
        self.model.save(model_path + 'model.h5')
        
    def save_params(self,model_path = '../model/'):
        with open(self.model_path + 'model_params.pkl', 'wb') as outfile:
            pickle.dump(self.params, outfile, protocol=2)
        
    def load_model(self, model_path = '../model/'):
        model = load_model(model_path + 'model.h5')
        return model
    
    