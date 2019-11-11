from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense,  GlobalMaxPool1D,Dropout
from keras.layers.embeddings import Embedding

class model():
    def __init__(self):
        self.params = {}
        
    def load_params(self,model_path = '../model/'):
        self.params = {}
        self.vocabulary_size = 20000
        self.embedding_dim = 100
        self.input_len = 350
        self.num_dense_1 = 512
        self.num_classes = 5
        return self.params
        
    def define_model(self):
        
        params = self.load_params(self)
        
        model = Sequential()
        model.add(Embedding(input_dim=self.vocabulary_size, 
                            output_dim=self.embedding_dim, 
                            input_length=self.input_len))
        model.add(GlobalMaxPool1D())
        model.add(Dense(self.num_dense_1, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train_model(self,X_train, X_test, y_train, y_test):
        history = self.model.fit(X_train, y_train, 
                  validation_data=(X_test,y_test), 
                  epochs=3, batch_size=64)
        
        return history

    
    def save_model(self, model_path = '../model/'):
        self.model.save(model_path + 'model_bk.h5')
        
    def save_params(self,model_path = '../model/'):
        pass
        
    def load_model(self, model_path = '../model/'):
        model = load_model(model_path + 'model.h5')
        return model
    
    