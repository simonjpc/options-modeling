class DummyModel:
    def __init__(self, preds):
        self.preds = preds
    
    def predict(self, X):
        return self.preds