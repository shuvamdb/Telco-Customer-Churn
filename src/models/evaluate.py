from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    
    '''
    model : trained model
    X_test : Test features
    y_test : Test labels
    
    '''
    pred = model.predict(X_test)
    print("Classification Report : \n", classification_report(y_test, pred))
    print("Confusion matrix : \n", confusion_matrix(y_test, pred))