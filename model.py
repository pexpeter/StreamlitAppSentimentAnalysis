from sklearn.naive_bayes import MultinomialNB

def model(X_train_vector, y_train):
    naves = MultinomialNB()
    naves.fit(X_train_vector, y_train)
    
    return naves
