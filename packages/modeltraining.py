import joblib
import preprocessing as pp
import model as naves


data = pp.load_data()

X_train_vector, y_train, vector = pp.model_feat()

model = naves.model(X_train_vector, y_train)

joblib.dump(model, 'model.pkl')

joblib.dump(vector, open('vector.pkl','wb'))