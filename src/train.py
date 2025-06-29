from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor

lr = LinearRegression()
lasso = Lasso(alpha=5)
ridge = Ridge(alpha=3, random_state=42)
svr = LinearSVR(random_state=42, verbose=1)
knr = KNeighborsRegressor(n_neighbors=10, n_jobs=6, weights='distance')
rfr = RandomForestRegressor(random_state=42, n_estimators=300, n_jobs=6, oob_score=True)

def train_model(x_train, y_train, model=lr):
    model.fit(x_train, y_train)
    print(f"Model Score: {model.score(x_train, y_train)}")

    return model

def test_model(x_test, y_test, model=lr):
    y_preds = model.predict(x_test)
    
    from sklearn.metrics import r2_score
    
    return r2_score(y_test, y_preds)
