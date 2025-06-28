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


