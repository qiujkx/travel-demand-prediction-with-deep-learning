from data_parsing import *
from sklearn.svm import SVR
import sklearn.preprocessing as pp

# Configuration
group_columns = ['LinkRef']
#categorial_columns = ['DayType', 'TimeOfDayClass']
categorical_columns = []
meta_columns = ['JourneyRef', 'DateTime', 'LineDirectionLinkOrder', 'LinkName']

results = pd.DataFrame()

# Load and pre-process data
data = load_csv('../data/4A_201701_Consistent.csv', group_columns = group_columns, categorical_columns = categorical_columns, meta_columns = meta_columns)

for group, X, Y, meta in data:

    print('Group:', group)

    # Split data into train and test    
    X_train, X_test = np.split(X, [int(.8*len(X))])
    Y_train, Y_test = np.split(Y, [int(.8*len(Y))])
    meta_train, meta_test = np.split(meta, [int(.8*len(meta))])

    clf = SVR()
    clf.fit(X_train, Y_train[:,0]) 

    Y_train_pred = clf.predict(X_train).reshape(-1, 1)

    metric_train_rmse = np.sqrt(np.mean((np.array(Y_train_pred) - np.array(Y_train))**2))
    print('Train RMSE:', metric_train_rmse)
    
    metric_train_mape = (np.abs(Y_train_pred - Y_train)/Y_train).mean()

    # Test

    Y_test_pred = clf.predict(X_test).reshape(-1, 1)

    metric_test_rmse = np.sqrt(np.mean((np.array(Y_test_pred) - np.array(Y_test))**2))
    print('Test RMSE:', metric_test_rmse)

    meta_test['LinkTravelTime_Predicted'] = Y_test_pred
    results = results.append(meta_test, ignore_index = True)
    
