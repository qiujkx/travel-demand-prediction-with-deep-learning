from data_parsing import *
from sklearn import linear_model
import sklearn.preprocessing as pp

# Configuration
group_columns = []
categorial_columns = ['LinkRef', 'DayType', 'TimeOfDayClass']
meta_columns = ['JourneyRef', 'DateTime', 'LineDirectionLinkOrder', 'LinkName']

results = pd.DataFrame()

# Load and pre-process data
data = load_csv('../data/4A_201701_Consistent.csv', group_columns = group_columns, categorial_columns = categorial_columns, meta_columns = meta_columns)

for group, X, Y, meta in data:
    
    # Split data into train and test    
    X_train, X_test = np.split(X, [int(.8*len(X))])
    Y_train, Y_test = np.split(Y, [int(.8*len(Y))])
    meta_train, meta_test = np.split(meta, [int(.8*len(meta))])
    print('Train data set (size, features):',  X_train.shape)

    clf = linear_model.LinearRegression()
    clf.fit(X_train, Y_train[:,0]) 

    Y_train_pred = clf.predict(X_train).reshape(-1, 1)
    
    # Test
    print('Group:', group, '\n\tTest data set (size, features):',  X_test.shape)

    Y_test_pred = clf.predict(X_test).reshape(-1, 1)

    meta_test['LinkTravelTime_Predicted'] = Y_test_pred
    results = results.append(meta_test, ignore_index = True)
    
    # Write predictions to CSV
    results.to_csv('../data/results_lr_single.csv', index = False, encoding = 'utf-8')
    