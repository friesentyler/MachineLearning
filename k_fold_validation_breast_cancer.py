import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sys

# fit trains the model and then it is scored
def run(x_train, y_train, x_test, y_test, clf):
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test)

# split the training set using K-fold validation x = data, y = labels, k = chooses which piece of data is chosen for the fold, m = total number of folds
# k is modified later in the main() function using a for loop
def split(x,y,k,m):
    # calculate number of samples per fold based on the number of folds
    ns = int(y.shape[0] / m)
    # list which is populated with an entire fold under one index
    s = []
    # loop through the number of folds
    for i in range(m):
        # add to the list another list containing the data for the entire fold at index 0, and the labels for the entire fold at index 1
        s.append([x[(ns * i):(ns * i + ns)], y[(ns * i):(ns * i + ns)]])
    # pick a specific fold from list "s" using "k"
    x_test, y_test = s[k]
    # make lists for the training data
    x_train = []
    y_train = []
    # iterate through the total number of folds
    for i in range(m):
        # if the current fold equals the kth fold don't add it to the list, if it is, make a new list containing the current fold
        if (i == k):
            continue
        else:
            a, b = s[i]
            # append the current fold the returned data and labels
            x_train.append(a)
            y_train.append(b)
    # reformat the data to be equal to the number of samples in the dataset (minus the fold)
    x_train = np.array(x_train).reshape(((m - 1) * ns, 30))
    y_train = np.array(y_train).reshape((m - 1) * ns)
    return [x_train, y_train, x_test, y_test]

# print statistical data relating to the model
def pp(z,k,s):
    m = z.shape[1]
    print("%-19s: %0.4f +/- %0.4f | " % (s, z[k].mean(),
          z[k].std()/np.sqrt(m)), end='')
    for i in range(m):
        print("%0.4f " % z[k,i], end='')
    print()

# use the different machine learning models and test them using k-fold validation
def main():
    x = np.load("breast cancer dataset/bc_features_standard.npy")
    y = np.load("breast cancer dataset/bc_labels.npy")
    idx = np.argsort(np.random.random(y.shape[0]))
    x = x[idx]
    y = y[idx]
    folds = input("how many folds?")
    m = int(folds)
    z = np.zeros((8, m))
    for k in range(m):
        x_train, y_train, x_test, y_test = split(x, y, k, m)
        z[0, k] = run(x_train, y_train, x_test, y_test,
                      NearestCentroid())
        z[1, k] = run(x_train, y_train, x_test, y_test,
                      KNeighborsClassifier(n_neighbors=3))
        z[2, k] = run(x_train, y_train, x_test, y_test,
                      KNeighborsClassifier(n_neighbors=7))
        z[3, k] = run(x_train, y_train, x_test, y_test,
                      GaussianNB())
        z[4, k] = run(x_train, y_train, x_test, y_test,
                      DecisionTreeClassifier())
        z[5, k] = run(x_train, y_train, x_test, y_test,
                      RandomForestClassifier(n_estimators=5))
        z[6, k] = run(x_train, y_train, x_test, y_test,
                      RandomForestClassifier(n_estimators=50))
        z[7, k] = run(x_train, y_train, x_test, y_test,
                      SVC(kernel="linear", C=1.0))
    pp(z, 0, "Nearest")
    pp(z, 1, "3-NN")
    pp(z, 2, "7-NN")
    pp(z, 3, "Naive Bayes")
    pp(z, 4, "Decision Tree")
    pp(z, 5, "Random Forest (5)")
    pp(z, 6, "Random Forest (50)")
    pp(z, 7, "SVM (linear)")

main()