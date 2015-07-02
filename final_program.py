import numpy as np
import urllib
import matplotlib.pyplot as plt
import sys
import pylab as pl
from sklearn import svm
from sklearn import svm, cross_validation
from sklearn.grid_search import GridSearchCV

# Load our matched data from the catalog
cat_data = 'catalog_data_final/matched_data.csv'
catalog = np.genfromtxt(cat_data, dtype=None, names=True, delimiter=',')

# Pull out certain features so we can work with them separately
data = catalog
classes, mag, v_mag, blend, amplitude, period, epoch_folding = data['Var_Type'], data['Mag'], data["V_mag"], data['Blend'], data['Amplitude'], data['Period_days'], data['epoch_folding']

# Grab our featuers and transpore to use in SVM
features = v_mag, amplitude, period
data_svm = np.array(features).transpose()

# How many of each Variable Type do we have
for x in range(1,4):
    print("Class size: {} {}".format(x,len(classes[classes == x])))

    # Figure out how many we need to train for accuracy

# Test size
N_test = 5000

clf = svm.LinearSVC()

# X_train, X_test, y_train, y_test = cross_validation.train_test_split (data_svm, classes, test_size=1./3.)
# print("training set = {} {}".format(  X_train.shape, y_train.shape ))
# print("test size = {} {}".format(X_test.shape, y_test.shape))

# clf.fit(X_train, y_train)
# pred_class = clf.predict(X_test)
# N_match = (pred_class == y_test).sum()
# print("N_match = {}".format(N_match))
# acc = 1. * N_match / len(pred_class)
# print("Accuracy = {}".format(acc))

# ss = cross_validation.StratifiedShuffleSplit(classes, 5, test_size = 1./3.)
# scores = cross_validation.cross_val_score(clf, data_svm, classes, cv=ss)
# print("Accuracy = {} +- {}",format(scores.mean(),scores.std()))

# Training Sizes
Ns = 2**np.arange(2,12)
print("Ns = {}".format(Ns))

scores = np.zeros(len(Ns))
stds = np.zeros(len(Ns))

for i in range(len(Ns)):
    N = Ns[i]
    ss = cross_validation.StratifiedShuffleSplit(classes, 5, test_size = N_test, train_size = N)
    scores_i = cross_validation.cross_val_score(clf, data_svm, classes, cv=ss)
    scores[i] = scores_i.mean()
    stds[i] = scores_i.std()


pl.clf()
fig = pl.figure()
ax = fig.add_subplot(1,1,1)
ax.errorbar (Ns, scores, yerr = stds)
ax.set_xscale("log")
ax.set_xlabel("N")
ax.set_ylabel("Accuracy")
pl.savefig('optimal_training_size.png')

scores.argmax()
optimal_n = 64
print("Optimal N = {}".format(optimal_n))


# Optimal C size

N_train = optimal_n
N_test = optimal_n

C_range = 10. ** np.arange(-5, 5)
param_grid = dict(C=C_range)
ss = cross_validation.StratifiedShuffleSplit(classes, 5, test_size = N_test, train_size = N_train)
grid = GridSearchCV(svm.LinearSVC(), param_grid=param_grid, cv=ss)
grid.fit (data_svm, classes)
print("The best classifier is: ".format(grid.best_estimator_))

# plot the scores of the grid grid_scores_ contains parameter settings and scores
# grid_scores_ contains parameter settings and scores
score_dict = grid.grid_scores_
# We extract just the scores
scores = [x[1] for x in score_dict]

pl.clf()
fig = pl.figure()
ax = fig.add_subplot(1,1,1)
ax.plot (C_range, scores)
ax.set_xscale("log")
ax.set_xlabel("C")
ax.set_ylabel("Accuracy")
pl.savefig('optimal_c_size.png')