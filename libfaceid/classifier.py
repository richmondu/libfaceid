from enum import Enum
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier





class FaceClassifierModels(Enum):

    NAIVE_BAYES         = 0
    LINEAR_SVM          = 1
    RBF_SVM             = 2
    NEAREST_NEIGHBORS   = 3
    DECISION_TREE       = 4
    RANDOM_FOREST       = 5
    NEURAL_NET          = 6
    ADABOOST            = 7
    QDA                 = 8
    DEFAULT = NAIVE_BAYES


class FaceClassifier():

    def __init__(self, classifier=FaceClassifierModels.DEFAULT):
        self._clf = None
        if classifier == FaceClassifierModels.NAIVE_BAYES:
            self._clf = GaussianNB()
        elif classifier == FaceClassifierModels.LINEAR_SVM:
            self._clf = SVC(C=1.0, kernel="linear", probability=True)
        elif classifier == FaceClassifierModels.RBF_SVM:
            self._clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
        elif classifier == FaceClassifierModels.NEAREST_NEIGHBORS:
            self._clf = KNeighborsClassifier(1)
        elif classifier == FaceClassifierModels.DECISION_TREE:
            self._clf = DecisionTreeClassifier(max_depth=5)
        elif classifier == FaceClassifierModels.RANDOM_FOREST:
            self._clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        elif classifier == FaceClassifierModels.NEURAL_NET:
            self._clf = MLPClassifier(alpha=1)
        elif classifier == FaceClassifierModels.ADABOOST:
            self._clf = AdaBoostClassifier()
        elif classifier == FaceClassifierModels.QDA:
            self._clf = QuadraticDiscriminantAnalysis()
        print("classifier={}".format(classifier))

    def fit(self, embeddings, labels):
        self._clf.fit(embeddings, labels)

    def predict(self, vec):
        return self._clf.predict_proba(vec)


