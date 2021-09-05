from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Step1: Create data set
X, y = make_moons(n_samples=10000, noise=.5, random_state=0)

# Step2: Split the training test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Fit a Decision Tree model as comparison
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

# output 0.7555

# Step 4: Fit a Random Forest model, " compared to "Decision Tree model, accuracy go up by 5-6%
clf = RandomForestClassifier(n_estimators=100,max_features="auto",random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
# output 0.7965
""" n_estimators is how many tree you want to grow. In other word, how many subset you want to split and train. 
max_features is the number of random features that individual decision tree will be used for finding the optimal splitting.
If “auto”, then max_features=sqrt(n_features).
If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
If “log2”, then max_features=log2(n_features).
If None, then max_features=n_features. """
