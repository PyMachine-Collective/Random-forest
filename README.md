# Random-forest
Random forest is the advanced version of decision tree. It combines/ensembles numbers of decision trees for prediction, instead of just one. 

# How does random forest work ?

Random forest is an ensemble model using bagging as the ensemble architecture and decision tree as individual model as weak learner.

Here are the four steps of building a random forest model:

<img src="https://raw.githubusercontent.com/lilly-chen/Bite-sized-Machine-Learning/f19b826cf8bbd4164fbb433039eb50ffebb9de59/Random%20Forest/Capture1.PNG"/>

Step 1: select n (e.g. 1000) random subsets from the training set

Step 2: train n decision tree
one random subset is used for training one decision tree
    the optimal splits for each decision trees is based on a random subset of features (e.g. 10 feature in total, use a 5 random feature to split)

Step 3: each tree predicts the records/candidates in the test set, independently.

Step 4: select the majority vote from these 1000 tree on each record/candidate in the test set as the final decision

# How to do it in python
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
    accuracy_score(y_test, y_pred)
    # output 0.7555
    
    # Step 4: Fit a Random Forest model, " compared to "Decision Tree model, accuracy go up by 5-6%
    clf = RandomForestClassifier(n_estimators=100,max_features="auto",random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy_score(y_test, y_pred)
    # output 0.7965
    """ n_estimators is how many tree you want to grow. In other word, how many subset you want to split and train. 
    max_features is the number of random features that individual decision tree will be used for finding the optimal splitting.
    If “auto”, then max_features=sqrt(n_features).
    If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
    If “log2”, then max_features=log2(n_features).
    If None, then max_features=n_features. """






