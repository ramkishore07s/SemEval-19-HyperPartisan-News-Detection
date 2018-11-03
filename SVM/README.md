The model uses Linear SVM for classification.

Features include tf-idf weights of unigram and bigram tokens. Also other features such as polarity and subjectivity and average sentence lenght are used.

The Ground-truth classifier is a binary Linear SVM classifier and the Bias Classifier (5 class) is a OneVsRest Linear SVM classifier.

The Accuracy obtained for truth classifier is 70% and for the bias classifier is around 30%.
