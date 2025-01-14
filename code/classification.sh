#!/bin/bash

# create directory if not yet existing
mkdir -p data/classification/

# run feature extraction on training set (may need to fit extractors)
echo "  training set"
#python -m code.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle --svm 'rbf' -s 42 -a -k -f1 -ba
python -m code.classification.run_classifier data/feature_extraction/training.pickle -e data/classification/classifier.pickle --bayes -s 42 -a -k -f1 -ba


# run feature extraction on validation set (with pre-fit extractors)
echo "  validation set"
#python -m code.classification.run_classifier data/dimensionality_reduction/validation.pickle -i data/classification/classifier.pickle -a -k -f1 -ba
python -m code.classification.run_classifier data/feature_extraction/validation.pickle -i data/classification/classifier.pickle -a -k -f1 -ba


# don't touch the test set, yet, because that would ruin the final generalization experiment!
echo "  test set"
python -m code.classification.run_classifier data/feature_extraction/test.pickle -i data/classification/classifier.pickle -a -k -f1 -ba