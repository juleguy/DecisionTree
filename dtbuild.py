#!/usr/bin/env python
# coding: utf-8
from foilprop import FoilProp, NotFoundTargetCol
from missingvaluesstrategy import UnknownMissingValuesStrategyException
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument("-f", "--file", help="path to the arff data file")

parser.add_argument("-m", "--mode", help="mode of the program : 'training_only' (default) or 'prediction'",
                    default="training_only")

parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

parser.add_argument("-missing", "--missing-values",
                    help="strategy to adopt in case of missing values : 'ignore' (default) or 'randomly_replace'",
                    default='ignore')

parser.add_argument("--target-col", help="name of the column that contains the target class name")

parser.add_argument("--target-class", help="name of the target class")


args = parser.parse_args()

if args.file is None:
    print("No file provided (argument -f or --file)")
    exit(1)

# Creating the computing object
foilProp = FoilProp(args.verbose)

# Setting the mode as prediction if the user asked
foilProp.prediction_mode = (args.mode == "prediction")

try:
    foilProp.fit(args.file, target_col_value=args.target_class,
                 target_col_name=args.target_col, missing_values_strategy_str=args.missing_values)
except FileNotFoundError:
    print("The file cannot be found")
    exit(1)
except NotFoundTargetCol:
    print("The target col name or the target col value is incorrect")
    exit(1)
except UnknownMissingValuesStrategyException:
    print("The strategy to manage the missing values does not exist ('ignore' or 'randomly_replace' are available)")
    exit(1)

if args.verbose:
    print()
    print("Train set size : " + str(foilProp.train_set_size))


if foilProp.prediction_mode:
    foilProp.predict()

    if args.verbose:
        print("Test set size : " + str(foilProp.test_set_size))

    print()
    print("True positives : "+str(foilProp.true_positives))
    print("True negatives : "+str(foilProp.true_negatives))
    print("False positives : "+str(foilProp.false_positives))
    print("False negatives : "+str(foilProp.false_negatives))

    print()
    print("Precision : " + str(foilProp.precision))
    print("Recall : " + str(foilProp.recall))
    print("f1_score : " + str(foilProp.f1_score))
