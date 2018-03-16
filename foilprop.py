from math import log

import pandas
from arff2pandas import a2p
from logic import Literal, Rule
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from random import Random

from missingvaluesstrategy import MissingValuesStrategyBuilder


class AbstractFoilProp(ABC):

    def __init__(self):

        # Attributes storing the input data
        self._df_train_pos = None
        self._df_train_neg = None
        self._df_test_pos = None
        self._df_test_neg = None
        self._attribute_dictionary = {}
        self._target_col_name = None
        self._target_col_value_index = None
        self._col_names = []

        # Attributes storing the learning results
        self._rules = []  # Accessed by obj.rules

        # Attribute storing if the prediction mode is on
        self.prediction_mode = False  # Accessed and modified by obj.prediction_rules

        # Attributes storing the prediction results
        self._true_positives = 0  # Accessed by obj.true_positives
        self._true_negatives = 0  # Accessed by obj.true_negatives
        self._false_positives = 0  # Accessed by obj.false_positives
        self._false_negatives = 0  # Accessed by obj.false_negatives

        # Other available properties :
        # train_set_size
        # test_set_size
        # precision
        # recall
        # f1_score

    """ Public methods """

    def fields_reinitialization(self):
        self._df_train_pos = None
        self._df_train_neg = None
        self._df_test_pos = None
        self._df_test_neg = None
        self._attribute_dictionary = {}
        self._target_col_name = None
        self._target_col_value_index = None
        self._col_names = []
        self._rules = []  # Accessed by obj.rules
        self._true_positives = 0  # Accessed by obj.true_positives
        self._true_negatives = 0  # Accessed by obj.true_negatives
        self._false_positives = 0  # Accessed by obj.false_positives
        self._false_negatives = 0  # Accessed by obj.false_negatives

    @abstractmethod
    def fit(self, filename, target_col_name=None, target_col_value=None):
        pass

    @abstractmethod
    def predict(self):
        pass

    """ Private methods """

    @abstractmethod
    def _load_fields_from_file(self, filename, target_col_name, target_col_value, missing_values_strategy_str):
        pass

    @abstractmethod
    def _gain(self, L, df_pos, df_neg):
        pass

    @abstractmethod
    def _best_gain_literal(self, literals_list, df_pos, df_neg):
        pass

    """ Getters for private fields """

    @property
    def train_set_size(self):
        return len(self._df_train_neg.index) + len(self._df_train_pos.index)

    @property
    def test_set_size(self):
        if self.prediction_mode:
            return len(self._df_test_neg.index) + len(self._df_test_pos.index)
        else:
            raise UnavailableInTrainingModeException

    @property
    def rules(self):
        return self._rules

    @property
    def true_positives(self):
        if self.prediction_mode:
            return self._true_positives
        else:
            raise UnavailableInTrainingModeException

    @property
    def true_negatives(self):
        if self.prediction_mode:
            return self._true_negatives
        else:
            raise UnavailableInTrainingModeException

    @property
    def false_positives(self):
        if self.prediction_mode:
            return self._false_positives
        else:
            raise UnavailableInTrainingModeException

    @property
    def false_negatives(self):
        if self.prediction_mode:
            return self._false_negatives
        else:
            raise UnavailableInTrainingModeException

    @property
    def precision(self):
        if self.prediction_mode:

            if self.true_positives + self.false_positives == 0:
                return "not computable"
            else:
                return self.true_positives / (self.true_positives + self.false_positives)
        else:
            raise UnavailableInTrainingModeException

    @property
    def recall(self):
        if self.prediction_mode:

            if self.true_positives + self.false_negatives == 0:
                return "not computable"
            else:
                return self.true_positives / (self.true_positives + self.false_negatives)
        else:
            raise UnavailableInTrainingModeException

    @property
    def f1_score(self):
        if self.prediction_mode:

            if self.precision == 0 or self.recall == 0 \
                    or self.precision == "not computable" or self.recall == "not computable":

                return "not computable"
            else:
                return 2 / ((1 / self.precision) + (1 / self.recall))
        else:
            raise UnavailableInTrainingModeException


class UnavailableInTrainingModeException(Exception):
    """ Thrown when the client tries to access fields that are only available in prediction mode """


class NotFoundTargetCol(Exception):
    """ Thrown when the client provides a target col name that doesn't exist """


class FoilProp(AbstractFoilProp):
    """ Implementation of the FoilProp class """

    def __init__(self, verbose):
        super().__init__()
        self._verbose = verbose

    def fit(self, filename, target_col_name=None, target_col_value=None, missing_values_strategy_str="ignore"):

        # Initialization of all the fields
        self.fields_reinitialization()

        self._load_fields_from_file(filename, target_col_name, target_col_value, missing_values_strategy_str)

        # Copying the data of self.dfPos so that it is no lost during the processing
        df_pos = pandas.DataFrame(self._df_train_pos.copy())
        df_neg = self._df_train_neg

        # No rule for now
        self._rules = []

        # Extracting all the literals from the attributes dictionary
        literals = []
        for attribute, values_list in self._attributes_dictionary.items():
            if attribute != self._target_col_name:
                for value in values_list:
                    literals.append(Literal(attribute, value))

        initial_size = len(df_pos)

        print()

        # Main loop : while all the positive examples haven't been processed
        while not df_pos.empty:

            if self._verbose:
                print("Computed : "+str(((initial_size-(len(df_pos)))/initial_size)*100)+" %")

            # Creating a empty rule
            current_rule = Rule()

            # Copying the sets of examples
            df_pos2 = pandas.DataFrame(df_pos.copy())
            df_neg2 = pandas.DataFrame(df_neg.copy())

            # Internal loop : while all the negative example haven't been put aside
            while not df_neg2.empty:

                # Looking for the literal with the best gain
                best_literal = self._best_gain_literal(literals, df_pos2, df_neg2)

                # Adding the best literal as a premise of the currently built rule
                current_rule.add_premise(best_literal)

                # Removing incompatible examples with the new premise from df_pos2
                df_pos2 = df_pos2[df_pos2[best_literal.attribute] == best_literal.value]

                # Removing incompatible examples with the new premise from df_neg2
                df_neg2 = df_neg2[df_neg2[best_literal.attribute] == best_literal.value]

            # Adding the new rule to the collection of rules
            conclusion = Literal(self._target_col_name,
                                 self._attributes_dictionary.get(self._target_col_name)[self._target_col_value_index])

            current_rule.set_conclusion(conclusion)
            self._rules.append(current_rule)

            # Removing from df_pos the examples satisfying the rule
            for index_row, row in df_pos.iterrows():
                if current_rule.fits_row(row):
                    df_pos.drop(index_row, inplace=True)

        print()
        print("Extracted rules : ")

        for rule in self._rules:
            print(str(rule))

    def predict(self):

        # Raising exception if the prediction mode is disabled
        if not self.prediction_mode:
            raise UnavailableInTrainingModeException

        # Predicting on the positive examples of the test set
        for index_row, row in self._df_test_pos.iterrows():

            compatible = False

            # Looking for at least one rule fitting the row
            for rule in self._rules:
                if rule.fits_row(row):
                    compatible = True
                    break

            if compatible:
                self._true_positives = self.true_positives + 1
            else:
                self._false_negatives = self._false_negatives + 1

        # Predicting on the negative examples of the test set
        for index_row, row in self._df_test_neg.iterrows():

            compatible = False

            # Looking for at least one rule fitting the row
            for rule in self._rules:
                if rule.fits_row(row):
                    compatible = True
                    break

            if compatible:
                self._false_positives = self.false_positives + 1
            else:
                self._true_negatives = self._true_negatives + 1

    def _best_gain_literal(self, literals_list, df_pos, df_neg):
        """ Returns the literal maximizing the gain among the list of given literals for the
            given positive and negative example sets """

        # If the list contains at least one element, computing the gain of the first literal
        if len(literals_list) >= 1:
            max_gain = self._gain(literals_list[0], df_pos, df_neg)
            best_literal = literals_list[0]

            # Looking for a better gain among all the others literals
            for i in range(1, len(literals_list)):
                computed_gain = self._gain(literals_list[i], df_pos, df_neg)
                if computed_gain is not None and (max_gain is None or computed_gain > max_gain):
                    max_gain = computed_gain
                    best_literal = literals_list[i]

            # If all the gains were None (not computable), returning a random literal
            if max_gain is None:
                best_literal = Random().choose(literals_list)
                print("max gain is none, returning "+str(best_literal))

            # Managing the case where all the gains = 0
            # This means that the neg and pos contains line that are identical except for one or more attributes that
            # are exactly symmetrical
            #
            # ex :
            #
            # pos = a b c d
            #       e f g h
            # neg = a b c h
            #       e f g d
            #
            # To determine the literal that will unlock the situation, we first compute the two rows that are the most
            # alike (that contain the most identical literals) and then the chosen literal is one of those that differ
            # on the most alike lines
            elif max_gain == 0:

                max_similarity = 0
                alike_row_pos = None
                alike_row_neg = None

                # Looking for the couple of rows with the best similarity
                for index_pos, row_pos in df_pos.iterrows():
                    for index_neg, row_neg in df_neg.iterrows():

                        current_similarity = 0

                        for col in df_pos.columns:
                            if row_pos[col] == row_neg[col]:
                                current_similarity = current_similarity + 1

                        if current_similarity >= max_similarity:
                            alike_row_pos = row_pos
                            alike_row_neg = row_neg
                            max_similarity = current_similarity

                # Looking for the first literal that differ on the most alike rows
                for col in df_pos.columns:
                    if alike_row_pos[col] != alike_row_neg[col]:
                        best_literal = Literal(col, alike_row_pos[col])
                        break

        else:
            print("Returning none because no literal found")
            best_literal = None  # Returning none if no literal has been found

        # Returning the literal with the best gain
        return best_literal

    def _gain(self, L, df_pos, df_neg):
        """ Computes the gain of the literal L on the positive and negative examples """

        # Computing P the number of positive examples and N the number of negative examples
        P = len(df_pos)
        N = len(df_neg)

        # Computing p the number of positive examples satisfying L
        p = self._compute_effective_occurrence(L, df_pos)

        # Computing n the number of negative examples satisfying L
        n = self._compute_effective_occurrence(L, df_neg)

        # Computing the gain and returning it, if no division by zero is necessary
        if not (p == 0) and not (P == 0) and not (p + n == 0) and not (P + N == 0):
            gain = p * (log(p / (p + n), 2) - log(P / (P + N), 2))
        else:
            gain = None

        return gain

    def _compute_effective_occurrence(self, L, df):
        """ Computes the value p or n of the gain computing """

        # Computing k the number of occurences of a literal in a DataFrame
        k = 0
        for index, row in df.iterrows():
            if row[L.attribute] == L.value:
                k = k + 1

        return k

    def _load_fields_from_file(self, filename, target_col_name, target_col_value, missing_values_strategy_str):
        """ Initializes all the fields according to the info read in the given file"""

        self._attributes_dictionary = {}
        self._col_names = []

        with open(filename) as f:
            df = a2p.load(f)

        # Extracting the columns names and the possible values for each column
        for col in df.columns:
            pos_at = col.find('@')
            col_name = col[:pos_at]

            values = col[pos_at + 1:]
            values = values.replace("{", "")
            values = values.replace("}", "")

            values = values.split(",")

            df[col_name] = df[col]
            df = df.drop(col, axis=1)
            self._attributes_dictionary[col_name] = values
            self._col_names.append(col_name)

        if target_col_name is None:
            # Supposing the target column is the last one
            target_index = len(self._col_names) - 1
            self._target_col_name = self._col_names[target_index]
        else:
            self._target_col_name = target_col_name
            found_target_col = False
            for i, col_name in enumerate(self._col_names):
                if col_name == target_col_name:
                    target_index = i
                    found_target_col = True
                    break
            if not found_target_col:
                raise NotFoundTargetCol

        # Creating the pos and neg dataframes
        self._create_dataframes(df, target_index, target_col_value, missing_values_strategy_str)

        if self._verbose:

            print()
            print("Positive examples of the train set :")
            print(self._df_train_pos)

            print()
            print("Negative examples of the train set :")
            print(self._df_train_neg)

            if self.prediction_mode:

                print()
                print("Positive examples of the test set :")
                print(self._df_test_pos)

                print()
                print("Negative examples of the test set :")
                print(self._df_test_neg)

            print()
            print("Attributes dictionary :")
            print(self._attributes_dictionary)

    def _create_dataframes(self, df, target_index, target_col_value, missing_values_strategy_str):
        """ Creates df_pos and df_neg dataframes.
            In case of prediction mode, also creates df_pos_test and df_neg_test dataframes """

        # Instantiation of  the missing values strategy from the given string
        missing_values_strategy = MissingValuesStrategyBuilder.build(missing_values_strategy_str)

        # Initializing the positive and negative arrays
        self._df_train_pos = pandas.DataFrame()
        self._df_train_neg = pandas.DataFrame()
        self._df_test_pos = pandas.DataFrame()
        self._df_test_neg = pandas.DataFrame()

        # Creating the columns in the dataframes
        for col_name in self._col_names:

            if col_name != self._target_col_name:
                self._df_train_pos[col_name] = None
                self._df_train_neg[col_name] = None
                self._df_test_pos[col_name] = None
                self._df_test_neg[col_name] = None

        if self.prediction_mode:
            # Using the sklearn library to split the data into two sets
            # The random state is fixed so that the resulting sets are always the same for a given input
            df_train_set, df_test_set = train_test_split(df, test_size=0.2, random_state=42)
        else:
            df_train_set = df

        if target_col_value is None:
            self._target_col_value_index = 0
        else:
            found_target_index = False
            for i, val in enumerate(self._attributes_dictionary[self._target_col_name]):
                if val == target_col_value:
                    self._target_col_value_index = i
                    found_target_index = True
                    break

            if not found_target_index:
                raise NotFoundTargetCol

        # Creating the Neg and Pos dataframes
        for index_row, row in df_train_set.iterrows():

            # Managing the missing values according to the current strategy
            new_row = missing_values_strategy.transform(row, self._attributes_dictionary,
                                                        df_train_set, self._target_col_name,
                                                        self._target_col_value_index)
            if new_row is not None:

                # Removing target attribute
                new_row = new_row.drop(self._target_col_name)

                # Adding the row to the right dataframe
                if row[target_index] == self._attributes_dictionary[self._target_col_name][self._target_col_value_index]:
                    self._df_train_pos = self._df_train_pos.append(new_row, ignore_index=True)
                else:
                    self._df_train_neg = self._df_train_neg.append(new_row, ignore_index=True)

        # If prediction mode, creating the test pos and neg dataframes
        if self.prediction_mode:

            for index_row, row in df_test_set.iterrows():

                # Managing the missing values according to the current strategy
                new_row = missing_values_strategy.transform(row, self._attributes_dictionary,
                                                            df_test_set, self._target_col_name,
                                                            self._target_col_value_index,
                                                            duplicates_allowed=True)
                if new_row is not None:

                    # Removing target attribute
                    new_row = new_row.drop(self._target_col_name)

                    # Adding the row to the right dataframe
                    if row[target_index] == \
                            self._attributes_dictionary[self._target_col_name][self._target_col_value_index]:
                        self._df_test_pos = self._df_test_pos.append(new_row, ignore_index=True)
                    else:
                        self._df_test_neg = self._df_test_neg.append(new_row, ignore_index=True)
