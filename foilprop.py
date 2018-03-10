from math import log

import pandas
from arff2pandas import a2p
from logique import Literal, Rule
from abc import ABC, abstractmethod


class AbstractFoilProp(ABC):

    def __init__(self):

        # Attributes storing the input data
        self._df_pos = pandas.DataFrame()
        self._df_neg = pandas.DataFrame()
        self._attribute_dictionary = {}
        self._target_col_name = None
        self._col_names = []
        self._training_set_size = None  # Accessed by obj.training_set_size

        # Attributes storing the learning results
        self._rules = []  # Accessed by obj.rules

        # Attribute storing if the prediction mode is on
        self.prediction_mode = False  # Accessed and modified by obj.prediction_rules

        # Attributes storing the prediction results
        self._test_set_size = None  # Accessed by obj.test_set_size
        self._true_positives = None  # Accessed by obj.true_positives
        self._true_negatives = None  # Accessed by obj.true_negatives
        self._false_positives = None  # Accessed by obj.false_positives
        self._false_negatives = None  # Accessed by obj.false_negatives
        self._precision = None  # Accessed by obj.precision
        self._recall = None  # Accessed by obj.recall
        self._f1_score = None  # Accessed by obj.f1_score

    """ Public methods """

    @abstractmethod
    def fit(self, filename):
        pass

    @abstractmethod
    def predict(self):
        pass

    """ Private methods """

    @abstractmethod
    def _load_fields_from_file(self, filename):
        pass

    @abstractmethod
    def _gain(self, L, df_pos, df_neg):
        pass

    @abstractmethod
    def _best_gain_literal(self, literals_list, df_pos, df_neg):
        pass

    """ Getters for private fields """

    @property
    def training_set_size(self):
        return self._training_set_size

    @property
    def test_set_size(self):
        if self.prediction_mode:
            return self._test_set_size
        else:
            raise UnavailableInTrainingModeException

    @property
    def rules(self):
        return self._rules

    @property
    def prediction_mode(self):
        return self.prediction_mode

    @prediction_mode.setter
    def prediction_mode(self, val):
        self.prediction_mode = val

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
            return self._precision
        else:
            raise UnavailableInTrainingModeException

    @property
    def recall(self):
        if self.prediction_mode:
            return self._recall
        else:
            raise UnavailableInTrainingModeException

    @property
    def f1_score(self):
        if self.prediction_mode:
            return self._f1_score
        else:
            raise UnavailableInTrainingModeException


class UnavailableInTrainingModeException(Exception):
    """ Thrown when the client tries to access fields that are only available in prediction mode """


class FoilProp(AbstractFoilProp):

    def predict(self):
        pass

    def __init__(self):
        super().__init__()

    def fit(self, filename):

        self._load_fields_from_file(filename)

        # Copying the data of self.dfPos so that it is no lost during the processing
        df_pos = pandas.DataFrame(self.dfPos.copy())
        df_neg = self.dfNeg

        # No rule for now
        rules = []

        # Extracting all the literals from the attributes dictionnary
        literals = []
        for attribute, values_list in self._attributes_dictionary.items():
            if attribute != self.target_col_name:
                for value in values_list:
                    literals.append(Literal(attribute, value))

        # Main loop : while all the positive examples haven't been processed
        while not df_pos.empty:

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
            conclusion = Literal(self.target_col_name, self._attributes_dictionary.get(self.target_col_name)[0])
            current_rule.set_conclusion(conclusion)
            rules.append(current_rule)

            # Removing from df_pos the examples satisfying the rule
            df_pos = df_pos[df_pos[best_literal.attribute] != best_literal.value]

            print(str(current_rule))

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
        else:
            best_literal = None  # Returning none if no literal has been found

        # Returning the literal with the best gain
        return best_literal

    def _gain(self, L, df_pos, df_neg):
        """ Computes the gain of the litteral L on the positive and negative examples """

        # Computing P the number of positive examples and N the number of negative examples
        P = len(df_pos)
        N = len(df_neg)

        # Computing p the number of positive examples satisfying L
        p = 0
        for index, row in df_pos.iterrows():
            if row[L.attribute] == L.value:
                p = p + 1

        # Computing n the number of negative examples satisfying L
        n = 0
        for index, row in df_neg.iterrows():
            if row[L.attribute] == L.value:
                n = n + 1

        print("p=" + str(p) + " n=" + str(n) + " P=" + str(P) + " N=" + str(N))

        # Computing the gain and returning it, if no division by zero is necessary
        if not (p == 0) and not (P == 0) and not (p + n == 0) and not (P + N == 0):
            gain = p * (log(p / (p + n), 2) - log(P / (P + N), 2))
        else:
            gain = None

        return gain

    def _load_fields_from_file(self, filename):
        self._attributes_dictionary = {}
        self._col_names = []

        try:
            with open(filename) as f:
                df = a2p.load(f)
        except:
            print("The file cannot be opened")
            return

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

        # Supposing the target column is the last one
        target_index = len(self._col_names) - 1
        self.target_col_name = self._col_names[target_index]

        # Initializing the positive and negative arrays
        self.dfPos = pandas.DataFrame()
        self.dfNeg = pandas.DataFrame()

        # Creating the columns in dfPos and dfNeg
        for col_name in self._col_names:

            if col_name != self.target_col_name:
                self.dfPos[col_name] = None
                self.dfNeg[col_name] = None

        # Creating the Neg and Pos dataframes
        for index_row, row in df.iterrows():

            new_row = {}

            for indexCol, col_name in enumerate(df.columns):

                if indexCol != target_index:
                    new_row[col_name] = row[col_name]

            # Adding the row to the right dataframe
            if row[target_index] == self._attributes_dictionary[self.target_col_name][0]:
                self.dfPos = self.dfPos.append(new_row, ignore_index=True)
            else:
                self.dfNeg = self.dfNeg.append(new_row, ignore_index=True)

        print(self.dfPos)
        print(self.dfNeg)
        print(self._col_names)
        print(self._attributes_dictionary)
