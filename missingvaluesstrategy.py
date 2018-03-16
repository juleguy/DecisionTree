from abc import ABC, abstractmethod
from random import Random


class AbstractMissingValuesStrategy(ABC):
    """ Abstract class of the missing values strategies """

    @abstractmethod
    def transform(self, row, attributes_dictionary, df, target_col_name, target_col_value_index,
                  duplicates_allowed=False):
        pass


class IgnoreMissingValuesStrategy(AbstractMissingValuesStrategy):
    """ Returns none if a value is missing """

    def transform(self, row, attributes_dictionary, df, target_col_name, target_col_value_index,
                  duplicates_allowed=False):

        for key, value in row.items():
            if row[key] not in attributes_dictionary[key]:
                return None

        return row


class RandomlyReplaceMissingValuesStrategy(AbstractMissingValuesStrategy):
    """ Randomly replaces one of the missing values by a value in the dictionary of possible values
        If the generated new row has already been generated or is already contained in the
        DataFrame, returns None. This test makes this solution quite slow """

    def __init__(self):
        super().__init__()
        self._random = Random()
        self._random.seed(12)
        self._generated_series = []

    def transform(self, row, attributes_dictionary, df, target_col_name, target_col_value_index,
                  duplicates_allowed=False):

        generated_row = False

        for key, value in row.items():
            if row[key] not in attributes_dictionary[key]:

                generated_row = True

                chosen_index = self._random.randint(0, len(attributes_dictionary[key]) - 1)
                chosen_val = attributes_dictionary[key][chosen_index]
                row[key] = chosen_val
        unique = True

        if generated_row:
            # Checking if the DataFrame already contains the generated row
            # Compulsory check because if two identical rows are in the positive and negative examples,
            # the algorithm never ends
            for index_row, rowDF in df.iterrows():
                if rowDF.equals(row):
                    unique = False
                    break

        for generated in self._generated_series:
            if row.equals(generated):
                unique = False
                break

        if not unique and not duplicates_allowed:
            return None
        else:
            self._generated_series.append(row)
            return row


class ReplaceByMostCommonValueInClassStrategy(AbstractMissingValuesStrategy):
    """ Replaces each missing value by the most common in the class """

    def __init__(self):
        self._is_most_common_computed = False
        self._most_commons_pos = {}
        self._most_commons_neg = {}
        self._generated_series = []

    def compute_most_commons(self, df, target_col_name, target_col_value_index, attributes_dictionary):
        """ Computes the most common value for each attribute """

        # Initialization of the keys
        for col in df.columns:
            if col != target_col_name:
                self._most_commons_pos[col] = []
                self._most_commons_neg[col] = []

        # Iterating over all the columns
        for col in df.columns:

            # Ignoring the column of labels
            if col != target_col_name:

                # Initialization of the counters
                counter_pos = []
                counter_neg = []
                for i in range(len(attributes_dictionary[col])):
                    counter_pos.append(0)
                    counter_neg.append(0)

                # Iterating over all the rows
                for index_row, row in df.iterrows():

                    # Only caring about non missing values
                    if row[col] in attributes_dictionary[col]:

                        # Computing the index of the current value
                        current_index = attributes_dictionary[col].index(row[col])

                        # Updating the right counter
                        if row[target_col_name] == attributes_dictionary[target_col_name][target_col_value_index]:
                            counter_pos[current_index] = counter_pos[current_index] + 1
                        else:
                            counter_neg[current_index] = counter_neg[current_index] + 1

                # Computing the index of the max occurrences value for each class
                max_pos = -1
                max_neg = -1
                max_index_pos = -1
                max_index_neg = -1

                for i in range(len(attributes_dictionary[col])):
                    if counter_pos[i] > max_pos:
                        max_index_pos = i
                        max_pos = counter_pos[i]

                    if counter_neg[i] > max_neg:
                        max_index_neg = i
                        max_neg = counter_neg[i]

                self._most_commons_pos[col] = attributes_dictionary[col][max_index_pos]
                self._most_commons_neg[col] = attributes_dictionary[col][max_index_neg]

        self._is_most_common_computed = True

    def transform(self, row, attributes_dictionary, df, target_col_name, target_col_value_index,
                  duplicates_allowed=False):

        # Computing the most common value for each attribute of each class if not already done
        if not self._is_most_common_computed:
            self.compute_most_commons(df, target_col_name, target_col_value_index, attributes_dictionary)

        generated_row = False

        # Replacing missing values by the most commons for the current attribute and the current class
        for key, value in row.items():
            if row[key] not in attributes_dictionary[key]:

                generated_row = True

                if row[target_col_name] == attributes_dictionary[target_col_name][target_col_value_index]:
                    row[key] = self._most_commons_pos[key]
                else:
                    row[key] = self._most_commons_neg[key]

        unique = True

        if generated_row:

            # Checking if the DataFrame already contains the generated row
            # Compulsory check because if two identical rows are in the positive and negative examples,
            # the algorithm never ends
            for index_row, row_df in df.iterrows():
                if row_df.equals(row):
                    unique = False
                    break

        for generated in self._generated_series:
            if row.equals(generated):
                unique = False
                break

        if not unique and not duplicates_allowed:
            return None
        else:
            self._generated_series.append(row)
            return row


class UnknownMissingValuesStrategyException(Exception):
    """ Exception raised if a MissingValuesStrategy cannot be built
     because the string that describes it is not known """


class MissingValuesStrategyBuilder:
    """ Builds a MissingValuesStrategy of the subclass indicated by the input string """

    @staticmethod
    def build(description):

        if description == "ignore":
            return IgnoreMissingValuesStrategy()
        elif description == "randomly_replace":
            return RandomlyReplaceMissingValuesStrategy()
        elif description == "most_common_replace":
            return ReplaceByMostCommonValueInClassStrategy()
        else:
            raise UnknownMissingValuesStrategyException
