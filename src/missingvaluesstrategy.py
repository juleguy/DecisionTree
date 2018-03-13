
from abc import ABC, abstractmethod
from random import Random


class AbstractMissingValuesStrategy(ABC):
    """ Abstract class of the missing values strategies """

    @abstractmethod
    def transform(self, row, attributes_dictionary):
        pass


class IgnoreMissingValuesStrategy(AbstractMissingValuesStrategy):
    """ Returns none if a value is missing """

    def transform(self, row, attributes_dictionary):

        for key, value in row.items():
            if row[key] not in attributes_dictionary[key]:
                return None

        return row


class RandomlyReplaceMissingValuesStrategy(AbstractMissingValuesStrategy):
    """ Randomly replaces one of the missing values by a value in the dictionary of possible values """

    def __init__(self):
        super().__init__()
        self._random = Random()
        self._random.seed(42)

    def transform(self, row, attributes_dictionary):

        for key, value in row.items():
            if row[key] not in attributes_dictionary[key]:

                chosen_index = self._random.randint(0, len(attributes_dictionary[key])-1)
                chosen_val = attributes_dictionary[key][chosen_index]
                row[key] = chosen_val

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
        else:
            raise UnknownMissingValuesStrategyException
