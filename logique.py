

class Literal:
    """ Class whose role is to store a literal and to print it"""

    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value

    def __eq__(self, other):
        return self.value == other.value and self.attribute == other.attribute

    def __str__(self):
        return self.attribute + " = " + self.value


class Rule:
    """ Class whose role is to store a rule and to print it"""

    def __init__(self, premises=[], conclusion=None):
        self.premises = premises
        self.conclusion = conclusion

    def add_premise(self, premise):
        self.premises.append(premise)

    def set_conclusion(self, conclusion):
        self.conclusion = conclusion

    def __str__(self):
        out = "IF "
        for i in range(len(self.premises)):
            out += str(self.premises[i]) + " "

            if 0 < i < len(self.premises) - 1:
                out += "AND "

        out += "THEN  "
        out += str(self.conclusion)

        return out
