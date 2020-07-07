import math

# This class read from the file the attributes input
class FileFromReader:

    """read from the file and return all attributes and values from the file.
    :param fileName is a string of the input file name
    """
    @staticmethod
    def get_attributes(fileName):
        attributes = []  # array
        with open(fileName, "r") as myFile:
            plant_att = myFile.readline().strip().split("\t")
            for line in myFile:
                line = line.strip()
                data = line.split("\t")
                plants = {}  # dictionary
                for column, value in enumerate(data):
                    plants[plant_att[column]] = value
                attributes.append(plants)
        return attributes


"""write the accuracy results to file"""
def write_results_to_file(DT_classifications, KNN_classifications, NB_classifications, test_examples, tree):
    # Number of samples
    lists_len = len(DT_classifications)
    DT_correct = 0
    KNN_correct = 0
    NB_correct = 0
    with open("output.txt", "w") as file:
        file.write(str(tree))
        file.write("\n")
        file.write("\n")
        for i in range(0, lists_len):
            true_classification = test_examples.get__attributes()[i][test_examples.get__classifier()]
            # Count number of correct prediction
            if DT_classifications[i] == true_classification:
                DT_correct += 1
            if KNN_classifications[i] == true_classification:
                KNN_correct += 1
            if NB_classifications[i] == true_classification:
                NB_correct += 1
        result_line = str((math.ceil((DT_correct / float(lists_len))*100.0) / 100.0)) + ' ' + str((math.ceil((KNN_correct / float(lists_len))*100.0) / 100.0)) + ' ' + str(math.ceil((NB_correct / float(lists_len))*100.0) / 100.0)
        file.write(result_line)


# This class is representing nodes in a tree.
class Node:
    """Constructor.
    :param label String - Label to represent node
    """
    def __init__(self, label):
        self.__label = label
        self.__child = []

    """Getter.
    return String label
    """
    def get__label(self):
        return self.__label

    """Getter.
    return List <Node> Children of node
    """
    def get__children(self):
        return self.__child

    """ Print the tree in the following format:
    ‫‪<attribute_name>=<attribute_value>
    ‫‪<tab>|<attribute_name>=<attribute_value>:class‬‬
    """
    def __str__(self, level=0):
        string = ""
        for index, child in enumerate(self.__child):
            if index > 0:
                string += "\n"
            if level != 0:
                string += level * "\t"
                string += "|"
            string += self.__label + "=" + child.get__label()
            for grandchild in child.get__children():
                if grandchild.get__children():
                    string += "\n"
                    string += str(grandchild.__str__(level+1))
                else:
                    string += ":" + grandchild.get__label()
        return string


class Examples:
    """
    Attributes is a dictionary from attribute to List<String> values.
    :param attributes List<Dict<String>> List of plants each one have dictionary of attributes and values
    :param attribute_value_types Option to keep value types from last iteration
    """
    def __init__(self, attributes, attribute_value_types = None):
        self.__attributes = attributes
        # Get list of classifiers
        self.__classification_types = list()
        try:
            self.__attribute_types = list(attributes[0].keys())
            self.__classifier = self.__attribute_types[-1]
        except IndexError:
            self.__attribute_types = []
            self.__classifier = None
        if attribute_value_types == None:
            self.__attribute_value_types = dict()
        else:
            self.__attribute_value_types = attribute_value_types
        for plants in self.__attributes:
            if plants[self.__classifier] not in self.__classification_types:
                self.__classification_types.append(plants[self.__classifier])
            # Add value types for each attribute
            for attribute_name in self.__attribute_types:
                try:
                    self.__attribute_value_types[attribute_name].add(plants[attribute_name])
                except KeyError:
                    self.__attribute_value_types[attribute_name] = set()

    """Getter.
    :return List<Dic<String,String>> List of plants and their attributes
    """
    def get__attributes(self):
        return self.__attributes

    """Getter.
    :return String classifier name
    """
    def get__classifier(self):
        return self.__classifier

    """Getter.
    :return List<String> List of classification names (yes,no) OR (true,false)
    """
    def get__classification_types(self):
        return self.__classification_types

    """Getter.
    :return List<String> List of attribute names in example
    """
    def get__attribute_types(self):
        return self.__attribute_types

    """Getter.
    :return Dic<AttributeName>: List<Values>
    """
    def get__attribute_value_types(self):
        return self.__attribute_types

    """Get examples of specific attribute.
    :return (List<String names>,List<Examples>) which matches attribute
    """
    def get_specific_examples(self, attribute_name):
        examples_data = {}  # Dictionary of examples
        examples = []
        values = []
        result = []
        for plant in self.__attributes:
            examples_data.setdefault(plant[attribute_name],[]).append(plant)

        # Add missing attribute values:
        for key, temp_values in self.__attribute_value_types.items():
            if key == attribute_name:
                for value in temp_values:
                    if value not in examples_data:
                        examples_data[value] = []


        for index, example_data in enumerate(examples_data.items()):
            # Add elements according to alphabetical order
            if index == 0:
                values.append(example_data[0])
                examples.append(Examples(example_data[1], self.__attribute_value_types))
            else:
                # Insert according to number / word order
                try:
                    i = 0
                    while i < len(values):
                        if (float(values[i]) >= float(example_data[0])):
                            break
                        else:
                            i+=1
                except ValueError:
                    i = 0
                    while i < len(values):
                        if (values[i] >= example_data[0]):
                            break
                        else:
                            i += 1
                values.insert(i, example_data[0])
                examples.insert(i, Examples(example_data[1], self.__attribute_value_types))
        result.append(values)
        result.append(examples)
        return result

    """
    :return List of 2 Dic<String> dictionary of value_name: number of yes classifier for positive and negetive examples
    """
    def get_classifier_values(self, att_name):
        result_p = dict()
        result_n = dict()
        classifier = self.__classifier
        for plant in self.get__attributes():
            att_value = plant[att_name]
            if (plant[classifier] == "yes" or plant[classifier] == "true"):
                result_p[att_value] = result_p.get(att_value, 0) + 1
                result_n[att_value] = result_n.get(att_value, 0)
            else:
                result_n[att_value] = result_n.get(att_value, 0) + 1
                result_p[att_value] = result_p.get(att_value, 0)
        return result_p, result_n


#ID3 Decision tree algorithm class.
class DTL:

    """
    :return Best attribute to divide decision tree node
    """
    @staticmethod
    def choose_attribute(attributes, examples):
        max_gain = float("-inf")
        max_att = None
        # att_name = Attribute name
        # att values = Attribute values
        for att_name in examples.get__attribute_types():
            att_values = []
            for plant in examples.get__attributes():
                att_values.append(plant[att_name])
            if att_name in attributes:
                sum = 0
                sum_p = 0
                sum_n = 0
                # dicts[0] is p value, dicts[1] is n value
                dicts = examples.get_classifier_values(att_name) # Dictionary of the form att_name: number of apperances
                # Calculate values for each Sv
                for value_name, p in dicts[0].items():
                    sum_p += p
                    n = dicts[1][value_name]
                    sum_n += n
                    total = n + p
                    try:
                        log_p = math.log(p/total, 2)
                    except ValueError:
                        log_p = 0   # Value is best for tree
                    try:
                        log_n = math.log(n/total, 2)
                    except ValueError:
                        log_n = 0  # Value is best for tree

                    entropy_sv = - (p/total)*log_p - (n/total)*log_n
                    sum += (p + n)*entropy_sv # Multiply by Sv

                # Gain(S,A) = Entropy(S) -sum((|Sv|/|S|) * Entropy(Sv))
                total_sum = sum_p + sum_n
                sum /= total_sum
                entropy_s = (-((sum_p / total_sum) * math.log((sum_p / total_sum), 2))) - ((sum_n / total_sum) * math.log((sum_n / total_sum), 2))
                current_gain = entropy_s - sum
                if max_gain < current_gain:
                    max_gain = current_gain
                    max_att = att_name
        return max_att
    """ Chooses classification using majority vote.
    :return Node(yes/no label)
    """
    @staticmethod
    def Mode(examples):
        yes_counter = 0
        no_counter = 0
        classifier_name = examples.get__classifier()
        for plant in examples.get__attributes():
            classifier = plant[classifier_name]
            if (classifier == "yes" or classifier == "true"):
                yes_classification = classifier
                yes_counter += 1
            else:
                no_classification = classifier
                no_counter += 1
        # return by majority vote
        if yes_counter >= no_counter:
            return Node(yes_classification)
        else:
            return Node(no_classification)

    """ID3 Algorithm.
    :param examples List<List<Attribute>> list of examples, classification in the last column
    """
    @staticmethod
    def run_dtl(examples, attributes, default):
        # Returns default if list is empty
        if not examples.get__attributes():
            return default
        # All the examples have the same classification return it
        else:
            classification_types = examples.get__classification_types()
            if len(classification_types) == 1:
                return Node(classification_types[0])
            # If attributes is empty return most common classification
            elif not attributes:
                return DTL.Mode(examples)
            else:
                best = DTL.choose_attribute(attributes, examples)
                tree = Node(best)
                new_examples = examples.get_specific_examples(best)
                for label, new_example in zip(new_examples[0], new_examples[1]):
                    new_attributes = attributes.copy()
                    new_attributes.remove(best)
                    subtree = DTL.run_dtl(examples=new_example, attributes=new_attributes, default=DTL.Mode(examples))
                    labelNode = Node(label)
                    labelNode.get__children().append(subtree)
                    tree.get__children().append(labelNode)
                return tree
    """
    :param tree Node: Node tree to analyse example by
    :param Examples: List of test examples
    """
    @staticmethod
    def analyse_dtl(tree, examples):
        classifications = []
        for example in examples.get__attributes():
            classifications.append(DTL.analyse_example(tree, example, examples.get__classification_types()))
        return classifications

    """ Checks value for example by tree
    :return String yes / no
    :param tree Node: Node tree to analyse example by
    :param example 
    """
    @staticmethod
    def analyse_example(tree, example, classifiers):
        if tree.get__children():
            for child in tree.get__children():
                if child.get__label() == example[tree.get__label()]:
                    return DTL.analyse_example(child.get__children()[0], example, classifiers)
            if "no" in classifiers:
                return "no"
            else:
                return "false"
        else:
            return tree.get__label()

# KNN algorithm class
class KNN:

    """ Run KNN algorithm on samples.
    :param
    """
    @staticmethod
    def run_KNN(attribute_names, train_examples, test_examples, k=5):
        classifications = []

        for test_plant in test_examples.get__attributes():
            yes_counter = 0
            no_counter = 0
            hamming = KNN.get_hamming(attribute_names, train_examples, test_plant)
            # Sort elements by hamming distance
            sorted = list(zip(hamming[0], hamming[1], hamming[2]))
            sorted.sort()
            hamming = list(zip(*sorted))
            for i in range(0, k):
                if (hamming[2][i] == "yes" or hamming[2][i] == "true"):
                    yes_classification = hamming[2][i]
                    yes_counter += 1
                else:
                    no_classification = hamming[2][i]
                    no_counter += 1
            if yes_counter > no_counter:
                classification = yes_classification
            else:
                classification = no_classification
            classifications.append(classification)
        return classifications

    """Get hamming distance between train and test people.
    :return List<Integer> distances from different train data
    """
    @staticmethod
    def get_hamming(attribute_names, train_examples, test_plant):
        ham_distance = []
        indexes = []
        classifiers = []
        for index, train_plant in enumerate(train_examples.get__attributes()):
            distance = 0
            for key, value in train_plant.items():
                if key in attribute_names:
                    if test_plant[key] != train_plant[key]:
                        distance += 1
            ham_distance.append(distance)
            indexes.append(index)
            classifiers.append(train_plant[train_examples.get__classifier()])
        return (ham_distance, indexes, classifiers)

# Class to represent Naive Base algorithm
class NaiveBase:

    """ Run naive base algorithm.
    :param attribute_names List<String> attribute names
    :param train_examples Examples of train
    :param test_examples Examples of test
    """
    @staticmethod
    def run_naive_base(attribute_names, train_examples, test_examples):
        classifications = []
        results = NaiveBase.calculate_naive_base(attribute_names, train_examples)
        classifiers = train_examples.get__classification_types()
        total_pos_prop = results[0]
        total_neg_prop = results[1]
        conditional_props = results[2]
        for plant in test_examples.get__attributes():
            yes_prop = 1
            no_prop = 1
            for attribute, value in plant.items():
                if attribute in attribute_names:
                    yes_prop *= conditional_props.setdefault("yes", "true")[attribute].setdefault(value, 0)
                    no_prop *= conditional_props.setdefault("no", "false")[attribute].setdefault(value, 0)
                else:
                    continue
            yes_prop *= total_pos_prop
            no_prop *= total_neg_prop

            # Return max attribute relative to the data
            if "yes" in classifiers:
                if yes_prop > no_prop:
                    classifications.append("yes")
                else:
                    classifications.append("no")
            else:
                if yes_prop > no_prop:
                    classifications.append("true")
                else:
                    classifications.append("false")

        return classifications


    """ Return necessary calculations for naive base. 
    param attribute_names List<String> Attribute names
    param train_examples Examples Train examples
    return YesProb, NoProb, dicResultsProbs for attribute
    """
    @staticmethod
    def calculate_naive_base(attribute_names, train_examples):
        results = {}
        classifier = train_examples.get__classifier()
        total_dic = train_examples.get_classifier_values(classifier)
        total_n = total_dic[1].setdefault("no","false")
        total_p = total_dic[0].setdefault("yes","true")
        total = total_n + total_p
        for plant in train_examples.get__attributes():
            for attribute in attribute_names:
                value_name = plant[attribute]
                inside_dic = results.setdefault(plant[classifier], {}).setdefault(attribute, {})
                try:
                    inside_dic[value_name] += 1
                except KeyError:
                    # Initialize dictionary
                    inside_dic[value_name] = 0
        for classifier_value, attributes in results.items():
            for attribute, attribute_values in attributes.items():
                sum_attribute = 0
                for attribute_value in attribute_values.values():
                    sum_attribute += attribute_value
                for name, attribute_value in attribute_values.items():
                    attribute_values[name] = (attribute_value + 1) / (sum_attribute + len(attribute_values))
        return (total_p / total), (total_n / total), results



# This is the main program
def main():
    # Read attributes for train and test files
    train_attributes = FileFromReader.get_attributes("train.txt")
    train_examples = Examples(train_attributes)
    test_attributes = FileFromReader.get_attributes("test.txt")
    test_examples = Examples(test_attributes)
    attribute_names = list(train_examples.get__attributes()[0].keys())
    # Remove classifier attribute
    attribute_names.pop()
    tree = DTL.run_dtl(train_examples, attribute_names, DTL.Mode(train_examples))
    # The three algorithms
    DT_classifications = DTL.analyse_dtl(tree, test_examples)
    KNN_classifications = KNN.run_KNN(attribute_names, train_examples, test_examples)
    NB_classifications = NaiveBase.run_naive_base(attribute_names, train_examples, test_examples)
    write_results_to_file(DT_classifications, KNN_classifications, NB_classifications, test_examples, tree)

# Run this function
if __name__ == "__main__":
    main()
