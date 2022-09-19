import numpy as np

'''
To get the entropy
train_data: dataset by pandas
class_list: a list of classes
label: a string of the current class(label)
'''


def get_total_entropy(train_data, class_list, label):
    total_row = train_data.shape[0]
    entropy = 0
    for c in class_list:
        class_num = train_data[train_data[label] == c].shape[0]
        class_entropy = - (class_num / total_row) / np.log2(class_num / total_row)
        entropy += class_entropy
    return entropy


'''
get entropy of specified feature
feature_value_data: dataset with specified feature
class_list: a list of classes
label: a string of the current label
'''


def get_entropy(feature_value_data, class_list, label):
    class_num = feature_value_data.shape[0]
    entropy = 0
    for c in class_list:
        label_class_num = feature_value_data[feature_value_data[label] == c].shape[0]
        class_entropy = 0
        if label_class_num != 0:
            class_probability = label_class_num / class_num
            class_entropy = - class_probability * np.log2(class_probability)
            entropy += class_entropy
    return entropy


'''
get the information gain
train_data: dataset by pandas
feature_name: a string of the current feature's name
class_list: a list of classes
label: a string of the current class(label)
'''


def get_IG(train_data, feature_name, class_list, label):
    feature_vale_names = train_data[feature_name].unique()
    row_num = train_data.shape[0]
    feature_information = 0

    for feature_value in feature_vale_names:
        feature_vale_data = train_data[train_data[feature_name] == feature_value]
        feature_value_num = feature_vale_data.shape[0]
        feature_value_entropy = get_entropy(feature_vale_data, class_list, feature_name)
        feature_value_probability = feature_value_num / row_num
        feature_information += feature_value_entropy * feature_value_probability
    return get_total_entropy(train_data, class_list, label) - feature_information


'''
To get the feature with highest IG
train_data: dataset by pandas
class_list: a list of classes
label: a string of the current class(label)
'''


def get_highest_IG_feature(train_data, class_list, label):
    feature_list = train_data.columns.drop(label)
    max_ig = -1
    max_ig_feature = None

    for feature in feature_list:
        feature_ig = get_IG(train_data, feature, class_list, label)
        if max_ig < feature_ig:
            max_ig = feature_ig
            max_ig_feature = feature
    return max_ig_feature


'''
Generate a asub tree
train_data: dataset by pandas
class_list: a list of classes
feature_name: a string of the current feature's name
label: a string of the current class(label)
'''


def generate_subtree(train_data, feature, class_list, label):
    feature_value_counter_dict = train_data[feature].value_counts(sort=False)
    tree = {}

    for feature_value, count in feature_value_counter_dict.iteritems():
        feature_value_data = train_data[train_data[feature] == feature_value]
        flag_for_node = False
        for c in class_list:
            class_num = feature_value[feature_value[label] == c].shape[0]
            if class_num == count:
                tree[feature_value] = c
                train_data = train_data[train_data[feature] != feature_value]
                flag_for_node = True
        if not flag_for_node:
            tree[feature_value] = "?"
    return tree, train_data


'''
Build up the tree
root: a dictionary of the current pointed node
prev_feature_value: the previous value of the pointed feature
train_data: dataset by pandas
class_list: a list of classes
label: a string of the current class(label)
'''


def build_tree(root, prev_feature_value, train_data, class_list, label):
    if train_data.shape[0] != 0:
        max_information_feature = get_highest_IG_feature(train_data, class_list, label)
        tree, train_data = generate_subtree(train_data, max_information_feature, class_list, label)
        root_next = None

        if prev_feature_value is not None:
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_information_feature] = tree
            root_next = root[prev_feature_value][max_information_feature]
        else:
            root[max_information_feature] = tree
            root_next = root[max_information_feature]
        for node, branch in list(root_next.items()):
            if branch == "?":
                feature_value_data = train_data[
                    train_data[max_information_feature] == node]  # using the updated dataset
                build_tree(root_next, node, feature_value_data, label, class_list)


'''
implementing ID3 algo
train_data: dataset by pandas
label: a string of the current class(label)
'''


def ID3(train_data, label):
    train_data_copy = train_data.copy()
    tree = {}
    class_list = train_data_copy[label].unique()
    build_tree(tree, None, train_data, class_list, label)
    return tree


'''
predicting with the trained tree
tree: trained tree
instance: testing data
'''


def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    else:
        root_node = next(iter(tree))
        feature_value = instance[root_node]
        if feature_value in tree[root_node]:
            return predict(tree[root_node][feature_value], instance)
        else:
            return None


def evaluate(tree, test_data_m, label):
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows():
        result = predict(tree, test_data_m.iloc[index])
        if result == test_data_m[label].iloc[index]:
            correct_preditct += 1
        else:
            wrong_preditct += 1
    accuracy = correct_preditct / (correct_preditct + wrong_preditct)
    return accuracy
