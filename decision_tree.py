from math import log


def create_branch(features, labels, depth_gas):
    if depth_gas < 0 or entropy(labels) <= 1e-10:
        return {'type': 'leaf', "label": labels[0]}

    feat, div, f_groups, l_groups, entrop = find_divider(features, labels)

    return {'type': 'branch', 'divider_feature': feat, 'divider_level': div, 'children':
        [create_branch(f_groups[i], l_groups[i], depth_gas - 1) for i in range(2)]
    }


def create_tree(features, labels, depth_gas):
    return create_branch(features, labels, depth_gas)


def find_divider(features, labels):
    best_divider = -1
    best_feature = 0
    best_entropy = None
    best_features = [[], []]
    best_labels = [[], []]
    for divider_feature_index, _ in enumerate(features[0]):
        for divider_sample_index, _ in enumerate(features):
            divided_labels = [[], []]
            divided_features = [[], []]
            for divided_sample_index, _ in enumerate(features):
                if features[divider_sample_index][divider_feature_index] < features[divided_sample_index][divider_feature_index]:
                    part = 0
                else:
                    part = 1
                divided_labels[part].append(labels[divided_sample_index])
                divided_features[part].append(features[divided_sample_index])
            current_entropy = sum(entropy(divided_labels[part]) for part in [0, 1])

            if current_entropy < 1e-6:
                return divider_feature_index, features[divider_sample_index][divider_feature_index], divided_features, divided_labels, 0

            if (not best_entropy or current_entropy < best_entropy) and len(divided_labels[0]) and len(divided_labels[1]):
                best_divider = features[divider_sample_index][divider_feature_index]
                best_feature = divider_feature_index
                best_features = divided_features
                best_labels = divided_labels
                best_entropy = current_entropy
    return best_feature, best_divider, best_features, best_labels, best_entropy


def entropy(samples_of_labels):
    result = 0.0
    population = len(samples_of_labels)
    if not population:
        return 0
    for label_index, _ in enumerate(samples_of_labels[0]):
        proportions = {}
        labels_trans = list(zip(*samples_of_labels))
        for item in labels_trans[label_index]:
            proportions[item] = proportions.get(item, 0) + 1
        for count in proportions.values():
            proportion = count / population
            result += count * log(proportion)
    return - result


def predict_branch(branch, feature):
    if branch['type'] == 'leaf':
        return branch['label']
    else:
        if branch['divider_level'] < feature[branch['divider_feature']]:
            return predict_branch(branch['children'][0], feature)
        else:
            return predict_branch(branch['children'][1], feature)


def predict(tree, features):
    results = []
    for feature in features:
        results.append(predict_branch(tree, feature))
    return results