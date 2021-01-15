import numpy as np
from sklearn.datasets import load_iris

# Version 2: Keeps track of surrounding feature values and their counts.
def get_candidate_splits2(x, y):
    """
    Extracts all candidate splits for a given feature vector.
    """
    assert len(x) == len(y) > 0

    indices = np.argsort(x)
    for ndx in indices:
        print(x[ndx], y[ndx])

    v1_count = -1
    v1_pos_count = -1
    v2_count = -1
    v2_pos_count = -1

    prev_val = x[indices[0]]
    prev_label = y[indices[0]]
    cur_val = None
    cur_label = None

    # candidate split containers
    left_counts = []
    left_pos_counts = []
    right_counts = []
    right_pos_counts = []
    split_vals = []

    count = 1
    pos_count = y[indices[0]]
    n_samples = len(x)
    n_pos_samples = np.sum(y)

    print(n_samples)
    print(n_pos_samples)

    v_count = 1
    v_pos_count = y[indices[0]]

    values = []
    counts = []
    pos_counts = []
    v_counts = []
    v_pos_counts = []

    # loop through sorted feature value indices
    for i in range(1, len(indices)):
        ndx = indices[i]
        cur_val = x[ndx]
        cur_label = y[ndx]

        # next feature
        if cur_val > prev_val + 1e-7:
            values.append(prev_val)
            counts.append(count)
            pos_counts.append(pos_count)
            v_counts.append(v_count)
            v_pos_counts.append(v_pos_count)
            v_count = 1
            v_pos_count = cur_label

        # same feature value
        else:
            v_count += 1
            v_pos_count += cur_label

        count += 1
        pos_count += cur_label
        prev_val = cur_val
        prev_label = cur_label

    # handle last feature
    if v_count > 0:
        values.append(prev_val)
        counts.append(count)
        pos_counts.append(pos_count)
        v_counts.append(v_count)
        v_pos_counts.append(v_pos_count)
        v_count = 0
        v_pos_count = 0

    # evaluate each pair of feature values
    results = []
    for i in range(1, len(values)):
        v1 = values[i-1]
        v2 = values[i]
        v1_count = counts[i-1]
        v2_count = counts[i]
        v1_pos_count = pos_counts[i-1]
        v2_pos_count = pos_counts[i]
        v1_v_count = v_counts[i-1]
        v2_v_count = v_counts[i]
        v1_v_pos_count = v_pos_counts[i-1]
        v2_v_pos_count = v_pos_counts[i]

        v1_label_ratio = v1_v_pos_count / v1_v_count
        v2_label_ratio = v2_v_pos_count / v2_v_count

        # save threshold
        if ((v1_label_ratio != v2_label_ratio) or
            (v1_label_ratio > 0.0 and v1_label_ratio < 1.0) or
            (v2_label_ratio > 0.0 and v2_label_ratio < 1.0)):
            value = (v1 + v2) / 2
            left_count = v1_count
            left_pos_count = v1_pos_count
            right_count = n_samples - left_count
            right_pos_count = n_pos_samples - left_pos_count
            results.append((value, v1_v_count, v1_v_pos_count, v2_v_count, v2_v_pos_count,
                            left_count, left_pos_count, right_count, right_pos_count))

    for result in results:
        print(result)


# Version 1: Does not take into account metadata about
#            surrounding feature value counts.
def get_candidate_splits1(x, y):
    """
    Extracts all candidate splits for a given feature vector.
    """
    assert len(x) == len(y) > 0

    indices = np.argsort(x)
    for ndx in indices:
        print(x[ndx], y[ndx])

    prev_val = x[indices[0]]
    prev_label = y[indices[0]]
    cur_val = None
    cur_label = None
    split_ndx = None
    split_val = None

    # candidate split containers
    left_counts = []
    left_pos_counts = []
    right_counts = []
    right_pos_counts = []
    split_vals = []

    count = 1
    pos_count = y[indices[0]]
    n_samples = len(x)
    n_pos_samples = np.sum(y)

    has_split = False

    # loop through sorted feature value indices
    for i in range(1, len(indices)):
        ndx = indices[i]
        cur_val = x[ndx]
        cur_label = y[ndx]

        # same feature value
        if cur_val <= prev_val + 1e-7:

            # feature value contains samples with multiple labels
            if cur_label != prev_label:
                cur_label = -1

        # next feature value
        else:

            # compute midway point between adjacent feature values
            split_ndx = i
            split_val = (cur_val + prev_val) / 2
            split_left_count = count
            split_left_pos_count = pos_count
            has_split = False

        # save candidate split
        if cur_label != prev_label and not has_split and split_ndx is not None:
            print('evaluate split at {}, {:.2f}'.format(split_ndx, split_val))

            left_counts.append(split_left_count)
            right_counts.append(n_samples - split_left_count)
            left_pos_counts.append(split_left_pos_count)
            right_pos_counts.append(n_pos_samples - split_left_pos_count)
            split_vals.append(split_val)

            has_split = True

        # move pointers
        prev_val = cur_val
        prev_label = cur_label

        count += 1
        pos_count += y[ndx]

    return list(zip(split_vals, left_counts, left_pos_counts, right_counts, right_pos_counts))


if __name__ == '__main__':
    data = load_iris()
    X = data['data']
    y = data['target']
    indices = np.where(y != 2)[0]

    X = X[indices]
    y = y[indices]

    x = X[:, 0]
    # x = np.concatenate([np.array([4.2, 4.2]), np.array([4.1, 4.1]), x, np.array([7.1])])
    # y = np.concatenate([np.array([0, 1]), np.array([0, 0]), y, np.array([0])])

    # x = np.array([1, 1, 1, 2, 2, 2])
    # y = np.array([0, 1, 0, 0, 0, 0])

    results = get_candidate_splits2(x, y)

    n_samples = len(x)
    n_pos_samples = np.sum(y)

    # print(n_samples, n_pos_samples)
    # for split_val, left_count, left_pos_count, right_count, right_pos_count in results:
    #     print(split_val, left_count, left_pos_count, right_count, right_pos_count)
    #     print(left_count + right_count, left_pos_count + right_pos_count)
    #     print()
