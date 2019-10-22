"""
Utility methods to make life easier.
"""
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


def _load_wikipedia(test_size=0.2, random_state=69, data_dir='data', return_feature=False):
    data = np.load(os.path.join(data_dir, 'wikipedia/data.npy'))
    X = data[:, :-1]
    y = data[:, -1].astype(np.int32)
    label = ['non-spammer', 'spammer']

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        result = (X_train, X_test, y_train, y_test, label)
    else:
        result = (X, y, label)

    if return_feature:
        feature_names = np.load(os.path.join(data_dir, 'wikipedia/feature.npy'))
        result += (feature_names,)

    return result


def _load_banknote(test_size=0.2, random_state=69, data_dir='data', return_feature=False):
    data = np.load(os.path.join(data_dir, 'banknote/data.npy'))
    X = data[:, :-1]
    y = data[:, -1].astype(np.int32)
    label = ['0', '1']

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        result = (X_train, X_test, y_train, y_test, label)
    else:
        result = (X, y, label)

    if return_feature:
        feature_names = np.load(os.path.join(data_dir, 'banknote/feature.npy'))
        result += (feature_names,)

    return result


def _load_iris(test_size=0.2, random_state=69, return_feature=False):
    data = load_iris()
    X, y, label = data['data'], data['target'], data['target_names']
    feature_names = data['feature_names']

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        result = (X_train, X_test, y_train, y_test, label)
    else:
        result = (X, y, label)

    if return_feature:
        result += (feature_names,)

    return result


def _load_breast(test_size=0.2, random_state=69, return_feature=False):
    data = load_breast_cancer()
    X, y, label = data['data'], data['target'], data['target_names']
    feature_names = data['feature_names']

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        result = (X_train, X_test, y_train, y_test, label)
    else:
        result = (X, y, label)

    if return_feature:
        result += (feature_names,)

    return result


def _load_wine(test_size=0.2, random_state=69, return_feature=False):
    data = load_wine()
    X, y, label = data['data'], data['target'], data['target_names']
    feature_names = data['feature_names']

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        result = (X_train, X_test, y_train, y_test, label)
    else:
        result = (X, y, label)

    if return_feature:
        result += (feature_names,)

    return result


def _load_adult(data_dir='data'):
    train = np.load(os.path.join(data_dir, 'adult/train.npy'))
    test = np.load(os.path.join(data_dir, 'adult/test.npy'))
    label = ['<=50K', '>50k']
    X_train = train[:, :-1]
    y_train = train[:, -1].astype(np.int32)
    X_test = test[:, :-1]
    y_test = test[:, -1].astype(np.int32)
    return X_train, X_test, y_train, y_test, label


def _load_amazon(data_dir='data'):
    train = np.load(os.path.join(data_dir, 'amazon/train.npy'))
    test = np.load(os.path.join(data_dir, 'amazon/test.npy'))
    label = ['0', '1']
    X_train = train[:, 1:]
    y_train = train[:, 0].astype(np.int32)
    X_test = test[:, 1:]
    y_test = test[:, 0].astype(np.int32)
    return X_train, X_test, y_train, y_test, label


def _load_churn(data_dir='data', test_size=0.2, random_state=69):
    data = np.load(os.path.join(data_dir, 'churn/data.npy'))
    label = ['no', 'yes']
    X = data[:, :-1]
    y = data[:, -1].astype(np.int32)

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test, label
    else:
        return X, y, label


def _load_creditcard(data_dir='data', test_size=0.2, random_state=69):
    data = np.load(os.path.join(data_dir, 'creditcard/data.npy'))
    label = ['0', '1']
    X = data[:, :-1]
    y = data[:, -1].astype(np.int32)

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test, label
    else:
        return X, y, label


def _load_heart(data_dir='data', test_size=0.2, random_state=69):
    data = np.load(os.path.join(data_dir, 'heart/data.npy'))
    label = ['other', 'Hungary']
    X = data[:, :-1]
    y = data[:, -1].astype(np.int32)

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test, label
    else:
        return X, y, label


def _load_hospital(data_dir='data', feature=False):
    train = np.load(os.path.join(data_dir, 'hospital/train.npy'))
    test = np.load(os.path.join(data_dir, 'hospital/test.npy'))
    label = ['not readmitted', 'readmitted']
    X_train = train[:, :-1]
    y_train = train[:, -1].astype(np.int32)
    y_train[np.where(y_train == -1)] = 0
    X_test = test[:, :-1]
    y_test = test[:, -1].astype(np.int32)
    y_test[np.where(y_test == -1)] = 0

    if feature:
        feature = np.load(os.path.join(data_dir, 'hospital/feature.npy'), allow_pickle=True)
        return X_train, X_test, y_train, y_test, label, feature
    else:
        return X_train, X_test, y_train, y_test, label


def _load_hospital2(data_dir='data', feature=False):
    train = np.load(os.path.join(data_dir, 'hospital2/train.npy'))
    test = np.load(os.path.join(data_dir, 'hospital2/test.npy'))
    label = ['not readmitted', 'readmitted']
    X_train = train[:, :-1]
    y_train = train[:, -1].astype(np.int32)
    y_train[np.where(y_train == -1)] = 0
    X_test = test[:, :-1]
    y_test = test[:, -1].astype(np.int32)
    y_test[np.where(y_test == -1)] = 0

    if feature:
        feature = np.load(os.path.join(data_dir, 'hospital2/feature.npy'), allow_pickle=True)
        return X_train, X_test, y_train, y_test, label, feature
    else:
        return X_train, X_test, y_train, y_test, label


def _load_nc17_mfc18(data_dir='data', return_feature=False, return_image_id=False, return_manipulation=False,
                     remove_missing_features=False):
    train = np.load(os.path.join(data_dir, 'nc17_mfc18/nc17.npy'))
    test = np.load(os.path.join(data_dir, 'nc17_mfc18/mfc18.npy'))
    label = ['non-manipulated', 'manipulated']
    X_train = train[:, :-1]
    y_train = train[:, -1].astype(np.int32)
    X_test = test[:, :-1]
    y_test = test[:, -1].astype(np.int32)

    # remove features that have missing values in the test set
    if remove_missing_features:
        remove_ndx = np.array([2, 4, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 29, 31, 32, 33, 34])
        X_train = np.delete(X_train, remove_ndx, axis=1)
        X_test = np.delete(X_test, remove_ndx, axis=1)

    result = (X_train, X_test, y_train, y_test, label)

    if return_feature:
        feature = np.load(os.path.join(data_dir, 'nc17_mfc18/feature.npy'))[:-1]
        if remove_missing_features:
            feature = np.delete(feature, remove_ndx)
        result += (feature,)

    if return_manipulation:
        manipulation = pd.read_csv(os.path.join(data_dir, 'nc17_mfc18/nc17_manipulations.csv'))
        manipulation_names = np.array(manipulation.columns)[2:]
        manipulation = manipulation.to_numpy()[:, 2:]
        result += (manipulation, manipulation_names)

        manipulation = pd.read_csv(os.path.join(data_dir, 'nc17_mfc18/mfc18_manipulations.csv'))
        manipulation_names = np.array(manipulation.columns)[2:]
        manipulation = manipulation.to_numpy()[:, 2:]
        result += (manipulation, manipulation_names)

    if return_image_id:
        train_id = pd.read_csv(os.path.join(data_dir, 'nc17_mfc18/nc17_reference.csv'))['image_id'].values
        test_id = pd.read_csv(os.path.join(data_dir, 'nc17_mfc18/mfc18_reference.csv'))['image_id'].values
        result += (train_id, test_id)

    return result


def _load_mfc18_mfc19(data_dir='data', return_feature=False, return_image_id=False, return_manipulation=False):
    train = np.load(os.path.join(data_dir, 'mfc18_mfc19/mfc18.npy'))
    test = np.load(os.path.join(data_dir, 'mfc18_mfc19/mfc19.npy'))
    label = ['non-manipulated', 'manipulated']
    X_train = train[:, :-1]
    y_train = train[:, -1].astype(np.int32)
    X_test = test[:, :-1]
    y_test = test[:, -1].astype(np.int32)

    # remove features that have missing values in the test set
    remove_train_ndx = np.where(X_train == -1)[0]
    remove_test_ndx = np.where(X_test == -1)[0]
    remove_ndx = np.union1d(remove_train_ndx, remove_test_ndx)
    X_train = np.delete(X_train, remove_ndx, axis=1)
    X_test = np.delete(X_test, remove_ndx, axis=1)

    result = (X_train, X_test, y_train, y_test, label)

    if return_feature:
        feature = np.load(os.path.join(data_dir, 'mfc18_mfc19/feature.npy'))
        feature = np.delete(feature, remove_ndx)
        result += (feature,)

    if return_manipulation:
        manipulation = pd.read_csv(os.path.join(data_dir, 'mfc18_mfc19/mfc18_manipulations.csv'))
        manipulation_names = np.array(manipulation.columns)[2:]
        manipulation = manipulation.to_numpy()[:, 2:]
        result += (manipulation, manipulation_names)

        manipulation = pd.read_csv(os.path.join(data_dir, 'mfc18_mfc19/mfc19_manipulations.csv'))
        manipulation_names = np.array(manipulation.columns)[2:]
        manipulation = manipulation.to_numpy()[:, 2:]
        result += (manipulation, manipulation_names)

    if return_image_id:
        train_id = pd.read_csv(os.path.join(data_dir, 'mfc18_mfc19/mfc18_reference.csv'))['image_id'].values
        test_id = pd.read_csv(os.path.join(data_dir, 'mfc18_mfc19/mfc19_reference.csv'))['image_id'].values
        result += (train_id, test_id)

    return result


def _load_medifor(data_dir='data', test_size=0.2, random_state=69, return_feature=False, return_manipulation=False,
                  return_image_id=False, dataset='MFC18_EvalPart1'):

    data = np.load(os.path.join(data_dir, dataset, 'data.npy'))
    label = ['non-manipulated', 'manipulated']
    X = data[:, :-1]
    y = data[:, -1].astype(np.int32)

    result = tuple()

    if test_size is not None:
        train_ndx, test_ndx = train_test_split(np.arange(len(X)), test_size=test_size,
                                               random_state=random_state, stratify=y)
        X_train, y_train = X[train_ndx], y[train_ndx]
        X_test, y_test = X[test_ndx], y[test_ndx]
        result += (X_train, X_test, y_train, y_test, label)

        if return_feature:
            feature = np.load(os.path.join(data_dir, dataset, 'feature.npy'))
            result += (feature,)

        if return_manipulation:
            manipulation = pd.read_csv(os.path.join(data_dir, dataset, 'manipulations.csv'))
            manip_label = np.array(manipulation.columns)[2:]
            manipulation = manipulation.to_numpy()[:, 2:]
            manip_train, manip_test = manipulation[train_ndx], manipulation[test_ndx]
            result += (manip_train, manip_test, manip_label)

        if return_image_id:
            assert dataset in ['MFC18_EvalPart1', 'MFC19_EvalPart1'], 'image_id not supported for {}'.format(dataset)
            image_name = pd.read_csv(os.path.join(data_dir, dataset, 'reference.csv'))['image_id'].values
            id_train, id_test = image_name[train_ndx], image_name[test_ndx]
            result += (id_train, id_test)

        return result

    else:
        result = (X, y, label)

        if return_feature:
            feature = np.load(os.path.join(data_dir, dataset, 'feature.npy'))
            result += (feature,)

        if return_manipulation:
            manipulation = pd.read_csv(os.path.join(data_dir, dataset, 'manipulations.csv'))
            manip_label = np.array(manipulation.columns)[2:]
            manipulation = manipulation.to_numpy()[:, 2:]
            result += (manipulation, manip_label)

        if return_image_id:
            assert dataset in ['MFC18_EvalPart1', 'MFC19_EvalPart1'], 'image_id not supported for {}'.format(dataset)
            image_id = pd.read_csv(os.path.join(data_dir, dataset, 'reference.csv'))['image_id'].values
            result += (image_id,)

        return result


def _load_mnist(data_dir='data', dataset='mnist'):

    train_data = np.load(os.path.join(data_dir, 'mnist', 'train.npy'))
    test_data = np.load(os.path.join(data_dir, 'mnist', 'test.npy'))

    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]

    # extract classes to keep
    if '_' in dataset:
        label = []
        classes = dataset.split('_')[-1]
        for c in classes:
            c = int(c)
            assert c >= 0 and c <= 9, '{} class not eligible!'.format(c)
            label.append(c)

        # filter images
        train_ndx_list = []
        test_ndx_list = []

        for c in label:
            train_ndx_list.append(np.where(y_train == c)[0])
            test_ndx_list.append(np.where(y_test == c)[0])

        train_ndx = np.sort(np.concatenate(train_ndx_list))
        test_ndx = np.sort(np.concatenate(test_ndx_list))

        X_train, y_train = X_train[train_ndx], y_train[train_ndx]
        X_test, y_test = X_test[test_ndx], y_test[test_ndx]

    return X_train, X_test, y_train, y_test, label


def _load_20newsgroups(data_dir='data', dataset='alt.atheism|talk.religion.misc',
                       remove=('headers', 'footers', 'quotes'), return_raw=True):
    categories = dataset.split('_')
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(newsgroups_train.data)
    y_train = newsgroups_train.target
    X_test = vectorizer.transform(newsgroups_test.data)
    y_test = newsgroups_test.target
    label = newsgroups_train.target_names

    result = X_train, X_test, y_train, y_test, label

    if return_raw:
        result += (newsgroups_train.data, newsgroups_test.data)

    return result


def _load_dota2(data_dir='data'):
    train = np.load(os.path.join(data_dir, 'dota2/train.npy'))
    test = np.load(os.path.join(data_dir, 'dota2/test.npy'))
    label = ['-1', '1']
    X_train = train[:, 1:]
    y_train = train[:, 0].astype(np.int32)
    X_test = test[:, 1:]
    y_test = test[:, 0].astype(np.int32)
    return X_train, X_test, y_train, y_test, label


def _load_census(data_dir='data'):
    train = np.load(os.path.join(data_dir, 'census/train.npy'))
    test = np.load(os.path.join(data_dir, 'census/test.npy'))
    label = ['0', '1']
    X_train = train[:, :-1]
    y_train = train[:, -1].astype(np.int32)
    X_test = test[:, :-1]
    y_test = test[:, -1].astype(np.int32)
    return X_train, X_test, y_train, y_test, label


def get_data(dataset, test_size=0.2, random_state=69, data_dir='data', return_feature=False,
             return_manipulation=False, return_image_id=False, remove_missing_features=False,
             categories='alt.atheism|talk.religion.misc', remove=('headers', 'footers', 'quotes'),
             return_raw=True):
    """Returns a train and test set from the desired dataset."""

    # load dataset
    if dataset == 'iris':
        return _load_iris(test_size=test_size, random_state=random_state, return_feature=return_feature)
    elif dataset == 'breast':
        return _load_breast(test_size=test_size, random_state=random_state, return_feature=return_feature)
    elif dataset == 'wine':
        return _load_wine(test_size=test_size, random_state=random_state, return_feature=return_feature)
    elif dataset == 'adult':
        return _load_adult(data_dir=data_dir)
    elif dataset == 'census':
        return _load_census(data_dir=data_dir)
    elif dataset == 'amazon':
        return _load_amazon(data_dir=data_dir)
    elif dataset == 'banknote':
        return _load_banknote(test_size=test_size, random_state=random_state, data_dir=data_dir,
                              return_feature=return_feature)
    elif dataset == 'wikipedia':
        return _load_wikipedia(test_size=test_size, random_state=random_state, data_dir=data_dir,
                               return_feature=return_feature)
    elif dataset == 'churn':
        return _load_churn(data_dir=data_dir, test_size=test_size, random_state=random_state)
    elif dataset == 'creditcard':
        return _load_creditcard(data_dir=data_dir, test_size=test_size, random_state=random_state)
    elif dataset == 'heart':
        return _load_heart(data_dir=data_dir, test_size=test_size, random_state=random_state)
    elif dataset == 'hospital':
        return _load_hospital(data_dir=data_dir, feature=return_feature)
    elif dataset == 'hospital2':
        return _load_hospital2(data_dir=data_dir, feature=return_feature)
    elif dataset == 'nc17_mfc18':
        return _load_nc17_mfc18(data_dir=data_dir, return_feature=return_feature,
                                return_manipulation=return_manipulation, return_image_id=return_image_id,
                                remove_missing_features=remove_missing_features)
    elif dataset == 'mfc18_mfc19':
        return _load_mfc18_mfc19(data_dir=data_dir, return_feature=return_feature,
                                 return_manipulation=return_manipulation, return_image_id=return_image_id)
    elif dataset == 'NC17_EvalPart1':
        return _load_medifor(data_dir=data_dir, return_feature=return_feature, return_manipulation=return_manipulation,
                             dataset=dataset, test_size=test_size, random_state=random_state)
    elif dataset == 'MFC18_EvalPart1':
        return _load_medifor(data_dir=data_dir, return_feature=return_feature, return_manipulation=return_manipulation,
                             return_image_id=return_image_id, dataset=dataset, test_size=test_size,
                             random_state=random_state)
    elif dataset == 'MFC19_EvalPart1':
        return _load_medifor(data_dir=data_dir, return_feature=return_feature, return_manipulation=return_manipulation,
                             return_image_id=return_image_id, dataset=dataset, test_size=test_size,
                             random_state=random_state)
    elif dataset.startswith('mnist'):
        return _load_mnist(data_dir=data_dir, dataset=dataset)
    elif dataset == '20newsgroups':
        return _load_20newsgroups(data_dir=data_dir, dataset=categories, remove=remove, return_raw=return_raw)
    elif dataset == 'dota2':
        return _load_dota2(data_dir=data_dir)
    else:
        exit('dataset {} not supported'.format(dataset))


def flip_labels(arr, k=100, random_state=69, return_indices=True):
    """Flips the label of random elements in an array; only for binary arrays."""

    assert arr.ndim == 1, 'arr is not 1d!'
    assert np.all(np.unique(arr) == np.array([0, 1])), 'arr is not binary!'
    if k <= 1.0:
        assert isinstance(k, float), 'k is not a float!'
        assert k > 0, 'k is less than zero!'
        k = int(len(arr) * k)
    assert k <= len(arr), 'k is greater than len(arr)!'

    np.random.seed(random_state)
    indices = np.random.choice(np.arange(len(arr)), size=k, replace=False)

    new_arr = arr.copy()
    ones_flipped = 0
    zeros_flipped = 0

    for ndx in indices:
        if new_arr[ndx] == 1:
            ones_flipped += 1
        else:
            zeros_flipped += 1
        new_arr[ndx] = 0 if new_arr[ndx] == 1 else 1

    print('sum before: {}'.format(np.sum(arr)))
    print('ones flipped: {}'.format(ones_flipped))
    print('zeros flipped: {}'.format(zeros_flipped))
    print('sum after: {}'.format(np.sum(new_arr)))
    assert np.sum(new_arr) == np.sum(arr) - ones_flipped + zeros_flipped

    if return_indices:
        return new_arr, indices
    else:
        return new_arr


def flip_labels_with_indices(arr, indices):
    """Flips the label of specified elements in an array; only for binary arrays."""

    assert arr.ndim == 1, 'arr is not 1d!'
    assert np.all(np.unique(arr) == np.array([0, 1])), 'arr is not binary!'

    new_arr = arr.copy()
    for ndx in indices:
        new_arr[ndx] = 0 if new_arr[ndx] == 1 else 1
    return new_arr
