import collections

import pandas
from typing import List, Any, Dict


def unique(list: List[Any]):
    last = object()
    for item in sorted(list, key=lambda item: (item is None, item)):
        if item == last:
            continue
        yield item
        last = item


def unique_dict_of_lists(dictionary: Dict[Any, List[Any]]):
    result = dictionary
    for key in dictionary.keys():
        result[key] = list(unique(result[key]))

    return result


def merge_dictionaries(dictionaries: List[Dict[Any, Any]]):
    merged = collections.defaultdict(list)
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            merged[key].append(value)

    return dict(merged)


def merge_tables(tables):
    return pandas.concat([table for table in tables if len(table) > 0], sort=False).reset_index(drop=True)
