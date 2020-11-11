# Try to determine the value type
from datetime import datetime


def determine_type(value):
    for (type, condition) in [
        (int, int),
        (float, float),
        (datetime, lambda value: datetime.strptime(value, "%Y/%m/%d"))
    ]:
        try:
            condition(value)
            return type
        except ValueError:
            continue

    return str
