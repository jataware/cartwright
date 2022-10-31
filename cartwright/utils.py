from collections import defaultdict
from fuzzywuzzy import fuzz

import dateutil
import dateutil.parser
import datetime
import logging


def fuzzy_match(word1, word2, ratio_=95):
    Ratio = fuzz.ratio(word1.lower(), word2.lower())

    if Ratio > ratio_:
        return True


def fuzzy_ratio(word1, word2, ratio_=95):
    Ratio = fuzz.ratio(word1.lower(), word2.lower())
    if Ratio > ratio_:
        return True, Ratio


def build_return_date_object(format, util="Util"):
    return {
        "category": "time",
        "subcategory": "date",
        "format": format,
        "match_type": ["LSTM"],
        "Parser": f"{util}",
    }


def build_return_standard_object(category, subcategory, match_type):
    return {
        "category": category,
        "subcategory": subcategory,
        "format": None,
        "match_type": [match_type],
        "Parser": None,
    }


def build_return_timespan(format):
    return {
        "category": "time",
        "subcategory": "timespan",
        "format": format,
        "match_type": ["LSTM"],
        "Parser": None,
    }


def date_util_span(dates):
    validDate = []
    for date in dates:
        dateUtil = dateutil.parser.parse(str(date), dayfirst=False)
        if isinstance(dateUtil, datetime.date):
            validDate.append({"value": date, "standard": dateUtil})
    return validDate


character_tokins = defaultdict(
    int,
    {
        "PAD": 0,
        "UNK": 1,
        "a": 2,
        "b": 3,
        "c": 4,
        "d": 5,
        "e": 6,
        "f": 7,
        "g": 8,
        "h": 9,
        "i": 10,
        "j": 11,
        "k": 12,
        "l": 13,
        "m": 14,
        "n": 15,
        "o": 16,
        "p": 17,
        "q": 18,
        "r": 19,
        "s": 20,
        "t": 21,
        "u": 22,
        "v": 23,
        "w": 24,
        "x": 25,
        "y": 26,
        "z": 27,
        "A": 28,
        "B": 29,
        "C": 30,
        "D": 31,
        "E": 32,
        "F": 33,
        "G": 34,
        "H": 35,
        "I": 36,
        "J": 37,
        "K": 38,
        "L": 39,
        "N": 40,
        "O": 41,
        "P": 42,
        "Q": 43,
        "R": 44,
        "S": 45,
        "T": 46,
        "U": 47,
        "V": 48,
        "W": 49,
        "X": 50,
        "Y": 51,
        "Z": 52,
        "1": 53,
        "2": 54,
        "3": 55,
        "4": 56,
        "5": 57,
        "6": 58,
        "7": 59,
        "8": 60,
        "9": 61,
        "0": 62,
        "'": 63,
        ",": 64,
        ".": 65,
        ";": 66,
        "*": 67,
        "!": 68,
        "@": 68,
        "#": 70,
        "$": 71,
        "%": 72,
        "^": 73,
        "&": 74,
        "(": 75,
        ")": 76,
        "_": 77,
        "=": 78,
        "-": 79,
        ":": 80,
        "+": 81,
        "/": 82,
        "\\": 83,
        "`": 84,
        "~": 85,
        "|": 86,
        "}": 87,
        "{": 88,
    },
)

days_of_the_week_A = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]
days_of_the_week_a = ["mon", "tues", "wed", "thur", "fri", "sat", "sun"]
months_of_the_year_B = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
months_of_the_year_b = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sept",
    "oct",
    "nov",
    "dec",
]
columns_to_classify_and_skip_if_found = [
    {"Lat": "latitude"},
    {"Latitude": "latitude"},
    {"lng": "latitude"},
    {"lon": "longitude"},
    {"long": "longitude"},
    {"Longitude": "longitude"},
    {"ISO2": "ISO2"},
    {"ISO3": "ISO3"},
]
columns_to_classify_if_found = [
    {"Date": "Date"},
    {"Datetime": "Datetime"},
    {"Timestamp": "Timestamp"},
    {"Epoch": "Epoch"},
    {"Time": "Time"},
    {"Year": "Year"},
    {"Month": "Month"},
    {"Lat": "Latitude"},
    {"Latitude": "Latitude"},
    {"lng": "Latitude"},
    {"lon": "Longitude"},
    {"long": "Longitude"},
    {"Longitude": "Longitude"},
    {"Geo": "Geo"},
    {"Coordinates": "Coordinates"},
    {"Location": "Location"},
    {"West": "West"},
    {"South": "South"},
    {"East": "East"},
    {"North": "North"},
    {"Country": "Country"},
    {"CountryName": "CountryName"},
    {"CC": "CC"},
    {"CountryCode": "CountryCode"},
    {"State": "State"},
    {"City": "City"},
    {"Town": "Town"},
    {"Region": "Region"},
    {"Province": "Province"},
    {"Territory": "Territory"},
    {"Address": "Address"},
    {"ISO2": "ISO2"},
    {"ISO3": "ISO3"},
    {"ISO_code": "ISO_code"},
    {"Results": "Results"},
]
