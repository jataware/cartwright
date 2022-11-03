from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from pydantic import BaseModel, constr, Field
from typing import List, Optional

class FuzzyCategory(str,Enum):
    """
        fuzzyCategory are the categories we try to capture with fuzzy matching.
    """
    Date= "Date"
    Datetime = "Datetime"
    Timestamp="Timestamp"
    Epoch= "Epoch"
    Time= "Time"
    Year= "Year"
    Month= "Month"
    Latitude = "Latitude"
    Longitude= "Longitude"
    Geo= "Geo"
    Coordinates= "Coordinates"
    Location= "Location"
    West= "West"
    South= "South"
    East= "East"
    North= "North"
    Country= "Country"
    CountryName= "CountryName"
    CC="CC"
    CountryCode= "CountryCode"
    State= "State"
    City ="City"
    Town= "Town"
    Region ="Region"
    Province= "Province"
    Territory= "Territory"
    Address= "Address"
    ISO2= "ISO2"
    ISO3 = "ISO3"
    ISO_code= "ISO_code"
    Results= "Results"

class Category(str, Enum):
    """
    category is the general classification for a column
    """
    geo= "geo"
    time="time"
    boolean="boolean"
    timeout="timeout"


class Subcategory(str, Enum):
    """
    subcategory is the classification of the column at a finer scale than category.
    """
    city_name="city_name"
    state_name="state_name"
    country_name="country_name"
    ISO3="ISO3"
    ISO2="ISO2"
    continent="continent"
    longitude="longitude"
    latitude="latitude"
    date="date"
    timespan="timespan"
    country="country"
    state="state"
    city="city"
    town="town"
    region="region"
    province="province"
    territory="territory"

class Matchtype(str, Enum):
    """
     is the type of match for classification if any.
    """
    fuzzy="fuzzy"
    LSTM="LSTM"

class FuzzyColumn(BaseModel):
    """
       fuzzyColumn is only defined when a column header matches a word we are looking for. fuzzyCategory is used for classifying a column.
    """
    matchedKey: str = Field(default=None, description='This is the word that was matched with the column header. If a column header was Lat, it would match with the the matchedKey of Lat, since it is one of the lookup words. In this case the fuzzyCategory would be returned as "Latitude".')
    FuzzyCategory: Optional[FuzzyCategory]
    ratio: int = Field(default=None, description='Ratio of the fuzzy match. If it was an exact match it would be 100')

class Parser(str,Enum):
    """
        Parser records which python library the date was parsed with. dateutil or arrow.
    """
    Util="Util"
    arrow="arrow"


class Uniformity(Enum):
    PERFECT = auto()
    UNIFORM = auto()
    NOT_UNIFORM = auto()

    def description(self):
        if self == Uniformity.PERFECT:
            return 'perfectly uniform'
        elif self == Uniformity.UNIFORM:
            return 'uniform to within 1%'
        elif self == Uniformity.NOT_UNIFORM:
            return 'not uniform'


class TimeUnit(float, Enum):
    millisecond = 1e-3
    second = 1
    minute = 60 * second
    hour = 60 * minute
    day = 24 * hour
    week = 7 * day
    year = 365 * day
    month = year / 12
    decade = 10 * year + 2 * day
    century = 100 * year + 24 * day
    millennium = 1000 * year + 242 * day


@dataclass
class TimeResolution:
    uniformity: Uniformity
    unit: TimeUnit
    density: float
    error: float


class Classification(BaseModel):
    """
        Classification is the classifciation information for one column.
    """
    column: str = Field(default=None, description='column name')
    category: Optional[Category]
    subcategory: Optional[Subcategory]
    format: str = Field(default=None, description='the date represented in strftime format')
    time_resolution: Optional[TimeResolution]
    match_type: List[Matchtype]
    Parser: Optional[Parser]
    fuzzyColumn: Optional[FuzzyColumn]


class Classifications(BaseModel):
    """
        Classifications are a list of Classification objects. This is what is returned from geotime_classify.
    """
    classifications: List[Classification]