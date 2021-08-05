from enum import Enum, IntEnum
from pydantic import BaseModel, constr, Field
from typing import List, Optional

class fuzzyCategory(str,Enum):
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

class category(str, Enum):
    """
    category is the general classification for a column
    """
    geo= "geo"
    time="time"
    boolean="boolean"
    timeout="timeout"


class subcategory(str, Enum):
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

class matchtype(str, Enum):
    """
     is the type of match for classification if any.
    """
    fuzzy="fuzzy"
    LSTM="LSTM"

class fuzzyColumn(BaseModel):
    """
       fuzzyColumn is only defined when a column header matches a word we are looking for. fuzzyCategory is used for classifying a column.
    """
    matchedKey: str = Field(default=None, description='This is the word that was matched with the column header. If a column header was Lat, it would match with the the matchedKey of Lat, since it is one of the lookup words. In this case the fuzzyCategory would be returned as "Latitude".')
    fuzzyCategory: Optional[fuzzyCategory]
    ratio: int = Field(default=None, description='Ratio of the fuzzy match. If it was an exact match it would be 100')

class Parser(str,Enum):
    """
        Parser records which python library the date was parsed with. dateutil or arrow.
    """
    Util="Util"
    arrow="arrow"

class Classification(BaseModel):
    """
        Classification is the classifciation information for one column.
    """
    column: str = Field(default=None, description='column name')
    category: Optional[category]
    subcategory: Optional[subcategory]
    format: str = Field(default=None, description='the date represented in strftime format')
    match_type: List[matchtype]
    Parser: Optional[Parser]
    DayFirst: bool = Field(default=None, description='Boolean: if day is first in date format' )
    fuzzyColumn: Optional[fuzzyColumn]


class Classifications(BaseModel):
    """
        Classifications are a list of Classification objects. This is what is returned from geotime_classify.
    """
    classifications: List[Classification]