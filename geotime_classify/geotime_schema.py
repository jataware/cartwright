
from enum import Enum, IntEnum
from pydantic import BaseModel, constr, Field
from typing import List, Optional, Literal

class fuzzyCategory(str,Enum):
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
    ISO2: "ISO2"
    ISO3 = "ISO3"
    ISO_code= "ISO_code"
    Results= "Results"

class category(str, Enum):
    geo= "geo"
    time="time"
    boolean="boolean"
    unknown_date = "unknown_date"

class subcategory(str, Enum):
    city_name="city_name"
    state_name="state_name"
    country_name="country_name"
    ISO3="ISO3"
    ISO2="ISO2"
    continent="continent"
    longitude="longitude"
    latitude="latitude"
    date="date"

class fuzzyColumn(BaseModel):
    matchedKey: str = Field(default=None)
    fuzzyCategory: Optional[fuzzyCategory]
    ratio: int = Field(default=None)

class match_type(str, Enum):
    LSTM="LSTM"
    fuzzy="fuzzy"

class Parser(str,Enum):
    Util="Util"
    arrow="arrow"



class Classification(BaseModel):
    column: str = Field(default=None)
    category: Optional[category]
    subcategory: Optional[subcategory]
    format: str = None
    match_type: List[Literal["LSTM", "fuzzy"]]
    Parser: Optional[Parser]
    DayFirst: bool = None
    fuzzyColumn: Optional[fuzzyColumn]


class Classifications(BaseModel):
    classifications: List[Classification]


