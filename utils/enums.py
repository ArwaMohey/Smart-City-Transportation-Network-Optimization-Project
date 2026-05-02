"""Enumerations for the Smart City Transportation Network System.

All domain-level symbolic constants are defined here so that other modules
import from a single, authoritative source and avoid magic strings.
"""

from enum import Enum


class NodeType(Enum):
    """Semantic category of a node in the transportation network.

    Values
    ------
    NEIGHBORHOOD  : Residential area.
    HOSPITAL      : Medical facility.
    FIRE_STATION  : Emergency fire-fighting station.
    POLICE        : Police station or headquarters.
    SCHOOL        : Educational institution.
    COMMERCIAL    : Shopping or business district.
    INDUSTRIAL    : Manufacturing or logistics zone.
    PARK          : Public recreational area.
    TRANSIT_HUB   : Major public-transport interchange.
    AIRPORT       : Airport or air-transport hub.
    BUSINESS      : Business district or technology park.
    TOURISM       : Tourist or heritage destination.
    SPORTS        : Sports venue or stadium.
    """

    NEIGHBORHOOD = "neighborhood"
    HOSPITAL = "hospital"
    FIRE_STATION = "fire_station"
    POLICE = "police"
    SCHOOL = "school"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    PARK = "park"
    TRANSIT_HUB = "transit_hub"
    AIRPORT = "airport"
    BUSINESS = "business"
    TOURISM = "tourism"
    SPORTS = "sports"


class RoadCondition(Enum):
    """Physical state of a road segment.

    Used to derive a penalty multiplier that increases the effective edge
    weight beyond the raw distance × traffic-factor product.

    Values
    ------
    EXCELLENT : Freshly paved; no penalty.
    GOOD      : Minor wear; small penalty.
    FAIR      : Noticeable deterioration; moderate penalty.
    POOR      : Severe damage; heavy penalty.
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class TimeOfDay(Enum):
    """Discrete time-of-day windows used for dynamic traffic modelling.

    Values
    ------
    MORNING   : Rush-hour, typically 06:00–10:00.
    AFTERNOON : Mid-day, typically 10:00–16:00.
    EVENING   : Evening rush, typically 16:00–20:00.
    NIGHT     : Off-peak, typically 20:00–06:00.
    """

    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


class TransportMode(Enum):
    """Modes of public transport supported by the network.

    Values
    ------
    METRO : Underground or surface rapid transit.
    BUS   : Standard city bus service.
    TRAM  : Light-rail street tram.
    """

    METRO = "metro"
    BUS = "bus"
    TRAM = "tram"
