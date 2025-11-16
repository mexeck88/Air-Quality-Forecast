"""
Health Recommendations Module

Provides personalized health recommendations based on AQI values,
user age, and respiratory condition status.
"""

import numpy as np


def get_aqi_level(aqi_value):
    """
    Get AQI level category based on value
    
    Args:
        aqi_value (float): AQI value
        
    Returns:
        str: AQI level category
    """
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_health_recommendation(aqi_value, age=None, respiratory_issues='None'):
    """
    Generate personalized health recommendations based on AQI and user profile
    
    Args:
        aqi_value (float): AQI value for tomorrow
        age (int, optional): User age in years
        respiratory_issues (str): 'None', 'Mild', 'Moderate', or 'Severe'
        
    Returns:
        str: Formatted health recommendation text
    """
    
    aqi_level = get_aqi_level(aqi_value)
    
    recommendations = {
        "Good": {
            "level": "AIR QUALITY: GOOD",
            "general": "Air quality is satisfactory. No health risks.",
            "action": "Enjoy outdoor activities! This is an excellent day.",
        },
        "Moderate": {
            "level": "AIR QUALITY: MODERATE",
            "general": "Acceptable air quality, but sensitive groups may be affected.",
            "action": "Outdoor activities okay for most people",
        },
        "Unhealthy for Sensitive Groups": {
            "level": "AIR QUALITY: UNHEALTHY FOR SENSITIVE GROUPS",
            "general": "Sensitive groups may experience health effects.",
            "action": "Sensitive groups should limit outdoor time",
        },
        "Unhealthy": {
            "level": "AIR QUALITY: UNHEALTHY",
            "general": "Everyone may begin to experience health effects.",
            "action": "Sensitive groups avoid outdoors; others limit exertion",
        },
        "Very Unhealthy": {
            "level": "AIR QUALITY: VERY UNHEALTHY",
            "general": "Health alert: serious effects likely for sensitive groups.",
            "action": "Everyone should avoid outdoor activities",
        },
        "Hazardous": {
            "level": "AIR QUALITY: HAZARDOUS",
            "general": "Health emergency conditions.",
            "action": "STAY INDOORS - Critical situation",
        },
    }
    
    base_rec = recommendations.get(aqi_level, recommendations["Moderate"])
    
    recommendation_text = f"""
{'='*50}
{base_rec['level']}
{'='*50}

AQI VALUE: {aqi_value:.1f}
{base_rec['general']}

RECOMMENDATION: {base_rec['action']}
"""
    
    if age:
        recommendation_text += f"\nAge: {age} years"
        if age < 6:
            recommendation_text += "\n   Young children are sensitive - limit outdoor play"
        elif age > 65:
            recommendation_text += "\n   Seniors are susceptible - monitor carefully"
    
    if respiratory_issues != 'None':
        recommendation_text += f"\nRespiratory Issues: {respiratory_issues}"
        if respiratory_issues == 'Mild':
            recommendation_text += "\n   Keep inhalers accessible"
        elif respiratory_issues == 'Moderate':
            recommendation_text += "\n   Avoid outdoor exposure on high AQI days"
        elif respiratory_issues == 'Severe':
            recommendation_text += "\n   Stay indoors and monitor closely"
    
    recommendation_text += f"\n\n{'='*50}\n"
    
    return recommendation_text.strip()


def get_activity_recommendation(aqi_value, activity_type='outdoor'):
    """
    Get specific activity recommendations based on AQI
    
    Args:
        aqi_value (float): AQI value
        activity_type (str): 'outdoor', 'exercise', 'children'
        
    Returns:
        str: Activity-specific recommendation
    """
    
    aqi_level = get_aqi_level(aqi_value)
    
    activities = {
        'outdoor': {
            'Good': 'Perfect day for outdoor activities',
            'Moderate': 'Can enjoy outdoor activities',
            'Unhealthy for Sensitive Groups': 'Sensitive groups should limit time',
            'Unhealthy': 'Avoid prolonged outdoor exposure',
            'Very Unhealthy': 'Avoid outdoor activities',
            'Hazardous': 'Stay indoors',
        },
        'exercise': {
            'Good': 'All outdoor exercise is fine',
            'Moderate': 'Moderate exercise is safe',
            'Unhealthy for Sensitive Groups': 'Reduce intensity for sensitive groups',
            'Unhealthy': 'Move exercise indoors',
            'Very Unhealthy': 'Exercise indoors only',
            'Hazardous': 'Indoor exercise only',
        },
        'children': {
            'Good': 'Safe for all outdoor play',
            'Moderate': 'Children can play outdoors',
            'Unhealthy for Sensitive Groups': 'Limit duration and intensity',
            'Unhealthy': 'Limit outdoor play time',
            'Very Unhealthy': 'Keep indoors',
            'Hazardous': 'Indoor activities only',
        },
    }
    
    return activities.get(activity_type, {}).get(aqi_level, 'Check current conditions')