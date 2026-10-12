FALLBACK_INR_PER_UNIT = {
    "INR": 1.0,
    "USD": 83.0,
    "EUR": 90.5,
    "GBP": 105.2,
    "AED": 22.6,
    "SGD": 62.4,
    "CAD": 61.2,
    "AUD": 54.8,
    "JPY": 0.55,
    "SAR": 22.1,
    "KWD": 270.5,
    "CHF": 94.1,
    "PEN": 22.3,
    "CNY": 11.5,
    "MYR": 18.2,
    "THB": 2.3,
    "QAR": 22.8,
    "OMR": 215.6,
    "BHD": 220.2
}

RAW_CURRENCIES = [
    ("India", "INR", "Indian Rupee"),
    ("United States", "USD", "US Dollar"),
    ("Eurozone", "EUR", "Euro"),
    ("United Kingdom", "GBP", "British Pound"),
    ("United Arab Emirates", "AED", "UAE Dirham"),
    ("Singapore", "SGD", "Singapore Dollar"),
    ("Canada", "CAD", "Canadian Dollar"),
    ("Australia", "AUD", "Australian Dollar"),
    ("Japan", "JPY", "Japanese Yen"),
    ("Saudi Arabia", "SAR", "Saudi Riyal"),
    ("Kuwait", "KWD", "Kuwaiti Dinar"),
    ("Switzerland", "CHF", "Swiss Franc"),
    ("Peru", "PEN", "Peruvian Sol"),
    ("China", "CNY", "Chinese Yuan"),
    ("Malaysia", "MYR", "Malaysian Ringgit"),
    ("Thailand", "THB", "Thai Baht"),
    ("Qatar", "QAR", "Qatari Riyal"),
    ("Oman", "OMR", "Omani Rial"),
    ("Bahrain", "BHD", "Bahraini Dinar")
]

COUNTRY_CURRENCIES = {
    country: {
        "code": code,
        "name": name,
        "rate_to_inr": FALLBACK_INR_PER_UNIT.get(code, 1.0)
    }
    for country, code, name in RAW_CURRENCIES
}

def country_label(country_tuple_or_name):
    if isinstance(country_tuple_or_name, (list, tuple)) and len(country_tuple_or_name) >= 3:
        return f"{country_tuple_or_name[0]} ({country_tuple_or_name[1]} - {country_tuple_or_name[2]})"
    c_str = str(country_tuple_or_name)
    info = COUNTRY_CURRENCIES.get(c_str, {})
    if info:
        return f"{c_str} ({info.get('code', '')} - {info.get('name', '')})"
    return c_str

def _fetch_inr_rates():
    from datetime import datetime
    return FALLBACK_INR_PER_UNIT, datetime.now().strftime("%d %b %Y, %I:%M %p")