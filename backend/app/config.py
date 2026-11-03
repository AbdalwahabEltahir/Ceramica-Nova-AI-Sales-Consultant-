"""Business-rule values and tunables. Change a rule here, not in code (NFR-10)."""

WASTAGE_SIMPLE_PERCENT = 10
WASTAGE_COMPLEX_PERCENT = 15

DISCOUNT_APPROVAL_THRESHOLD_PERCENT = 12

MAX_PRIMARY_PRODUCTS = 3
MAX_CLARIFYING_QUESTIONS = 2
MAX_TOOL_CALLS_PER_TURN = 6

OUTDOOR_MIN_SLIP = 'R11'
INDOOR_HIGH_SLIP_MIN = 'R10'  # assumption A-3

RANKING_WEIGHTS = {
    'style_match': 0.35,
    'price_fit': 0.30,
    'colour_match': 0.15,
    'retrieval_relevance': 0.10,
    'stock_margin': 0.10,
    'finish_match': 0.15,  # bonus on top of the base weights when a stated finish matches
}

SQFT_TO_M2 = 0.092903

SESSION_COOKIE_NAME = 'cn_session_id'

SCARCITY_PHRASES = [
    'only a few left', 'selling fast', 'last chance', 'hurry', 'almost gone',
]

AVAILABILITY_WORDS = {
    'AVAILABLE': ['in stock for your quantity', 'in stock', 'available', 'متوفر'],
    'PARTIALLY_AVAILABLE': ['partially available', 'limited stock', 'جزئيا متوفر'],
    'OUT_OF_STOCK': ['out of stock', 'غير متوفر'],
}

FORBIDDEN_DELIVERY_PHRASES = ['available today', 'delivered by', 'متوفر اليوم']
