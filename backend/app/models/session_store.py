"""In-memory session repository. Swap for Redis/DB without touching callers."""

import secrets
import threading
from datetime import datetime, timezone


class SessionRecord:
    def __init__(self, session_id):
        self.session_id = session_id
        self.requirements = {}
        self.stage = 'discovery'  # discovery -> recommendation -> quantity -> quote -> handoff
        self.history = []
        self.tool_results = {}
        self.last_recommendations = []  # SKUs shown last turn — lets a short
        # follow-up like "are they safe?" resolve "they" without re-stating names
        self.shown_skus = []  # everything shown for the current requirements, so
        # "other options" moves on to new products instead of repeating
        self.pending_quote = None  # {'sku', 'discount_percent'} while waiting for a missing detail
        self.created_at = datetime.now(timezone.utc).isoformat()


class SessionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions = {}

    def new_id(self):
        return secrets.token_urlsafe(24)

    def get_or_create(self, session_id):
        with self._lock:
            if not session_id or session_id not in self._sessions:
                session_id = self.new_id()
                self._sessions[session_id] = SessionRecord(session_id)
            return self._sessions[session_id]

    def get(self, session_id):
        with self._lock:
            return self._sessions.get(session_id)

    def save(self, record):
        with self._lock:
            self._sessions[record.session_id] = record


class QuoteStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._quotes = {}

    def new_id(self):
        return 'QN-' + secrets.token_hex(6).upper()

    def save(self, quote):
        with self._lock:
            self._quotes[quote['quotation_id']] = quote

    def get(self, quotation_id):
        with self._lock:
            return self._quotes.get(quotation_id)

    def get_for_owner(self, quotation_id, owner_id):
        quote = self.get(quotation_id)
        if not quote or quote.get('owner_id') != owner_id:
            return None
        return quote


session_store = SessionStore()
quote_store = QuoteStore()
