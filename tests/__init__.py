"""Tests must never depend on live network access to Gemini/HF — it makes
the suite slow, flaky, and burns real API quota every run (discovered the
hard way). See the top of each test_*.py for the actual enforcement — this
file alone isn't sufficient because `python -m unittest discover -s tests`
(no `-t`) doesn't reliably import this package before test modules do.
"""
