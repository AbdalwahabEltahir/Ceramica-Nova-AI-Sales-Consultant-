"""CLI: builds the RAG index and reports what was indexed vs. quarantined.
Demonstrates the exam C3 scenario live: running this with the malicious
fixture included shows the record gets quarantined, not indexed.

Usage:
  python -m rag.ingest              # normal corpus only
  python -m rag.ingest --with-fixture   # include the C3 malicious record
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from rag import lancedb_store  # noqa: E402
from rag.index import build_index  # noqa: E402
from rag.embeddings_client import is_configured, model_id  # noqa: E402


def main():
    include_fixture = '--with-fixture' in sys.argv
    fixture = None
    if include_fixture:
        fixture_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'data', 'injection_fixtures.json')
        with open(fixture_path, 'r', encoding='utf-8') as fh:
            fixture = json.load(fh)

    index = build_index(include_fixture=fixture)

    print(f'Embeddings configured: {is_configured()} ({model_id()})')
    print(f'Indexed documents: {len(index.documents)}')
    print(f'LanceDB rows ({lancedb_store.DB_PATH}): {lancedb_store.row_count()}')
    print(f'Quarantined documents: {len(index.quarantined)}')
    for doc in index.quarantined:
        print(f"  BLOCKED {doc['id']}: matched pattern {doc['quarantine_reason']!r}")
        print(f"    text: {doc['text'][:100]}...")


if __name__ == '__main__':
    main()
