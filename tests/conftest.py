"""
pytest configuration for AAOS test suite
"""

import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for all async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "asyncio: mark test as async")
