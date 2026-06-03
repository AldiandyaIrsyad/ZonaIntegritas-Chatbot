"""Tests for NLI infrastructure module."""

import httpx
import pytest
import respx

from app.infra.nli import NLIProvider, LABEL_ENTAILMENT, LABEL_NEUTRAL, LABEL_CONTRADICTION

@pytest.fixture
def provider():
    """Fixture providing an NLIProvider instance."""
    return NLIProvider(
        base_url="http://infinity:7997",
        model="StevenLimcorn/indo-roberta-indonli"
    )

@respx.mock
async def test_check_raw_scores_true(provider):
    """Test parsing when raw_scores=True (list of dicts)."""
    respx.post("http://infinity:7997/classify").respond(
        json={
            "data": [
                [
                    {"label": "entailment", "score": 0.8},
                    {"label": "neutral", "score": 0.1},
                    {"label": "contradiction", "score": 0.1}
                ]
            ]
        }
    )
    
    result = await provider.check("premise", "hypothesis")
    
    assert result.label == LABEL_ENTAILMENT
    assert result.entailment_score == 0.8
    assert result.neutral_score == 0.1
    assert result.contradiction_score == 0.1

@respx.mock
async def test_check_fallback_dict(provider):
    """Test parsing when fallback returns a single dict with 'label' and 'score' float."""
    respx.post("http://infinity:7997/classify").respond(
        json={
            "data": [
                {"label": "contradiction", "score": 0.9}
            ]
        }
    )
    
    result = await provider.check("premise", "hypothesis")
    
    assert result.label == LABEL_CONTRADICTION
    assert result.contradiction_score == 0.9
    assert result.entailment_score == 0.0
    assert result.neutral_score == 0.0

@respx.mock
async def test_check_fallback_dict_with_score_dict(provider):
    """Test parsing when fallback returns a dict where score is a dict."""
    respx.post("http://infinity:7997/classify").respond(
        json={
            "data": [
                {
                    "score": {
                        "entailment": 0.2,
                        "neutral": 0.7,
                        "contradiction": 0.1
                    }
                }
            ]
        }
    )
    
    result = await provider.check("premise", "hypothesis")
    
    assert result.label == LABEL_NEUTRAL
    assert result.neutral_score == 0.7

@respx.mock
async def test_check_empty_response(provider):
    """Test handling of empty response."""
    respx.post("http://infinity:7997/classify").respond(json={"data": []})
    
    result = await provider.check("premise", "hypothesis")
    
    assert result.label == LABEL_NEUTRAL
    assert result.neutral_score == 0.5
    assert result.entailment_score == 0.5

@respx.mock
async def test_check_network_error(provider):
    """Test that network errors return a neutral fallback result instead of raising."""
    respx.post("http://infinity:7997/classify").mock(side_effect=httpx.ConnectError("Connection refused"))
    
    result = await provider.check("premise", "hypothesis")
    
    assert result.label == LABEL_NEUTRAL
    assert result.neutral_score == 0.5

async def test_close(provider):
    """Test closing the client."""
    await provider.close()
    assert provider._client.is_closed
