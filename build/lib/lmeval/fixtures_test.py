import pytest
from .fixtures import eu_countries, get_country_boolean, get_country_multi_choice, gemini

def test_mock_gemini_fixture(gemini):

    # check our model is working
    mock_answers = {'Say hello in french!': 'bonjour'}
    gemini.set_request_response(mock_answers)
    answer = gemini.generate_text('Say hello in french!')

    assert 'bonjour' in answer.answer.lower()
    assert not answer.iserror
    assert not answer.error_reason
    assert not answer.ispunting
    assert not answer.punting_reason
    assert answer.steps[0].execution_time > 0.1

    assert answer.score == 0.0  # no scoring yet
    assert 'gemini' in answer.model.name.lower()
    assert 'gemini' in answer.model.version_string.lower()


def test_get_country_boolean():
    result = get_country_boolean()
    assert isinstance(result, dict)
    assert 'question' in result
    assert 'answer' in result
    assert isinstance(result['question'], str)
    assert isinstance(result['answer'], str)


def test_get_country_multi_choice():
    # Test with default number of choices (4)
    result = get_country_multi_choice()
    assert isinstance(result, dict)
    assert 'question' in result
    assert 'choices' in result
    assert 'answer' in result
    assert isinstance(result['question'], str)
    assert isinstance(result['choices'], list)
    assert len(result['choices']) == 3
    assert result['answer'] not in result['choices']

    # Test with different num choices
    for num_choices in [2, 3, 4, 5]:
        result = get_country_multi_choice(num_choices)
        assert len(result['choices']) == num_choices - 1


def test_get_country_multi_choice_errors():
    # Test for ValueError when num_choices is less than 2
    with pytest.raises(ValueError):
        get_country_multi_choice(1)

    # Test for ValueError when num_choices is greater than number of countries
    with pytest.raises(ValueError):
        get_country_multi_choice(len(eu_countries) + 1)

def test_country_capital_pairs():
    # Test that all country-capital pairs are unique
    assert len(eu_countries) == len(set(eu_countries))

    # Test that all countries and capitals are strings
    for country, capital in eu_countries:
        assert isinstance(country, str)
        assert isinstance(capital, str)

def test_multiple_runs():
    # Test that multiple runs of get_country_boolean produce different results
    boolean_results = set(get_country_boolean()['question'] for _ in range(10))
    assert len(boolean_results) > 1, "get_country_boolean() is not producing varied results"

    # Test that multiple runs of get_country_multi_choice produce different results
    multi_choice_results = set(get_country_multi_choice()['question'] for _ in range(10))
    assert len(multi_choice_results) > 1, "get_country_multi_choice() is not producing varied results"