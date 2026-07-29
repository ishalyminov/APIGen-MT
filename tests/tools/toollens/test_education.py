import pytest
import json
from tools.toollens.education import EducationTools


@pytest.fixture
def education_instance():
    config = {
        'random_treasures': [
            {'id': 't1', 'name': 'Gold Coin', 'value': 500},
            {'id': 't2', 'name': 'Silver Chalice', 'value': 300}
        ],
        'topics': {
            'math': [{'title': 'Algebra Basics', 'content': 'Linear equations and variables'}],
            'science': [{'title': 'Photosynthesis', 'content': 'Process of converting light to energy'}],
            'history': [{'title': 'World War II', 'content': '1939-1945 global conflict'}]
        },
        'ai_questions': [{'q': 'What is gravity?', 'a': 'Force attracting objects with mass'}],
        'names': ['Alice', 'Bob', 'Charlie'],
        'chemical_elements': [
            {'symbol': 'H', 'name': 'Hydrogen', 'number': 1},
            {'symbol': 'O', 'name': 'Oxygen', 'number': 8},
            {'symbol': 'Au', 'name': 'Gold', 'number': 79}
        ],
        'physical_constants': [
            {'name': 'Speed of Light', 'value': '299792458', 'unit': 'm/s'},
            {'name': 'Planck Constant', 'value': '6.626e-34', 'unit': 'J*s'}
        ],
        'space_news': [
            {'title': 'Mars Rover Discovery', 'date': '2024-01-15', 'summary': 'New evidence of water'}
        ],
        'random_words': ['serendipity', 'ephemeral', 'quintessential'],
        'date_facts': {
            '01-15': 'Martin Luther King Jr. born in 1929',
            '07-04': 'US Declaration of Independence adopted in 1776'
        },
        'equations': [
            {'name': "Einstein's Mass-Energy", 'formula': 'E=mc^2'},
            {'name': "Newton's Second Law", 'formula': 'F=ma'}
        ],
        'numbers_translations': {'42': 'forty-two', '100': 'one hundred'},
        'planets': [
            {'name': 'Mercury', 'star': 'Sun'},
            {'name': 'Venus', 'star': 'Sun'},
            {'name': 'Earth', 'star': 'Sun'}
        ],
        'quiz_today': [{'question': 'Capital of France?', 'answer': 'Paris'}],
        'recent_words_dc': [{'word': 'petrichor', 'definition': 'Earthy scent after rain'}],
        'word_of_day_ld': {'word': 'ubiquitous', 'definition': 'Present everywhere'},
        'word_of_day_mw': {'word': 'candid', 'definition': 'Honest, frank'},
        'word_of_day_pm': {'word': 'nostalgia', 'definition': 'Sentimental longing'},
        'dashboard': {'stats': {'users': 1250, 'courses': 45}},
        'questions_answers': [
            {'type': 'multiple_choice', 'question': '2+2?', 'answer': '4', 'feedback': 'Basic addition'}
        ],
        'periodic_table': [{'element': 'Carbon', 'symbol': 'C', 'mass': 12.011}],
        'quotes': [{'text': 'Knowledge is power', 'author': 'Francis Bacon'}],
        'turkish_words': [{'word': 'kitap', 'meaning': 'book'}, {'word': 'su', 'meaning': 'water'}],
        'empty_collections': [],
        'sparse_data': {}
    }
    return EducationTools(initial_config=config)


@pytest.fixture
def empty_education_instance():
    return EducationTools(initial_config=None)


def test_random_returns_dict(education_instance):
    """Test that random returns a dict with treasure data."""
    result = education_instance.random()
    assert isinstance(result, dict)
    assert 'id' in result or 'name' in result or 'value' in result


def test_random_with_empty_config(empty_education_instance):
    """Test random method with no initial config."""
    result = empty_education_instance.random()
    assert isinstance(result, dict)


def test_topic_topic_valid(education_instance):
    """Test topic_topic with a valid topic."""
    result = education_instance.topic_topic('math')
    assert isinstance(result, dict)
    assert 'topic' in result or 'entries' in result or 'data' in result


def test_topic_topic_invalid(education_instance):
    """Test topic_topic with an invalid topic returns error info."""
    result = education_instance.topic_topic('nonexistent')
    assert isinstance(result, dict)


def test_ask_question_to_ai_service_valid(education_instance):
    """Test Ask_question_to_AI_Service with a valid question."""
    result = education_instance.Ask_question_to_AI_Service('What is gravity?')
    assert isinstance(result, dict)
    assert 'answer' in result or 'response' in result or 'a' in result


def test_ask_question_to_ai_service_none(education_instance):
    """Test Ask_question_to_AI_Service with None question."""
    result = education_instance.Ask_question_to_AI_Service(None)
    assert isinstance(result, dict)


def test_get_returns_list(education_instance):
    """Test Get returns a list of names."""
    result = education_instance.Get()
    assert isinstance(result, list)


def test_get_with_empty_config(empty_education_instance):
    """Test Get with no initial config."""
    result = empty_education_instance.Get()
    assert isinstance(result, list)


def test_get_all_chemical_elements_returns_dict(education_instance):
    """Test Get_All_Chemical_Elements returns a dict."""
    result = education_instance.Get_All_Chemical_Elements()
    assert isinstance(result, dict)


def test_get_all_chemical_elements_empty(empty_education_instance):
    """Test Get_All_Chemical_Elements with no config."""
    result = empty_education_instance.Get_All_Chemical_Elements()
    assert isinstance(result, dict)


def test_get_all_constants_returns_dict(education_instance):
    """Test Get_All_constants returns a dict."""
    result = education_instance.Get_All_constants()
    assert isinstance(result, dict)


def test_get_all_constants_empty(empty_education_instance):
    """Test Get_All_constants with no config."""
    result = empty_education_instance.Get_All_constants()
    assert isinstance(result, dict)


def test_get_space_news_returns_dict(education_instance):
    """Test Get_Space_News returns a dict."""
    result = education_instance.Get_Space_News()
    assert isinstance(result, dict)


def test_get_space_news_empty(empty_education_instance):
    """Test Get_Space_News with no config."""
    result = empty_education_instance.Get_Space_News()
    assert isinstance(result, dict)


def test_get_a_random_word_returns_dict(education_instance):
    """Test Get_a_random_word returns a dict."""
    result = education_instance.Get_a_random_word()
    assert isinstance(result, dict)
    assert 'word' in result


def test_get_a_random_word_empty(empty_education_instance):
    """Test Get_a_random_word with no config."""
    result = empty_education_instance.Get_a_random_word()
    assert isinstance(result, dict)


def test_get_date_fact_valid(education_instance):
    """Test Get_date_fact with valid month and day."""
    result = education_instance.Get_date_fact('01', '15')
    assert isinstance(result, dict)


def test_get_date_fact_none(education_instance):
    """Test Get_date_fact with None params."""
    result = education_instance.Get_date_fact(None, None)
    assert isinstance(result, dict)


def test_get_equations_returns_dict(education_instance):
    """Test Get_equations returns a dict."""
    result = education_instance.Get_equations()
    assert isinstance(result, dict)


def test_get_equations_empty(empty_education_instance):
    """Test Get_equations with no config."""
    result = empty_education_instance.Get_equations()
    assert isinstance(result, dict)


def test_get_multiple_random_words_valid(education_instance):
    """Test Get_multiple_random_words with valid count."""
    result = education_instance.Get_multiple_random_words(3)
    assert isinstance(result, dict)


def test_get_multiple_random_words_none(education_instance):
    """Test Get_multiple_random_words with None count."""
    result = education_instance.Get_multiple_random_words(None)
    assert isinstance(result, dict)


def test_get_random_fact_valid(education_instance):
    """Test Get_random_fact with valid type."""
    result = education_instance.Get_random_fact('trivia')
    assert isinstance(result, dict)


def test_get_random_fact_none(education_instance):
    """Test Get_random_fact with None type."""
    result = education_instance.Get_random_fact(None)
    assert isinstance(result, dict)


def test_numbers_translator_returns_dict(education_instance):
    """Test Numbers_Translator returns a dict."""
    result = education_instance.Numbers_Translator()
    assert isinstance(result, dict)


def test_numbers_translator_empty(empty_education_instance):
    """Test Numbers_Translator with no config."""
    result = empty_education_instance.Numbers_Translator()
    assert isinstance(result, dict)


def test_planet_list_returns_dict(education_instance):
    """Test Planet_list returns a dict."""
    result = education_instance.Planet_list()
    assert isinstance(result, dict)


def test_planet_list_empty(empty_education_instance):
    """Test Planet_list with no config."""
    result = empty_education_instance.Planet_list()
    assert isinstance(result, dict)


def test_quiz_for_today_returns_dict(education_instance):
    """Test Quiz_For_Today returns a dict."""
    result = education_instance.Quiz_For_Today()
    assert isinstance(result, dict)


def test_quiz_for_today_empty(empty_education_instance):
    """Test Quiz_For_Today with no config."""
    result = empty_education_instance.Quiz_For_Today()
    assert isinstance(result, dict)


def test_recent_word_of_the_day_from_dc_returns_dict(education_instance):
    """Test Recent_word_of_the_day_from_DC returns a dict."""
    result = education_instance.Recent_word_of_the_day_from_DC()
    assert isinstance(result, dict)


def test_recent_word_of_the_day_from_dc_empty(empty_education_instance):
    """Test Recent_word_of_the_day_from_DC with no config."""
    result = empty_education_instance.Recent_word_of_the_day_from_DC()
    assert isinstance(result, dict)


def test_today_s_international_current_affairs_returns_dict(education_instance):
    """Test Today_s_International_Current_Affairs returns a dict."""
    result = education_instance.Today_s_International_Current_Affairs()
    assert isinstance(result, dict)


def test_today_s_international_current_affairs_empty(empty_education_instance):
    """Test Today_s_International_Current_Affairs with no config."""
    result = empty_education_instance.Today_s_International_Current_Affairs()
    assert isinstance(result, dict)


def test_word_of_the_day_from_ld_returns_dict(education_instance):
    """Test Word_of_the_day_from_ld returns a dict."""
    result = education_instance.Word_of_the_day_from_ld()
    assert isinstance(result, dict)


def test_word_of_the_day_from_ld_empty(empty_education_instance):
    """Test Word_of_the_day_from_ld with no config."""
    result = empty_education_instance.Word_of_the_day_from_ld()
    assert isinstance(result, dict)


def test_word_of_the_day_from_mw_returns_dict(education_instance):
    """Test Word_of_the_day_from_mw returns a dict."""
    result = education_instance.Word_of_the_day_from_mw()
    assert isinstance(result, dict)


def test_word_of_the_day_from_mw_empty(empty_education_instance):
    """Test Word_of_the_day_from_mw with no config."""
    result = empty_education_instance.Word_of_the_day_from_mw()
    assert isinstance(result, dict)


def test_word_of_the_day_from_pm_returns_dict(education_instance):
    """Test Word_of_the_day_from_pm returns a dict."""
    result = education_instance.Word_of_the_day_from_pm()
    assert isinstance(result, dict)


def test_word_of_the_day_from_pm_empty(empty_education_instance):
    """Test Word_of_the_day_from_pm with no config."""
    result = empty_education_instance.Word_of_the_day_from_pm()
    assert isinstance(result, dict)


def test_dashboard_returns_dict(education_instance):
    """Test dashboard returns a dict."""
    result = education_instance.dashboard()
    assert isinstance(result, dict)


def test_dashboard_empty(empty_education_instance):
    """Test dashboard with no config."""
    result = empty_education_instance.dashboard()
    assert isinstance(result, dict)


def test_get_questions_and_answers_returns_dict(education_instance):
    """Test getQuestionsandAnswers returns a dict."""
    result = education_instance.getQuestionsandAnswers()
    assert isinstance(result, dict)


def test_get_questions_and_answers_empty(empty_education_instance):
    """Test getQuestionsandAnswers with no config."""
    result = empty_education_instance.getQuestionsandAnswers()
    assert isinstance(result, dict)


def test_periodic_table_returns_dict(education_instance):
    """Test periodic_table returns a dict."""
    result = education_instance.periodic_table()
    assert isinstance(result, dict)


def test_periodic_table_empty(empty_education_instance):
    """Test periodic_table with no config."""
    result = empty_education_instance.periodic_table()
    assert isinstance(result, dict)


def test_quote_returns_dict(education_instance):
    """Test quote returns a dict."""
    result = education_instance.quote()
    assert isinstance(result, dict)


def test_quote_empty(empty_education_instance):
    """Test quote with no config."""
    result = empty_education_instance.quote()
    assert isinstance(result, dict)


def test_word_search_turkish_valid(education_instance):
    """Test wordSearchTurkish with a valid query."""
    result = education_instance.wordSearchTurkish('kitap')
    assert isinstance(result, dict)


def test_word_search_turkish_none(education_instance):
    """Test wordSearchTurkish with None query."""
    result = education_instance.wordSearchTurkish(None)
    assert isinstance(result, dict)