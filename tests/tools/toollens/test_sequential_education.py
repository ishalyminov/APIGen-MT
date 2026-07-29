import pytest
import json
from tools.toollens.education import EducationTools


@pytest.fixture
def education_config() -> dict:
    """Fixture providing a complete initial config for EducationTools."""
    return json.loads(json.dumps({
        "random_treasures": [
            {"id": "t1", "name": "Gold Coin", "value": 500},
            {"id": "t2", "name": "Silver Chalice", "value": 300}
        ],
        "topics": {
            "math": [
                {"title": "Algebra Basics", "content": "Linear equations and variables"}
            ],
            "science": [
                {"title": "Photosynthesis", "content": "Process of converting light to energy"}
            ],
            "history": [
                {"title": "World War II", "content": "1939-1945 global conflict"}
            ]
        },
        "ai_questions": [
            {"q": "What is gravity?", "a": "Force attracting objects with mass"}
        ],
        "names": ["Alice", "Bob", "Charlie"],
        "chemical_elements": [
            {"symbol": "H", "name": "Hydrogen", "number": 1},
            {"symbol": "O", "name": "Oxygen", "number": 8},
            {"symbol": "Au", "name": "Gold", "number": 79}
        ],
        "physical_constants": [
            {"name": "Speed of Light", "value": "299792458", "unit": "m/s"},
            {"name": "Planck Constant", "value": "6.626e-34", "unit": "J*s"}
        ],
        "space_news": [
            {"title": "Mars Rover Discovery", "date": "2024-01-15", "summary": "New evidence of water"}
        ],
        "random_words": ["serendipity", "ephemeral", "quintessential"],
        "date_facts": {
            "01-15": "Martin Luther King Jr. born in 1929",
            "07-04": "US Declaration of Independence adopted in 1776"
        },
        "equations": [
            {"name": "Einstein's Mass-Energy", "formula": "E=mc^2"},
            {"name": "Newton's Second Law", "formula": "F=ma"}
        ],
        "numbers_translations": {
            "42": "forty-two",
            "100": "one hundred"
        },
        "planets": [
            {"name": "Mercury", "star": "Sun"},
            {"name": "Venus", "star": "Sun"},
            {"name": "Earth", "star": "Sun"}
        ],
        "quiz_today": [
            {"question": "Capital of France?", "answer": "Paris"}
        ],
        "recent_words_dc": [
            {"word": "petrichor", "definition": "Earthy scent after rain"}
        ],
        "word_of_day_ld": {"word": "ubiquitous", "definition": "Present everywhere"},
        "word_of_day_mw": {"word": "candid", "definition": "Honest, frank"},
        "word_of_day_pm": {"word": "nostalgia", "definition": "Sentimental longing"},
        "dashboard": {"stats": {"users": 1250, "courses": 45}},
        "questions_answers": [
            {"type": "multiple_choice", "question": "2+2?", "answer": "4", "feedback": "Basic addition"}
        ],
        "periodic_table": [
            {"element": "Carbon", "symbol": "C", "mass": 12.011}
        ],
        "quotes": [
            {"text": "Knowledge is power", "author": "Francis Bacon"}
        ],
        "turkish_words": [
            {"word": "merhaba", "meaning": "hello"}
        ]
    }))


class TestEducationToolsSequentialCorrect:
    """Correct ordered sequences exercising typical user trajectories."""

    def test_topic_then_ai_question(self, education_config: dict) -> None:
        """Retrieve a topic entry, then ask AI a related question."""
        tools = EducationTools(initial_config=education_config)
        topic_result = tools.topic_topic(topic="science")
        assert topic_result is not None
        assert isinstance(topic_result, dict)

        ai_result = tools.Ask_question_to_AI_Service(question="What is gravity?")
        assert ai_result is not None
        assert isinstance(ai_result, dict)

    def test_get_random_word_then_multiple(self, education_config: dict) -> None:
        """Get a single random word, then get multiple random words."""
        tools = EducationTools(initial_config=education_config)
        single = tools.Get_a_random_word()
        assert single is not None
        assert isinstance(single, dict)

        multiple = tools.Get_multiple_random_words(count=3)
        assert multiple is not None
        assert isinstance(multiple, dict)

    def test_chemical_elements_then_periodic_table(self, education_config: dict) -> None:
        """Get all chemical elements, then get periodic table details."""
        tools = EducationTools(initial_config=education_config)
        elements = tools.Get_All_Chemical_Elements()
        assert elements is not None
        assert isinstance(elements, dict)

        table = tools.periodic_table()
        assert table is not None
        assert isinstance(table, dict)

    def test_word_of_day_sources_in_sequence(self, education_config: dict) -> None:
        """Fetch word of the day from multiple sources in sequence."""
        tools = EducationTools(initial_config=education_config)
        word_ld = tools.Word_of_the_day_from_ld()
        assert word_ld is not None
        assert isinstance(word_ld, dict)

        word_mw = tools.Word_of_the_day_from_mw()
        assert word_mw is not None
        assert isinstance(word_mw, dict)

        word_pm = tools.Word_of_the_day_from_pm()
        assert word_pm is not None
        assert isinstance(word_pm, dict)

    def test_dashboard_then_quiz_then_qa(self, education_config: dict) -> None:
        """Get dashboard, then today's quiz, then all Q&A."""
        tools = EducationTools(initial_config=education_config)
        dash = tools.dashboard()
        assert dash is not None
        assert isinstance(dash, dict)

        quiz = tools.Quiz_For_Today()
        assert quiz is not None
        assert isinstance(quiz, dict)

        qa = tools.getQuestionsandAnswers()
        assert qa is not None
        assert isinstance(qa, dict)


class TestEducationToolsSequentialProblematic:
    """Problematic sequences: nonexistent resources, invalid params, wrong order."""

    def test_nonexistent_topic_then_valid_topic(self, education_config: dict) -> None:
        """Request a nonexistent topic, then a valid topic should still work."""
        tools = EducationTools(initial_config=education_config)
        missing = tools.topic_topic(topic="nonexistent_topic")
        assert missing is not None
        assert isinstance(missing, dict)

        valid = tools.topic_topic(topic="math")
        assert valid is not None
        assert isinstance(valid, dict)

    def test_invalid_multiple_words_count_then_single_word(self, education_config: dict) -> None:
        """Request multiple words with invalid count, then single word should work."""
        tools = EducationTools(initial_config=education_config)
        invalid = tools.Get_multiple_random_words(count=-5)
        assert invalid is not None
        assert isinstance(invalid, dict)

        single = tools.Get_a_random_word()
        assert single is not None
        assert isinstance(single, dict)

    def test_missing_date_fact_then_valid_date_fact(self, education_config: dict) -> None:
        """Request a date fact for a missing date, then a valid date."""
        tools = EducationTools(initial_config=education_config)
        missing = tools.Get_date_fact(month="13", day="99")
        assert missing is not None
        assert isinstance(missing, dict)

        valid = tools.Get_date_fact(month="01", day="15")
        assert valid is not None
        assert isinstance(valid, dict)

    def test_empty_ai_question_then_valid_question(self, education_config: dict) -> None:
        """Ask AI an empty question, then a valid question."""
        tools = EducationTools(initial_config=education_config)
        empty = tools.Ask_question_to_AI_Service(question="")
        assert empty is not None
        assert isinstance(empty, dict)

        valid = tools.Ask_question_to_AI_Service(question="What is gravity?")
        assert valid is not None
        assert isinstance(valid, dict)

    def test_turkish_search_empty_then_random_treasure(self, education_config: dict) -> None:
        """Search Turkish dictionary with empty query, then get random treasure."""
        tools = EducationTools(initial_config=education_config)
        empty_search = tools.wordSearchTurkish(query="")
        assert empty_search is not None
        assert isinstance(empty_search, dict)

        treasure = tools.random()
        assert treasure is not None
        assert isinstance(treasure, dict)