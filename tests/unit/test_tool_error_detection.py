from apigen_step_by_step import StepByStepGenerator


def test_user_content_with_failure_language_is_not_a_tool_error():
    output = {
        "id": 1001,
        "title": "Login failure",
        "description": "Users unable to log in after password reset.",
        "status": "Open",
    }

    assert StepByStepGenerator._detect_tool_error(
        "get_user_tickets", output
    ) == (False, "")


def test_nested_post_content_with_failure_language_is_not_a_tool_error():
    output = {
        "matching_tweets": [
            {"content": "I could not find the report and am unable to proceed."}
        ]
    }

    assert StepByStepGenerator._detect_tool_error(
        "search_tweets", output
    ) == (False, "")


def test_explicit_status_and_error_fields_still_fail_closed():
    assert StepByStepGenerator._detect_tool_error(
        "cancel_order", {"status": "Order not found"}
    )[0] is True
    assert StepByStepGenerator._detect_tool_error(
        "add_contact",
        {"added_status": False, "message": "Already exists."},
    )[0] is True
    assert StepByStepGenerator._detect_tool_error(
        "ls", {"error": "Directory not found"}
    )[0] is True


def test_empty_error_field_is_not_a_failure():
    assert StepByStepGenerator._detect_tool_error(
        "some_tool", {"error": "", "result": "ok"}
    ) == (False, "")
