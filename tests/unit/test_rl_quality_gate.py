from rl_quality_gate import validate_transition_quality


def issue_codes(result):
    return {issue["code"] for issue in result["issues"]}


def test_rejects_noop_mutation():
    state = {"trading_bot": {"watch_list": ["AAPL"]}}
    result = validate_transition_quality(
        tool_name="add_to_watchlist",
        tool_output={"symbol": "AAPL"},
        pre_state=state,
        post_state=state,
    )
    assert result["passed"] is False
    assert "MUTATION_NO_EFFECT" in issue_codes(result)


def test_rejects_create_identifier_collision():
    pre = {
        "posting_api": {
            "tweets": {
                "25": {"id": 25, "content": "old"},
            }
        }
    }
    post = {
        "posting_api": {
            "tweets": {
                "25": {"id": 25, "content": "new"},
            }
        }
    }
    result = validate_transition_quality(
        tool_name="post_tweet",
        tool_output={"id": 25, "content": "new"},
        pre_state=pre,
        post_state=post,
    )
    codes = issue_codes(result)
    assert result["passed"] is False
    assert "CREATED_ID_COLLISION" in codes
    assert "CREATE_COLLECTION_DID_NOT_GROW" in codes


def test_accepts_clean_create_transition():
    pre = {
        "posting_api": {
            "tweets": {
                "25": {"id": 25, "content": "old"},
            }
        }
    }
    post = {
        "posting_api": {
            "tweets": {
                "25": {"id": 25, "content": "old"},
                "26": {"id": 26, "content": "new"},
            }
        }
    }
    result = validate_transition_quality(
        tool_name="post_tweet",
        tool_output={"id": 26, "content": "new"},
        pre_state=pre,
        post_state=post,
    )
    assert result == {
        "tool_name": "post_tweet",
        "passed": True,
        "issues": [],
    }


def test_read_only_noop_is_allowed():
    state = {"vehicle_control": {"fuelLevel": 15.0}}
    result = validate_transition_quality(
        tool_name="displayCarStatus",
        tool_output={"status": {"fuelLevel": 15.0}},
        pre_state=state,
        post_state=state,
    )
    assert result["passed"] is True


def test_zero_is_a_valid_first_created_identifier():
    result = validate_transition_quality(
        tool_name="post_tweet",
        tool_output={"id": 0},
        pre_state={"posting_api": {"tweets": {}}},
        post_state={"posting_api": {"tweets": {"0": {"id": 0}}}},
    )
    assert result["passed"] is True


def test_terminal_echo_is_not_rejected_as_a_noop_mutation():
    result = validate_transition_quality(
        tool_name="echo",
        tool_output={"terminal_output": "hello"},
        pre_state={"gorilla_file_system": {"current_dir": "/"}},
        post_state={"gorilla_file_system": {"current_dir": "/"}},
        tool_arguments={"content": "hello"},
    )
    assert result["passed"] is True
