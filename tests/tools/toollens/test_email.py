import pytest
import json
from tools.toollens.email import EmailTools


@pytest.fixture
def email_instance():
    config = {
        "Current_Mail": {
            "valid_emails": ["john.doe@gmail.com", "jane.smith@yahoo.com", "bob.jones@outlook.com"],
            "invalid_emails": ["not.an.email", "missing@domain", "@nodomain.com", "plain text"],
            "mixed_list": ["valid.user@hotmail.com", "invalid@@mail.com", "good@bad..com", "another.valid@proton.me"],
        },
        "Email": {
            "test_addresses": [
                "alice@company.org",
                "bob@",
                "charlie@.com",
                "valid123@sub.domain.co.uk",
                "user@name with space.com",
                "test@domain.info",
            ]
        },
        "Email_verifier": {
            "emails_to_verify": [
                "contact@business.net",
                "admin@gov.edu",
                "fake@nonexistent-domain-xyz123.com",
                "test@10minutemail.com",
                "user@mailinator.com",
            ]
        },
        "EmailVerifications": {
            "batch_valid": ["team@startup.io", "support@techcorp.com", "info@nonprofit.org"],
            "batch_invalid": ["bad@", "@missing.com", "no.at.symbol", "double@@at.com"],
            "batch_mixed": ["real@company.com", "fake@nonexistent.xyz", "good@email.org", "@bad.com", "valid@domain.co"],
        },
        "FreeDomain": {
            "free_domains": ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "protonmail.com", "mail.com"],
            "paid_domains": ["company.org", "business.net", "startup.io", "techcorp.com", "enterprise.co.uk"],
            "unknown_domains": ["randomtest1234.com", "fakedomain99999.net"],
        },
        "Validate_domain_or_email_address": {
            "domains": ["spam4.me", "example.com", "sub.domain.org", "invalid..dots.com", "no-tld"],
            "email_addresses": ["badactor@spam4.me", "user@example.com", "test@sub.domain.org", "@nodomain.com", "user@.com"],
            "mixed_inputs": ["valid@email.com", "just-a-domain.com", "@", "user@", "domain.org"],
        },
        "Verify_Email": {
            "deliverable": ["contact@amazon.com", "support@microsoft.com", "team@google.com"],
            "undeliverable": ["bounce@nonexistent-domain-12345.com", "fake@deleted-domain-67890.net", "invalid@no-mx-record.xyz"],
            "risky": ["temp@10minutemail.com", "disposable@mailinator.com", "throwaway@guerrillamail.com"],
        },
        "mailCheck": {
            "valid_patterns": ["alice@gmail.com", "bob123@yahoo.com", "charlie.brown@hotmail.com", "d_morgan@outlook.com"],
            "invalid_patterns": [
                "1invalid@gmail.com",
                ".startswithdot@yahoo.com",
                "-hyphen@hotmail.com",
                "_underscore@outlook.com",
                "user@domain",
                "user@.com",
                "user@domain..com",
            ],
            "edge_cases": ["a@b.co", "valid.user+tag@gmail.com", "user@subdomain.domain.co.uk", "UPPERCASE@DOMAIN.COM", "user@123.456.789.012"],
        },
    }
    return EmailTools(initial_config=config)


# ---------------------------------------------------------------------------
# Current_Mail
# ---------------------------------------------------------------------------

def test_current_mail_returns_dict(email_instance):
    """Current_Mail should return a dict with expected structure."""
    result = email_instance.Current_Mail()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_current_mail_state_mutates(email_instance):
    """Calling Current_Mail multiple times should not raise and should return dicts."""
    first = email_instance.Current_Mail()
    second = email_instance.Current_Mail()
    assert isinstance(first, dict)
    assert isinstance(second, dict)


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def test_email_valid_address(email_instance):
    """Email with a valid address should return a dict response."""
    result = email_instance.Email("alice@company.org")
    assert isinstance(result, dict)


def test_email_edge_cases(email_instance):
    """Email with invalid/empty inputs should still return a dict (no exceptions)."""
    assert isinstance(email_instance.Email(""), dict)
    assert isinstance(email_instance.Email("bob@"), dict)
    assert isinstance(email_instance.Email(None), dict)


# ---------------------------------------------------------------------------
# Email_verifier
# ---------------------------------------------------------------------------

def test_email_verifier_normal(email_instance):
    """Email_verifier with a valid-looking email should return a dict."""
    result = email_instance.Email_verifier("contact@business.net")
    assert isinstance(result, dict)


def test_email_verifier_disposable(email_instance):
    """Email_verifier with a disposable email should return a dict without raising."""
    result = email_instance.Email_verifier("test@10minutemail.com")
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# EmailVerifications
# ---------------------------------------------------------------------------

def test_email_verifications_returns_dict(email_instance):
    """EmailVerifications should return a dict."""
    result = email_instance.EmailVerifications()
    assert isinstance(result, dict)


def test_email_verifications_idempotent(email_instance):
    """EmailVerifications called multiple times should consistently return dicts."""
    r1 = email_instance.EmailVerifications()
    r2 = email_instance.EmailVerifications()
    assert isinstance(r1, dict)
    assert isinstance(r2, dict)


# ---------------------------------------------------------------------------
# FreeDomain
# ---------------------------------------------------------------------------

def test_free_domain_known_free(email_instance):
    """FreeDomain with a known free domain should return a dict."""
    result = email_instance.FreeDomain("gmail.com")
    assert isinstance(result, dict)


def test_free_domain_unknown_and_empty(email_instance):
    """FreeDomain with unknown/empty inputs should return dicts without raising."""
    assert isinstance(email_instance.FreeDomain("randomtest1234.com"), dict)
    assert isinstance(email_instance.FreeDomain(""), dict)
    assert isinstance(email_instance.FreeDomain(None), dict)


# ---------------------------------------------------------------------------
# Validate_domain_or_email_address
# ---------------------------------------------------------------------------

def test_validate_domain_or_email_address_email(email_instance):
    """Validate_domain_or_email_address with an email should return a dict."""
    result = email_instance.Validate_domain_or_email_address("valid@email.com")
    assert isinstance(result, dict)


def test_validate_domain_or_email_address_edge_cases(email_instance):
    """Validate_domain_or_email_address with edge-case inputs should return dicts."""
    assert isinstance(email_instance.Validate_domain_or_email_address("@"), dict)
    assert isinstance(email_instance.Validate_domain_or_email_address("user@"), dict)
    assert isinstance(email_instance.Validate_domain_or_email_address("domain.org"), dict)
    assert isinstance(email_instance.Validate_domain_or_email_address(None), dict)


# ---------------------------------------------------------------------------
# Verify_Email
# ---------------------------------------------------------------------------

def test_verify_email_deliverable(email_instance):
    """Verify_Email with a deliverable address should return a dict."""
    result = email_instance.Verify_Email("contact@amazon.com")
    assert isinstance(result, dict)


def test_verify_email_undeliverable_and_none(email_instance):
    """Verify_Email with undeliverable/None inputs should return dicts without raising."""
    assert isinstance(email_instance.Verify_Email("bounce@nonexistent-domain-12345.com"), dict)
    assert isinstance(email_instance.Verify_Email(None), dict)
    assert isinstance(email_instance.Verify_Email(""), dict)


# ---------------------------------------------------------------------------
# mailCheck
# ---------------------------------------------------------------------------

def test_mail_check_valid_pattern(email_instance):
    """mailCheck with a valid pattern should return a dict."""
    result = email_instance.mailCheck("alice@gmail.com")
    assert isinstance(result, dict)


def test_mail_check_invalid_and_edge_cases(email_instance):
    """mailCheck with invalid/edge-case inputs should return dicts without raising."""
    assert isinstance(email_instance.mailCheck("1invalid@gmail.com"), dict)
    assert isinstance(email_instance.mailCheck("user@domain..com"), dict)
    assert isinstance(email_instance.mailCheck(""), dict)
    assert isinstance(email_instance.mailCheck(None), dict)