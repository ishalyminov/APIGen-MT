import pytest
import json
import copy
from tools.toollens.email import EmailTools


@pytest.fixture
def email_tools_instance():
    """Create a fresh EmailTools instance with full initial config."""
    config = {
        "Current_Mail": {
            "valid_emails": [
                "john.doe@gmail.com",
                "jane.smith@yahoo.com",
                "bob.jones@outlook.com"
            ],
            "invalid_emails": [
                "not.an.email",
                "missing@domain",
                "@nodomain.com",
                "plain text"
            ],
            "mixed_list": [
                "valid.user@hotmail.com",
                "invalid@@mail.com",
                "good@bad..com",
                "another.valid@proton.me"
            ]
        },
        "Email": {
            "test_addresses": [
                "alice@company.org",
                "bob@",
                "charlie@.com",
                "valid123@sub.domain.co.uk",
                "user@name with space.com",
                "test@domain.info"
            ]
        },
        "Email_verifier": {
            "emails_to_verify": [
                "contact@business.net",
                "admin@gov.edu",
                "fake@nonexistent-domain-xyz123.com",
                "test@10minutemail.com",
                "user@mailinator.com"
            ]
        },
        "EmailVerifications": {
            "batch_valid": [
                "team@startup.io",
                "support@techcorp.com",
                "info@nonprofit.org"
            ],
            "batch_invalid": [
                "bad@",
                "@missing.com",
                "no.at.symbol",
                "double@@at.com"
            ],
            "batch_mixed": [
                "real@company.com",
                "fake@nonexistent.xyz",
                "good@email.org",
                "@bad.com",
                "valid@domain.co"
            ]
        },
        "FreeDomain": {
            "free_domains": [
                "gmail.com",
                "yahoo.com",
                "hotmail.com",
                "outlook.com",
                "protonmail.com",
                "mail.com"
            ],
            "paid_domains": [
                "company.org",
                "business.net",
                "startup.io",
                "techcorp.com",
                "enterprise.co.uk"
            ],
            "unknown_domains": [
                "randomtest1234.com",
                "fakedomain99999.net"
            ]
        },
        "Validate_domain_or_email_address": {
            "domains": [
                "spam4.me",
                "example.com",
                "sub.domain.org",
                "invalid..dots.com",
                "no-tld"
            ],
            "email_addresses": [
                "badactor@spam4.me",
                "user@example.com",
                "test@sub.domain.org",
                "@nodomain.com",
                "user@.com"
            ],
            "mixed_inputs": [
                "valid@email.com",
                "just-a-domain.com",
                "@",
                "user@",
                "domain.org"
            ]
        },
        "Verify_Email": {
            "deliverable": [
                "contact@amazon.com",
                "support@microsoft.com",
                "team@google.com"
            ],
            "undeliverable": [
                "bounce@nonexistent-domain-12345.com",
                "fake@deleted-domain-67890.net",
                "invalid@no-mx-record.xyz"
            ],
            "risky": [
                "temp@10minutemail.com",
                "disposable@mailinator.com",
                "throwaway@guerrillamail.com"
            ]
        },
        "mailCheck": {
            "valid_patterns": [
                "alice@gmail.com",
                "bob123@yahoo.com",
                "charlie.brown@hotmail.com",
                "d_morgan@outlook.com"
            ],
            "invalid_patterns": [
                "1invalid@gmail.com",
                ".startswithdot@yahoo.com",
                "-hyphen@hotmail.com",
                "_underscore@outlook.com",
                "user@domain",
                "user@.com",
                "user@domain..com"
            ],
            "edge_cases": [
                "a@b.co",
                "valid.user+tag@gmail.com",
                "user@subdomain.domain.co.uk",
                "UPPERCASE@DOMAIN.COM"
            ]
        }
    }
    return EmailTools(initial_config=copy.deepcopy(config))


class TestEmailToolsSequentialCorrect:
    """Correct ordered sequences for EmailTools."""

    def test_email_validation_then_domain_check(self, email_tools_instance):
        """Validate an email, then check if its domain is a free domain."""
        email = "john.doe@gmail.com"

        # Step 1: Validate email format
        email_result = email_tools_instance.Email(email=email)
        assert email_result is not None
        assert isinstance(email_result, dict)

        # Step 2: Extract domain and check if it's free
        domain_result = email_tools_instance.FreeDomain(domain="gmail.com")
        assert domain_result is not None
        assert isinstance(domain_result, dict)

    def test_email_verifier_then_verify_email(self, email_tools_instance):
        """Run email_verifier on an address, then run Verify_Email on the same address."""
        email = "contact@business.net"

        # Step 1: Email verifier check
        verifier_result = email_tools_instance.Email_verifier(email=email)
        assert verifier_result is not None
        assert isinstance(verifier_result, dict)

        # Step 2: Verify email deliverability
        verify_result = email_tools_instance.Verify_Email(query=email)
        assert verify_result is not None
        assert isinstance(verify_result, dict)

    def test_mailcheck_then_validate_domain_or_email(self, email_tools_instance):
        """Run mailCheck on an email, then validate the same email with Validate_domain_or_email_address."""
        email = "user@example.com"

        # Step 1: mailCheck pattern validation
        mailcheck_result = email_tools_instance.mailCheck(email=email)
        assert mailcheck_result is not None
        assert isinstance(mailcheck_result, dict)

        # Step 2: Validate domain or email address
        validate_result = email_tools_instance.Validate_domain_or_email_address(validate=email)
        assert validate_result is not None
        assert isinstance(validate_result, dict)

    def test_current_mail_then_email_verifications(self, email_tools_instance):
        """Get current mail session, then run batch email verifications."""
        # Step 1: Get current mail
        current_mail_result = email_tools_instance.Current_Mail()
        assert current_mail_result is not None
        assert isinstance(current_mail_result, dict)

        # Step 2: Run email verifications batch
        verifications_result = email_tools_instance.EmailVerifications()
        assert verifications_result is not None
        assert isinstance(verifications_result, dict)

    def test_validate_domain_then_freedomain_check(self, email_tools_instance):
        """Validate a domain, then check if it's a free domain."""
        domain = "gmail.com"

        # Step 1: Validate domain or email address
        validate_result = email_tools_instance.Validate_domain_or_email_address(validate=domain)
        assert validate_result is not None
        assert isinstance(validate_result, dict)

        # Step 2: Check if domain is free
        free_result = email_tools_instance.FreeDomain(domain=domain)
        assert free_result is not None
        assert isinstance(free_result, dict)


class TestEmailToolsSequentialProblematic:
    """Problematic sequences for EmailTools."""

    def test_email_invalid_then_freedomain_still_works(self, email_tools_instance):
        """Call Email with invalid address, then FreeDomain should still work normally."""
        # Step 1: Invalid email - should not crash
        invalid_email = "not.an.email"
        email_result = email_tools_instance.Email(email=invalid_email)
        assert email_result is not None
        assert isinstance(email_result, dict)

        # Step 2: FreeDomain should still function
        domain_result = email_tools_instance.FreeDomain(domain="yahoo.com")
        assert domain_result is not None
        assert isinstance(domain_result, dict)

    def test_verify_nonexistent_email_then_validate_bad_domain(self, email_tools_instance):
        """Verify a nonexistent email, then validate an invalid domain - both should return error info."""
        # Step 1: Verify nonexistent email
        verify_result = email_tools_instance.Verify_Email(query="bounce@nonexistent-domain-12345.com")
        assert verify_result is not None
        assert isinstance(verify_result, dict)

        # Step 2: Validate invalid domain
        validate_result = email_tools_instance.Validate_domain_or_email_address(validate="no-tld")
        assert validate_result is not None
        assert isinstance(validate_result, dict)

    def test_mailcheck_invalid_then_email_verifier_invalid(self, email_tools_instance):
        """Run mailCheck on invalid pattern, then email_verifier on another invalid address."""
        # Step 1: mailCheck with invalid pattern
        mailcheck_result = email_tools_instance.mailCheck(email="user@domain..com")
        assert mailcheck_result is not None
        assert isinstance(mailcheck_result, dict)

        # Step 2: Email_verifier with invalid email
        verifier_result = email_tools_instance.Email_verifier(email="fake@nonexistent-domain-xyz123.com")
        assert verifier_result is not None
        assert isinstance(verifier_result, dict)

    def test_freedomain_unknown_then_email_missing_arg(self, email_tools_instance):
        """Check an unknown domain, then call Email without proper args."""
        # Step 1: FreeDomain with unknown domain
        domain_result = email_tools_instance.FreeDomain(domain="randomtest1234.com")
        assert domain_result is not None
        assert isinstance(domain_result, dict)

        # Step 2: Email with empty string
        email_result = email_tools_instance.Email(email="")
        assert email_result is not None
        assert isinstance(email_result, dict)

    def test_email_verifier_empty_then_current_mail_still_works(self, email_tools_instance):
        """Call email_verifier with empty string, then Current_Mail should still work."""
        # Step 1: Email_verifier with empty email
        verifier_result = email_tools_instance.Email_verifier(email="")
        assert verifier_result is not None
        assert isinstance(verifier_result, dict)

        # Step 2: Current_Mail should still function
        current_result = email_tools_instance.Current_Mail()
        assert current_result is not None
        assert isinstance(current_result, dict)