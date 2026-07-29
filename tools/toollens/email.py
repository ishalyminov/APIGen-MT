"""Auto-generated EmailTools implementation."""

import re
import math
import json
import copy
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union


class EmailTools:
    """Email-related tools for validation, verification, and temporary mail management."""

    METHOD_NAME_MAP = {
        'Current Mail': 'Current_Mail',
        'Email': 'Email',
        'Email verifier': 'Email_verifier',
        'EmailVerifications': 'EmailVerifications',
        'FreeDomain': 'FreeDomain',
        'Validate domain or email address': 'Validate_domain_or_email_address',
        'Verify Email': 'Verify_Email',
        'mailCheck': 'mailCheck',
    }

    # Common free email domains
    _FREE_DOMAINS = {
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'live.com',
        'aol.com', 'icloud.com', 'mail.com', 'protonmail.com', 'zoho.com',
        'msn.com', 'yandex.com', 'gmx.com',
    }

    # Common disposable email domains
    _DISPOSABLE_DOMAINS = {
        'spam4.me', 'mailinator.com', 'guerrillamail.com', '10minutemail.com',
        'tempmail.com', 'throwawaymail.com', 'trashmail.com', 'fakeinbox.com',
        'sharklasers.com', 'guerrillamailblock.com', 'dispostable.com',
        'maildrop.cc', 'getnada.com', 'tempmailaddress.com',
    }

    def __init__(self, initial_config: dict = None) -> None:
        """Initialize the EmailTools instance with optional configuration."""
        self._config_data: Dict[str, Any] = {}
        if initial_config is None:
            self._init_state()
        else:
            self._init_state()
            if isinstance(initial_config, dict):
                for key, value in initial_config.items():
                    existing = getattr(type(self), key, None)
                    if existing is not None and callable(existing):
                        self._config_data[key] = value
                    elif key.startswith('_'):
                        self._config_data[key] = value
                    else:
                        setattr(self, key, value)

    def _init_state(self) -> None:
        """Set up internal state for temporary mail sessions and caches."""
        self._session_counter = 0
        self._current_session_id = "sess_" + str(random.randint(100000, 999999))
        self._current_user = "user" + str(random.randint(1000, 9999))
        self._current_host = "smailpro.com"
        self._current_mail = self._current_user + "@" + self._current_host
        self._current_key = "key_" + str(random.randint(100000, 999999))
        self._server_time = int(datetime.now().timestamp())
        self._mail_time = self._server_time - 3600
        self._due_time = self._server_time + 3600
        self._left_time = 3600
        self._validation_cache: Dict[str, dict] = {}
        self._verification_cache: Dict[str, dict] = {}

    def _is_valid_email_format(self, email: str) -> bool:
        """Check if email has a valid format using regex."""
        if not email or not isinstance(email, str):
            return False
        pattern = r'^[a-zA-Z0-9][a-zA-Z0-9._+%-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def _extract_domain(self, email: str) -> Optional[str]:
        """Extract the domain part from an email address."""
        if not email or '@' not in email:
            return None
        parts = email.rsplit('@', 1)
        if len(parts) == 2:
            return parts[1].lower()
        return None

    def _extract_username(self, email: str) -> Optional[str]:
        """Extract the username part from an email address."""
        if not email or '@' not in email:
            return None
        parts = email.rsplit('@', 1)
        if len(parts) == 2:
            return parts[0]
        return None

    def Current_Mail(self) -> dict:
        """
        Retrieve the current temporary mail session information.
        Returns details about the active temporary email address including
        session ID, mail host, and timing information.
        """
        try:
            self._server_time = int(datetime.now().timestamp())
            self._left_time = max(0, self._due_time - self._server_time)

            permalink = {
                "host": self._current_host,
                "mail": self._current_mail,
                "key": self._current_key,
                "url": "https://" + self._current_host + "/mail/" + self._current_key,
            }

            return {
                "mail_get_user": self._current_user,
                "mail_get_mail": self._current_mail,
                "mail_get_host": self._current_host,
                "mail_get_time": self._mail_time,
                "mail_get_duetime": self._due_time,
                "mail_server_time": self._server_time,
                "mail_get_key": self._current_key,
                "mail_left_time": self._left_time,
                "mail_recovering_key": None,
                "mail_recovering_mail": None,
                "session_id": self._current_session_id,
                "permalink": permalink,
            }
        except Exception as e:
            return {
                "mail_get_user": "",
                "mail_get_mail": "",
                "mail_get_host": "",
                "mail_get_time": 0,
                "mail_get_duetime": 0,
                "mail_server_time": 0,
                "mail_get_key": "",
                "mail_left_time": 0,
                "mail_recovering_key": None,
                "mail_recovering_mail": None,
                "session_id": "",
                "permalink": {"host": "", "mail": "", "key": "", "url": ""},
            }

    def Email(self, email: str) -> dict:
        """
        Validate an email address with domain name check.
        Filters out invalid emails and domains to improve delivery rate
        and minimize email bounce.
        """
        try:
            if not email or not isinstance(email, str):
                return {
                    "valid": False,
                    "email": email if isinstance(email, str) else "",
                    "deliverable": False,
                    "domain_valid": False,
                    "risk_level": "high",
                    "reason": "Email parameter is missing or invalid",
                }

            email = email.strip().lower()
            is_valid_format = self._is_valid_email_format(email)
            domain = self._extract_domain(email)

            if not is_valid_format:
                return {
                    "valid": False,
                    "email": email,
                    "deliverable": False,
                    "domain_valid": False,
                    "risk_level": "high",
                    "reason": "Invalid email format",
                }

            domain_valid = domain is not None and '.' in domain
            is_disposable = domain in self._DISPOSABLE_DOMAINS
            is_free = domain in self._FREE_DOMAINS

            if is_disposable:
                risk_level = "high"
                reason = "Disposable email domain detected"
                deliverable = False
            elif not domain_valid:
                risk_level = "high"
                reason = "Domain does not have valid DNS records"
                deliverable = False
            elif is_free:
                risk_level = "low"
                reason = "Free email provider, domain is valid"
                deliverable = True
            else:
                risk_level = "medium"
                reason = "Custom domain, deliverability uncertain"
                deliverable = True

            return {
                "valid": is_valid_format and domain_valid,
                "email": email,
                "deliverable": deliverable,
                "domain_valid": domain_valid,
                "risk_level": risk_level,
                "reason": reason,
            }
        except Exception as e:
            return {
                "valid": False,
                "email": email if isinstance(email, str) else "",
                "deliverable": False,
                "domain_valid": False,
                "risk_level": "high",
                "reason": "An error occurred during validation",
            }

    def Email_verifier(self, email: str) -> dict:
        """
        Verify the validity of an email address.
        Checks email format, domain, disposable status, MX records,
        and simulates SMTP connection to confirm mailbox existence.
        """
        try:
            if not email or not isinstance(email, str):
                return {
                    "reason": "Email parameter is missing",
                    "status": "invalid",
                }

            email = email.strip().lower()
            is_valid_format = self._is_valid_email_format(email)
            domain = self._extract_domain(email)

            if not is_valid_format:
                return {
                    "reason": "Invalid email format",
                    "status": "invalid",
                }

            if domain in self._DISPOSABLE_DOMAINS:
                return {
                    "reason": "Disposable email address detected",
                    "status": "invalid",
                }

            if domain and '.' in domain:
                return {
                    "reason": "Email address is valid and mailbox exists",
                    "status": "valid",
                }
            else:
                return {
                    "reason": "Domain does not have valid MX records",
                    "status": "invalid",
                }
        except Exception as e:
            return {
                "reason": "An error occurred during verification",
                "status": "invalid",
            }

    def EmailVerifications(self) -> dict:
        """
        Verify a list of email addresses with different dimensions.
        Returns verification details including syntax, SMTP, disposable,
        role account, and free domain checks.
        """
        try:
            email = self._current_mail
            domain = self._extract_domain(email) or ""
            username = self._extract_username(email) or ""

            is_valid = self._is_valid_email_format(email)
            is_disposable = domain in self._DISPOSABLE_DOMAINS
            is_free = domain in self._FREE_DOMAINS
            has_mx = is_valid and '.' in domain

            role_accounts = {'admin', 'info', 'support', 'sales', 'contact', 'help', 'postmaster', 'webmaster'}
            is_role = username.lower() in role_accounts

            reachable = "safe" if (is_valid and not is_disposable) else "risky"

            return {
                "email": email,
                "reachable": reachable,
                "syntax": {
                    "username": username,
                    "domain": domain,
                    "valid": is_valid,
                },
                "smtp": None,
                "gravatar": None,
                "suggestion": "",
                "disposable": is_disposable,
                "role_account": is_role,
                "free": is_free,
                "has_mx_records": has_mx,
            }
        except Exception as e:
            return {
                "email": "",
                "reachable": "unknown",
                "syntax": {
                    "username": "",
                    "domain": "",
                    "valid": False,
                },
                "smtp": None,
                "gravatar": None,
                "suggestion": "",
                "disposable": False,
                "role_account": False,
                "free": False,
                "has_mx_records": False,
            }

    def FreeDomain(self, domain: str) -> dict:
        """
        Check whether or not a domain is a free domain.
        Returns the domain name and whether it is classified as free.
        """
        try:
            if not domain or not isinstance(domain, str):
                return {
                    "FreeDomain": "false",
                }

            domain = domain.strip().lower()
            # Remove protocol if present
            domain = re.sub(r'^https?://', '', domain)
            # Remove path if present
            domain = domain.split('/')[0]

            is_free = domain in self._FREE_DOMAINS

            return {
                "FreeDomain": "true" if is_free else "false",
            }
        except Exception as e:
            return {
                "FreeDomain": "false",
            }

    def Validate_domain_or_email_address(self, validate: str) -> dict:
        """
        Validate either a domain (e.g., spam4.me) or an email address
        (e.g., badactor@spam4.me). Checks if the input is a disposable domain.
        """
        try:
            if not validate or not isinstance(validate, str):
                return {
                    "is_disposable_domain": False,
                }

            validate = validate.strip().lower()

            # Determine if input is an email or a domain
            if '@' in validate:
                domain = self._extract_domain(validate)
            else:
                # Remove protocol if present
                domain = re.sub(r'^https?://', '', validate)
                domain = domain.split('/')[0]

            if domain is None:
                return {
                    "is_disposable_domain": False,
                }

            is_disposable = domain in self._DISPOSABLE_DOMAINS

            return {
                "is_disposable_domain": is_disposable,
            }
        except Exception as e:
            return {
                "is_disposable_domain": False,
            }

    def Verify_Email(self, query: str) -> dict:
        """
        Validate an email address and check if it is deliverable.
        Performs comprehensive checks including format, domain, and mailbox.
        """
        try:
            if not query or not isinstance(query, str):
                return {
                    "message": "Email address is missing or invalid",
                }

            query = query.strip().lower()
            is_valid = self._is_valid_email_format(query)
            domain = self._extract_domain(query)

            if not is_valid:
                return {
                    "message": "The email address " + query + " is not valid. Invalid format.",
                }

            if domain in self._DISPOSABLE_DOMAINS:
                return {
                    "message": "The email address " + query + " is valid but uses a disposable domain. Not deliverable.",
                }

            if domain and '.' in domain:
                return {
                    "message": "The email address " + query + " is valid and deliverable.",
                }
            else:
                return {
                    "message": "The email address " + query + " has an invalid domain. Not deliverable.",
                }
        except Exception as e:
            return {
                "message": "An error occurred during email verification",
            }

    def mailCheck(self, email: str) -> dict:
        """
        Check email validity with strict filtering rules.
        Validates that the email:
        - Does not start with a special character (must begin with letter, digit, or +, _, ., -)
        - Contains @ symbol separating username and domain
        - Domain does not start or end with dot or hyphen
        - Domain contains only alphanumeric characters or dots
        - No additional characters after the domain part
        """
        try:
            if not email or not isinstance(email, str):
                return {
                    "valid": False,
                    "email": email if isinstance(email, str) else "",
                }

            email = email.strip()

            # Check overall format with strict pattern
            # Username: starts with letter, digit, or +, _, ., - followed by alphanumeric and allowed chars
            # Domain: alphanumeric and dots, no leading/trailing dot or hyphen
            pattern = r'^[a-zA-Z0-9+_.\-][a-zA-Z0-9._+%-]*@[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}$'
            is_valid = re.match(pattern, email) is not None

            if is_valid:
                # Additional check: domain part should not start or end with dot or hyphen
                domain = self._extract_domain(email)
                if domain:
                    if domain.startswith('.') or domain.startswith('-') or domain.endswith('.') or domain.endswith('-'):
                        is_valid = False
                    # Check domain only contains alphanumeric and dots (no hyphens in TLD area)
                    if '..' in domain:
                        is_valid = False

            return {
                "valid": is_valid,
                "email": email,
            }
        except Exception as e:
            return {
                "valid": False,
                "email": email if isinstance(email, str) else "",
            }