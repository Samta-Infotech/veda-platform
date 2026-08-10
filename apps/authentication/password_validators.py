"""Password composition rules Django's own validators don't check.

Django ships ``MinimumLengthValidator``/``CommonPasswordValidator``/
``NumericPasswordValidator``/``UserAttributeSimilarityValidator`` (all four already
configured in ``AUTH_PASSWORD_VALIDATORS``) — none of them require a mix of
uppercase, lowercase, digit and special characters. This validator fills exactly
that gap and nothing else; it does not re-check length or similarity, which are
already someone else's job.

Follows Django's own ``validate_password()`` protocol (``validate(password, user=
None)`` raising ``django.core.exceptions.ValidationError``, plus ``get_help_text()``),
so it plugs into ``AUTH_PASSWORD_VALIDATORS`` exactly like the four stock ones — no
new call site anywhere that already calls ``validate_password()``
(``UserCreateSerializer`` today; the password-change endpoint reuses the same call).

CONFIGURABLE, NOT HARDCODED
    Every threshold is a constructor argument, read from this validator's own
    ``OPTIONS`` block in ``AUTH_PASSWORD_VALIDATORS`` (see ``config/settings/base.py``)
    — the same mechanism Django's stock validators use for their own options (e.g.
    ``MinimumLengthValidator``'s ``min_length``). Changing the policy is an edit to
    that OPTIONS dict, never a code change here.
"""
from __future__ import annotations

import re

from django.core.exceptions import ValidationError

#: Any of these counts as "a special character" — deliberately a fixed character
#: class (not itself an OPTIONS knob): configurability belongs to HOW MANY of each
#: class are required, not to what counts as a class, which is a definition, not a
#: policy.
_SPECIAL_CHARS = re.escape("!@#$%^&*()_+-=[]{}|;:,.<>?/~`\"'\\")


class PasswordComplexityValidator:
    """Requires a minimum count of uppercase/lowercase/digit/special characters.

    A count of 0 for any category disables that check entirely — set
    ``min_special: 0`` in OPTIONS for a deployment that does not want to force
    special characters, rather than needing a different validator class.
    """

    def __init__(self, min_uppercase: int = 1, min_lowercase: int = 1,
                min_digits: int = 1, min_special: int = 1):
        self.min_uppercase = min_uppercase
        self.min_lowercase = min_lowercase
        self.min_digits = min_digits
        self.min_special = min_special

    def validate(self, password: str, user=None) -> None:
        counts = {
            "uppercase letter": (self.min_uppercase, sum(c.isupper() for c in password)),
            "lowercase letter": (self.min_lowercase, sum(c.islower() for c in password)),
            "digit": (self.min_digits, sum(c.isdigit() for c in password)),
            "special character": (
                self.min_special, len(re.findall(f"[{_SPECIAL_CHARS}]", password))),
        }
        errors = [
            f"This password must contain at least {required} {label}"
            f"{'s' if required != 1 else ''}."
            for label, (required, actual) in counts.items()
            if actual < required
        ]
        if errors:
            # One ValidationError, all failing rules at once — a caller correcting
            # a weak password one rejected rule at a time is a bad round trip.
            raise ValidationError(errors)

    def get_help_text(self) -> str:
        parts = []
        if self.min_uppercase:
            parts.append(f"{self.min_uppercase} uppercase letter(s)")
        if self.min_lowercase:
            parts.append(f"{self.min_lowercase} lowercase letter(s)")
        if self.min_digits:
            parts.append(f"{self.min_digits} digit(s)")
        if self.min_special:
            parts.append(f"{self.min_special} special character(s)")
        return "Your password must contain at least " + ", ".join(parts) + "."
