"""Regression coverage for a real production gap found during live/manual
Postman testing (2026-08-08), not by any unit or integration test in this
repo: ``VEDA_RBAC_MODE`` was set correctly in the deployment environment
(``.env`` / ``docker compose``), but ``apps.access_management.gate.rbac_mode()``
reported ``"off"`` regardless — because ``config/settings/base.py`` never
copied the env var onto a Django setting at all. ``rbac_mode()``'s fallback
(``getattr(settings, "VEDA_RBAC_MODE", MODE_OFF)``) triggers on an ABSENT
setting ATTRIBUTE, not on an absent env var, so a real deployment silently
never enforced RBAC no matter what was configured — while every
``override_settings(VEDA_RBAC_MODE=...)``-based test in this whole RBAC
programme (correctly) bypassed this exact wiring gap, so nothing caught it
until a real container was actually started with a real environment.

This file tests the wiring itself, end to end, in a FRESH subprocess with a
controlled environment BEFORE Django settings load — the only way to catch
"the env var never reaches the setting" (patching ``django.conf.settings``
after the fact, or using ``override_settings``, would hide the exact bug this
guards against).

Run from repo root: ``pytest tests/test_settings_env_wiring.py``
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PROBE = (
    "import django, os; "
    "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.dev'); "
    "django.setup(); "
    "from django.conf import settings; "
    "print('VEDA_RBAC_MODE=' + repr(getattr(settings, 'VEDA_RBAC_MODE', '<ABSENT>'))); "
    "print('VEDA_JWT_AUTH=' + repr(getattr(settings, 'VEDA_JWT_AUTH', '<ABSENT>')))"
)


def _probe(env_overrides: dict) -> dict:
    env = dict(os.environ)
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"probe process failed:\n{result.stdout}\n{result.stderr}"
    out = {}
    for line in result.stdout.strip().splitlines():
        key, _, value = line.partition("=")
        out[key] = eval(value)  # noqa: S307 — controlled probe output, not untrusted input
    return out


def test_veda_rbac_mode_env_var_reaches_the_django_setting():
    """The exact bug: VEDA_RBAC_MODE=enforce in the environment must produce
    settings.VEDA_RBAC_MODE == "enforce", not an absent attribute that
    rbac_mode() silently falls back to "off" for."""
    out = _probe({"VEDA_RBAC_MODE": "enforce"})
    assert out["VEDA_RBAC_MODE"] == "enforce"


def test_veda_rbac_mode_defaults_to_off_when_unset():
    env = dict(os.environ)
    env.pop("VEDA_RBAC_MODE", None)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"probe process failed:\n{result.stdout}\n{result.stderr}"
    assert "VEDA_RBAC_MODE='off'" in result.stdout


def test_veda_jwt_auth_env_var_also_reaches_the_django_setting():
    """Sibling flag, same wiring pattern — confirms this file's approach also
    exercises the ONE flag that already worked, as a sanity check on the probe
    itself (not just on the newly-fixed one)."""
    out = _probe({"VEDA_JWT_AUTH": "1"})
    assert out["VEDA_JWT_AUTH"] is True
