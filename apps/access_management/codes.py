"""The permission vocabulary — one constant per code, referenced everywhere a view
declares ``required_permission`` instead of retyping the string.

Does NOT replace ``migrations/0004_seed_permissions.py`` as the source that writes
these rows into the database — migrations must stay self-contained so an old
migration keeps reproducing the same database state even if this file changes later.
This module exists so a typo in a view (``"roles.manage"`` for ``"role.manage"``)
raises ``AttributeError`` at import time instead of silently never matching a grant.
``tests/test_permission_management.py`` asserts the two lists stay identical.
"""
from __future__ import annotations


class PermissionCode:
    QUERY_EXECUTE = "query.execute"
    DATA_READ = "data.read"
    SOURCE_MANAGE = "source.manage"
    INGESTION_RUN = "ingestion.run"
    EVALUATION_RUN = "evaluation.run"
    USER_MANAGE = "user.manage"
    ROLE_MANAGE = "role.manage"
    PERMISSION_READ = "permission.read"
