"""``manage.py bootstrap_admin`` — create the platform's first administrator.

The ONLY way to create the first admin. There is no HTTP endpoint for this and
there must never be one — see ``services/bootstrap.py`` for why. Run once, on a
fresh install, by whoever has shell/deploy access; refuses (exit code 1, no
partial write) if any user already exists.
"""
import getpass

from django.core.management.base import BaseCommand, CommandError

from apps.access_management.services import AdminBootstrapService, AlreadyBootstrapped
from apps.access_management.services import AccessManagementError


class Command(BaseCommand):
    help = ("Create the platform's first administrator: is_staff=True and the "
            "RBAC Admin role, atomically. Fails if any user already exists.")

    def add_arguments(self, parser):
        parser.add_argument("--username", required=True)
        parser.add_argument("--email", required=True)
        parser.add_argument(
            "--password", default=None,
            help="Omit to be prompted (recommended — avoids the password landing "
                 "in shell history).")
        parser.add_argument("--first-name", default="")
        parser.add_argument("--last-name", default="")

    def handle(self, *args, **options):
        password = options["password"] or getpass.getpass("Admin password: ")

        try:
            user = AdminBootstrapService().bootstrap(
                username=options["username"],
                email=options["email"],
                password=password,
                first_name=options["first_name"],
                last_name=options["last_name"],
            )
        except AlreadyBootstrapped as exc:
            raise CommandError(exc.message) from exc
        except AccessManagementError as exc:
            # Username/email conflict etc. — the exception's own safe message,
            # never a raw traceback, matches every other admin-facing surface.
            raise CommandError(exc.message) from exc

        self.stdout.write(self.style.SUCCESS(
            f"Admin created: user_id={user.pk} username={user.username}"))
