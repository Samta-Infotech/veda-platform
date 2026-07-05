#!/bin/bash
# entrypoint for the api image (Django api / celery worker / celery beat share one image,
# migration_plan.md §3). Role is selected by $ROLE so the same container image can run
# any of the three processes.
#
# NOTE: must be chmod +x (already set in Dockerfile.api; re-set it if you copy this file
# elsewhere).
set -euo pipefail

ROLE="${ROLE:-api}"

case "$ROLE" in
    api)
        python manage.py migrate --noinput
        python manage.py collectstatic --noinput
        exec gunicorn config.wsgi:application \
            --bind 0.0.0.0:8000 \
            --workers "${GUNICORN_WORKERS:-4}" \
            --timeout "${GUNICORN_TIMEOUT:-60}"
        ;;
    worker)
        exec celery -A config worker \
            --loglevel="${CELERY_LOGLEVEL:-info}" \
            --queues="${CELERY_QUEUES:-ingestion,high,default}" \
            --concurrency="${CELERY_CONCURRENCY:-4}"
        ;;
    beat)
        # Default (file-based PersistentScheduler) keeps beat dependency-free in dev.
        # PROD should install django-celery-beat and set
        # CELERY_BEAT_SCHEDULER=django_celery_beat.schedulers:DatabaseScheduler (§5).
        exec celery -A config beat \
            --loglevel="${CELERY_LOGLEVEL:-info}" \
            ${CELERY_BEAT_SCHEDULER:+--scheduler "$CELERY_BEAT_SCHEDULER"} \
            --schedule /tmp/celerybeat-schedule
        ;;
    *)
        echo "entrypoint.api.sh: unknown ROLE '$ROLE' (expected api|worker|beat)" >&2
        exit 1
        ;;
esac
