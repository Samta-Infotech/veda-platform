"""Access-management URLs, mounted under ``api/v1/`` by ``config/urls.py``.

One module per domain, concatenated here — so adding permissions later means a new
module plus one line, with no risk of touching the routes already in service.
"""
from .catalog import urlpatterns as catalog_urlpatterns
from .grants import urlpatterns as grant_urlpatterns
from .resolver import urlpatterns as resolver_urlpatterns
from .permissions import urlpatterns as permission_urlpatterns
from .roles import urlpatterns as role_urlpatterns
from .users import urlpatterns as user_urlpatterns

urlpatterns = [
    *user_urlpatterns,
    *role_urlpatterns,
    *permission_urlpatterns,
    *catalog_urlpatterns,
    *grant_urlpatterns,
    *resolver_urlpatterns,
]
