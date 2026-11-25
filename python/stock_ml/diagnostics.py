"""Static diagnostic labels only: never return exception messages or DSNs."""
from contextlib import contextmanager
import re


def safe_error(error, stage='database_work'):
    code = getattr(error, 'sqlstate', None)
    code = code if isinstance(code, str) and re.fullmatch(r'[0-9A-Z]{5}', code) else None
    # Inspect locally to classify. Never emit this message into logs.
    message = str(error).lower()
    category, hint = 'database_error', 'Check the project schema and database connection.'
    if code == '28P01' or 'password authentication failed' in message:
        category, hint = 'authentication_failed', 'Check the database password and URL-encode special characters.'
    elif 'tenant or user not found' in message:
        category, hint = 'pooler_identity_invalid', 'Copy this project\'s complete Session pooler URL; verify its username and pooler region.'
    elif 'could not translate host name' in message or 'name or service not known' in message or 'nodename nor servname' in message:
        category, hint = 'hostname_not_found', 'Replace placeholder hosts with the Session pooler hostname copied from Connect.'
    elif 'network is unreachable' in message or 'no route to host' in message:
        category, hint = 'network_unreachable', 'Use the IPv4-compatible Session pooler connection instead of an IPv6-only direct host.'
    elif 'certificate verify failed' in message or 'ssl' in message or 'tls' in message:
        category, hint = 'tls_connection_failed', 'Recopy the official connection string and verify the project is active; do not disable TLS.'
    elif 'timeout' in message or 'timed out' in message:
        category, hint = 'connection_timeout', 'Check the project is active and its Session pooler is reachable.'
    elif 'connection refused' in message:
        category, hint = 'connection_refused', 'Verify the pooler host and Session mode port 5432.'
    elif code in ('42P01', '3F000'):
        category, hint = 'missing_schema', 'Run actions_setup.sql in this assigned stock project.'
    elif code == '42501':
        category, hint = 'database_permission_denied', 'The Actions worker needs the assigned project database worker permissions.'
    elif 'assignment mismatch' in message:
        category, hint = 'project_assignment_mismatch', 'Match each JSON project number to that same sg project; never use the main database.'
    elif stage == 'connect':
        category, hint = 'connection_failed', 'Recopy the complete Session pooler URL for this project, including its own database password.'
    return dict(stage=stage, category=category, sqlstate=code, hint=hint)


class ConnectionStageError(Exception):
    def __init__(self, error):
        self.original = error
        super().__init__('Database connection failed')


@contextmanager
def project_connection(connect, dsn):
    try:
        conn = connect(dsn)
    except Exception as error:
        raise ConnectionStageError(error) from None
    try:
        yield conn
    finally:
        conn.close()
