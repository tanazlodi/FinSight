"""Email-and-password authentication backed by Supabase Auth.

FinSight sends credentials directly to Supabase Auth. The app never stores
passwords or uses a Supabase service-role key.
"""

from __future__ import annotations

import os
from typing import Any

from supabase import Client, create_client


def is_configured() -> bool:
    """Return whether the public Supabase project credentials are available."""
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_ANON_KEY"))


def client() -> Client:
    """Create a short-lived client so browser sessions never share auth state."""
    url = os.getenv("SUPABASE_URL")
    anon_key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not anon_key:
        raise RuntimeError(
            "Supabase authentication is not configured. Add SUPABASE_URL and "
            "SUPABASE_ANON_KEY to your environment or Streamlit secrets."
        )
    return create_client(url, anon_key)


def sign_up(email: str, password: str) -> Any:
    """Create an account; Supabase sends verification email when enabled."""
    return client().auth.sign_up({"email": email, "password": password})


def sign_in(email: str, password: str) -> Any:
    """Authenticate an existing account and return its Supabase session."""
    return client().auth.sign_in_with_password({"email": email, "password": password})
