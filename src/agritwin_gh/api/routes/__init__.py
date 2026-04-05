"""
Route modules for the AgriTwin-GH FastAPI backend.

Each module registers one resource group:

    dt              → /api/dt/*
    actuators       → /api/actuators/*
    weather         → /api/weather/*
    intelligence    → /api/intelligence/*
    resources       → /api/resources/*
    media           → /api/media/*
    system          → /api/system/*
    greenhouse_3d   → /api/greenhouse-3d/*

All routers are imported by ``agritwin_gh.api.app.create_app()`` and
mounted under the ``/api`` prefix.
"""
