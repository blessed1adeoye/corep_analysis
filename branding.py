"""
Central branding configuration for COREP Analytics.
Edit once here → applies everywhere.
"""

# ---- Edit these ----
DEVELOPER_NAME   = "omobuwa BLESSED ADEOYE"
DEVELOPER_TITLE  = "Data Scientist & Analytics Engineer, Health Information Management Specialist"
COMPANY_NAME     = "Blessedera Glowtechies Innovative Enterprises / CAC NO. 9156249"
TRADEMARK        = "™"                     # or "®"
YEAR             = "2026"
LOGO_PATH        = "assets/logo.png"                    # e.g. "assets/logo.png" or None

# ---- Derived strings (do not edit) ----
BRAND_LINE       = f"{COMPANY_NAME}{TRADEMARK}"
DEVELOPER_LINE   = f"Developed by {DEVELOPER_NAME} · {DEVELOPER_TITLE}"
COPYRIGHT        = f"© {YEAR} {COMPANY_NAME}{TRADEMARK}. All rights reserved."
POWERED_BY       = f"Powered by {COMPANY_NAME}{TRADEMARK} Data Engineering Pipeline"
APP_TITLE        = f"COREP Outreach Dashboard · {BRAND_LINE}"

# Example output:
#   BRAND_LINE      →  "Acme Analytics™"
#   DEVELOPER_LINE  →  "Developed by Jane Doe · Data Scientist & Analytics Engineer"
#   COPYRIGHT       →  "© 2026 Acme Analytics™. All rights reserved."