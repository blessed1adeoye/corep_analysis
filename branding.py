"""
Central branding configuration for COREP Analytics.
Edit once here → applies everywhere.
"""

# ============================================================
# EDIT THESE
# ============================================================
DEVELOPER_NAME        = "Omobuwa Blessed Adeoye"

DEVELOPER_TITLE       = ("Data Scientist & Data Engineer · "
                         "Health Information Management Specialist · "
                         "Machine Learning & AI Engineer")

DEVELOPER_TITLE_SHORT = "AI/ML Engineer · Health Informatics"

COMPANY_NAME          = "Blessedera Glowtechies Innovative Enterprises"
COMPANY_REG           = "CAC NO. 9156249"
TRADEMARK             = "™"
YEAR                  = "2026"
LOGO_PATH             = None      # set to "assets/logo.png" if the file exists

# ============================================================
# Derived strings (do not edit)
# ============================================================
BRAND_LINE            = f"{COMPANY_NAME}{TRADEMARK}"
BRAND_LINE_FULL       = f"{COMPANY_NAME}{TRADEMARK} ({COMPANY_REG})"

DEVELOPER_LINE        = f"Developed by {DEVELOPER_NAME}"
DEVELOPER_LINE_FULL   = f"Developed by {DEVELOPER_NAME} · {DEVELOPER_TITLE}"
DEVELOPER_LINE_SHORT  = f"{DEVELOPER_NAME} · {DEVELOPER_TITLE_SHORT}"

COPYRIGHT             = f"© {YEAR} {COMPANY_NAME}{TRADEMARK}. All rights reserved."
POWERED_BY            = f"Powered by {COMPANY_NAME}{TRADEMARK} Data Engineering Pipeline"
APP_TITLE             = f"COREP Medical Outreach Dashboard · {BRAND_LINE}"
WATERMARK             = f"{COMPANY_NAME}{TRADEMARK} · {DEVELOPER_NAME}"