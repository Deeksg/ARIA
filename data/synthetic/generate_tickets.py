# =============================================================================
# ARIA — Synthetic Ticket Generator
# =============================================================================
# WHY THIS FILE EXISTS:
# Real SAP support ticket data is confidential enterprise operational data.
# No public dataset exists. This generator creates 25,000 realistic synthetic
# tickets using LLM-assisted augmentation — a standard technique in enterprise
# ML where domain-specific labeled data is unavailable.
#
# WHAT IT PRODUCES:
# 1. 15,000 incident tickets  — real SAP problems, some with error codes,
#                               some without, in non-technical user language
# 2.  6,500 development requests — new feature/build requests
# 3.  3,500 edge cases        — ambiguous, cross-module, unusual formats
#
# HOW IT WORKS:
# Step 1 — We define seed templates for each module and issue type
# Step 2 — We call LLM API (Google Gemini) to generate realistic variations of each seed
# Step 3 — We save everything to two CSV files:
#           - incoming_tickets.csv  (messy, unstructured, what users write)
#           - resolved_tickets.csv  (structured, what consultants document)
# =============================================================================

import os
import re
import csv
import sys
import json
import time
import math
import signal
import random
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv  # type: ignore

# Load environment variables from .env file into os.environ
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
# Read API key from environment variable — never hardcode API keys in code
# GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
# GROQ_FAST_MODEL = os.environ.get("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
# GROQ_BASE_URL  = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
# GROQ_URL       = f"{GROQ_BASE_URL.rstrip('/')}/chat/completions"

# Google Gemini API configuration
# API key — get a free key at aistudio.google.com
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")

# Model name — configurable via environment variable
GEMINI_MODEL    = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Base URL — configurable via environment variable, not hardcoded
# Default is Google's OpenAI-compatible Gemini endpoint
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
GEMINI_URL      = f"{GEMINI_BASE_URL.rstrip('/')}/chat/completions"

# Output file paths
INCOMING_CSV  = "data/synthetic/incoming_tickets.csv"
RESOLVED_CSV  = "data/synthetic/resolved_tickets.csv"

# Progress checkpoint file — gitignored, runtime artifact only
# Allows the script to resume from the last completed seed on restart
PROGRESS_FILE = "data/synthetic/generation_progress.json"

# Target counts & multi-pass configuration
TARGET_INCIDENTS   = 15000
TARGET_DEVELOPMENT = 6500
TARGET_EDGE_CASES  = 3500
VARIATIONS_PER_PASS = 20   # Number of variations requested per API call

# =============================================================================
# SAP MODULES
# Each module represents a functional area of SAP.
# FI = Finance, MM = Materials Management, SD = Sales & Distribution,
# PP = Production Planning, BASIS = System Administration,
# ABAP = Custom Development, HR = Human Resources
# =============================================================================
MODULES = ["FI", "MM", "SD", "PP", "BASIS", "ABAP", "HR"]

# =============================================================================
# ISSUE TYPES
# These are the categories of problems that occur in SAP.
# A consultant needs to know which type of issue this is immediately
# because each type has a completely different investigation approach.
# =============================================================================
ISSUE_TYPES = [
    "Authorization",      # User doesn't have permission to do something
    "Configuration",      # System is set up incorrectly
    "Master Data",        # Wrong or missing data records (vendors, materials etc)
    "Performance",        # System is slow or timing out
    "Integration",        # Two SAP modules or external systems not talking correctly
    "Data Inconsistency", # Data doesn't match between tables or systems
    "User Error",         # User is doing something wrong, needs guidance
    "System Error",       # Technical crash or unexpected system behaviour
]

# =============================================================================
# SAP ERROR CODES BY MODULE
# Real SAP error codes follow the pattern: 2-letter prefix + space + number
# These are actual SAP error code formats used in real systems.
# We include these so some tickets look technical (with error codes) and
# some look non-technical (without) — mirroring real world variety.
# =============================================================================
ERROR_CODES = {
    "FI":    ["F5 805", "F5 001", "F5 702", "FF 747", "F0 151",
              "FB 422", "FI 130", "F5 307"],
    "MM":    ["ME 343", "ME 062", "ME 803", "M7 021", "M8 108",
              "ME 395", "MM 024", "M3 186"],
    "SD":    ["VL 032", "V1 390", "V2 395", "VF 044", "SD 512",
              "V1 032", "VB 043", "VF 051"],
    "PP":    ["CO 624", "CO 347", "CO 413", "PP 102", "CO 530",
              "PP 038", "CO 188", "PP 217"],
    "BASIS": ["SY 001", "SY 212", "RFC 102", "BC 412", "SY 399",
              "AB 110", "SY 104", "TS 105"],
    "ABAP":  ["AB 185", "AB 075", "AB 217", "SR 044", "AB 301",
              "AB 022", "AB 154", "AB 406"],
    "HR":    ["PY 602", "PA 406", "PY 031", "HR 143", "PY 288",
              "PA 212", "PY 115", "HR 077"],
}

# =============================================================================
# T-CODES BY MODULE
# T-codes (transaction codes) are how consultants navigate SAP.
# Each T-code opens a specific screen or function.
# These are included in resolved tickets so the resolution agent
# learns which T-codes are relevant to which modules and issues.
# =============================================================================
TCODES = {
    "FI":    ["FB60", "FB70", "F-43", "F-32", "FBL1N", "FBL5N",
              "MIRO", "F110", "FB50", "FS10N"],
    "MM":    ["ME21N", "ME22N", "ME23N", "MIGO", "MB51", "MB52",
              "ME2M", "MM60", "ME51N", "MMBE"],
    "SD":    ["VA01", "VA02", "VA03", "VL01N", "VF01", "VF02",
              "VL02N", "VKM3", "VD01", "VD02"],
    "PP":    ["CO01", "CO02", "CO03", "MD01", "MD04", "CO11N",
              "CO15", "PP01", "MD02", "CO24"],
    "BASIS": ["SU01", "SU10", "PFCG", "SM50", "SM51", "ST05",
              "SU53", "RZ10", "SM37", "AL08"],
    "ABAP":  ["SE38", "SE80", "SE37", "SE16N", "ST22", "SM21",
              "SE11", "SNOTE", "SE24", "SE91"],
    "HR":    ["PA30", "PA40", "PA20", "PT60", "PY60", "PE03",
              "PT61", "PA10", "PU01", "PT50"],
}

# =============================================================================
# SEED TEMPLATES — INCIDENT TICKETS
# These are carefully written examples that represent the range of
# real SAP support ticket language.
#
# IMPORTANT: These are written in MESSY, NON-TECHNICAL user language
# intentionally. Real users write like this. BERT must learn to classify
# vague, panicked, poorly written text — not clean technical descriptions.
#
# Structure: (module, issue_type, has_error_code, seed_text)
# has_error_code = True means we will insert an error code into the text
# has_error_code = False means pure plain language, no error code
# =============================================================================
INCIDENT_SEEDS = [
    # ── FI (Finance) ──────────────────────────────────────────────────────
    ("FI", "Authorization",      False,
     "user cannot post invoice, says no authorization, please check"),
    ("FI", "Authorization",      True,
     "getting {error} when trying to post vendor invoice, checked with user and they cannot proceed"),
    ("FI", "Configuration",      False,
     "payment terms not showing correctly in vendor master, amount calculating wrong"),
    ("FI", "Configuration",      True,
     "company code 1000 giving {error} during document posting, was working fine yesterday"),
    ("FI", "Master Data",        False,
     "vendor account blocked, client says they never blocked it, please help urgent"),
    ("FI", "Master Data",        True,
     "GL account not found error {error} when creating journal entry, account exists in chart of accounts"),
    ("FI", "System Error",       True,
     "month end closing failing with {error}, finance team cannot close the period"),
    ("FI", "Data Inconsistency", False,
     "invoice amount in system not matching what vendor sent, difference of small amount but blocking payment"),
    ("FI", "User Error",         False,
     "user trying to reverse document but option is greyed out, how to reverse posted document"),
    ("FI", "Performance",        False,
     "FBL1N report taking very long to load, client says it used to be fast, now hanging"),

    # ── MM (Materials Management) ─────────────────────────────────────────
    ("MM", "Authorization",      False,
     "warehouse user cannot do goods receipt, authorization error coming"),
    ("MM", "Authorization",      True,
     "movement type 101 giving {error} for goods receipt, user could do this last week"),
    ("MM", "Configuration",      False,
     "purchase order not releasing even after all approvers approved, stuck in workflow"),
    ("MM", "Configuration",      True,
     "goods receipt posting failing with {error}, plant 1000 issue only"),
    ("MM", "Master Data",        False,
     "material master not showing in purchase order, material exists in system"),
    ("MM", "Master Data",        True,
     "vendor not found when creating PO, {error} showing, vendor was created last month"),
    ("MM", "Integration",        False,
     "goods receipt done but stock not updating in warehouse management, showing zero"),
    ("MM", "Data Inconsistency", True,
     "PO quantity and GR quantity not matching in report, {error} when reconciling"),
    ("MM", "System Error",       False,
     "MIGO transaction crashing when trying to post, happening to all warehouse users"),
    ("MM", "User Error",         False,
     "user accidentally posted goods receipt twice, how to reverse one of them"),

    # ── SD (Sales and Distribution) ───────────────────────────────────────
    ("SD", "Authorization",      False,
     "sales user cannot create delivery, says missing authorization for shipping point"),
    ("SD", "Authorization",      True,
     "billing document creation giving {error}, sales team cannot invoice customer"),
    ("SD", "Configuration",      False,
     "pricing not calculating correctly on sales order, discount not applying"),
    ("SD", "Configuration",      True,
     "delivery cannot be posted, {error} for goods issue, plant configuration issue maybe"),
    ("SD", "Master Data",        False,
     "customer account blocked for orders, customer says they paid everything"),
    ("SD", "Integration",        False,
     "sales order created but not visible in warehouse for delivery, integration issue"),
    ("SD", "System Error",       True,
     "VA01 crashing with {error} when trying to save sales order with specific material"),
    ("SD", "Data Inconsistency", False,
     "invoice amount different from sales order amount, no manual changes made"),
    ("SD", "User Error",         False,
     "user wants to cancel invoice that is already accounting document posted, how to do"),
    ("SD", "Performance",        True,
     "sales order report very slow, {error} timeout happening for large customers"),

    # ── PP (Production Planning) ──────────────────────────────────────────
    ("PP", "Authorization",      False,
     "production operator cannot confirm production order, no authorization error"),
    ("PP", "Configuration",      True,
     "material requirements planning giving {error}, MRP run failing for plant 2000"),
    ("PP", "Master Data",        False,
     "work center not available when creating production order, shows deleted but is active"),
    ("PP", "Integration",        True,
     "production order confirmed but inventory not updated, {error} in stock posting"),
    ("PP", "System Error",       False,
     "CO01 transaction freezing when saving production order with large BOM"),
    ("PP", "Data Inconsistency", False,
     "planned quantity and actual quantity not matching in production report"),
    ("PP", "User Error",         False,
     "user closed production order without goods issue, how to reopen closed order"),
    ("PP", "Performance",        True,
     "MRP run taking too long {error}, used to finish in 2 hours now taking all night"),

    # ── BASIS (System Administration) ─────────────────────────────────────
    ("BASIS", "Authorization",   False,
     "new user created but cannot log in to SAP, password correct but access denied"),
    ("BASIS", "Authorization",   True,
     "user getting {error} when accessing transaction, role assigned but still error"),
    ("BASIS", "Configuration",   False,
     "system performance very slow for all users since yesterday morning, no changes made"),
    ("BASIS", "Configuration",   True,
     "background job failing with {error}, was running fine for months"),
    ("BASIS", "System Error",    False,
     "SAP not starting after weekend, basis team checked server seems fine"),
    ("BASIS", "System Error",    True,
     "dump {error} happening randomly for multiple users, no pattern found yet"),
    ("BASIS", "Performance",     False,
     "system response time very slow only during lunch hours, all transactions affected"),
    ("BASIS", "User Error",      False,
     "user locked out of system after too many wrong password attempts, need to unlock"),

    # ── ABAP (Development) ────────────────────────────────────────────────
    ("ABAP", "System Error",     True,
     "custom report giving dump {error}, was working fine until system update"),
    ("ABAP", "System Error",     False,
     "user exit not triggering in purchase order, customization seems broken"),
    ("ABAP", "Configuration",    True,
     "background job for custom program failing with {error}, code not changed"),
    ("ABAP", "Data Inconsistency", False,
     "custom report showing different data than standard SAP report for same selection"),
    ("ABAP", "Performance",      False,
     "custom report taking very long, used to be fast, no code changes recently"),
    ("ABAP", "Integration",      True,
     "RFC call to external system failing with {error}, connection was working yesterday"),

    # ── HR (Human Resources) ──────────────────────────────────────────────
    ("HR", "Authorization",      False,
     "HR user cannot access employee salary data, says no authorization"),
    ("HR", "Authorization",      True,
     "payroll run giving {error}, HR admin cannot execute monthly payroll"),
    ("HR", "Configuration",      False,
     "leave quota not calculating correctly, employees showing wrong balance"),
    ("HR", "Master Data",        False,
     "employee not showing in payroll, was added last month but payroll not picking up"),
    ("HR", "System Error",       True,
     "time evaluation failing with {error}, affects all employees in plant 1000"),
    ("HR", "Data Inconsistency", False,
     "attendance data in time system not matching HR master data, creating payroll issues"),
]

# =============================================================================
# SEED TEMPLATES — NEW DEVELOPMENT REQUESTS
# These are requests to build something new in SAP.
# They have very consistent language patterns: "please create", "we need",
# "can you build", "new report", "enhancement" etc.
# DistilBERT learns to separate these from incident tickets.
# =============================================================================
DEVELOPMENT_SEEDS = [
    ("FI",    "please create a new report showing all open vendor invoices by aging bucket"),
    ("FI",    "we need a new approval workflow for journal entries above 100000 rupees"),
    ("MM",    "can you develop a new purchase order report with vendor performance metrics"),
    ("MM",    "we need an enhancement to the goods receipt screen to add a custom field for inspection status"),
    ("SD",    "please build a new customer credit limit report by sales region"),
    ("SD",    "we need a new pricing procedure for special discount customers, current one does not support this"),
    ("PP",    "can you create a new production planning report showing machine utilization by shift"),
    ("PP",    "we need a custom BOM explosion report that shows material costs at each level"),
    ("BASIS", "please create a new user access report showing all roles and authorizations by department"),
    ("BASIS", "we need automated monitoring for system performance and email alerts when threshold is crossed"),
    ("ABAP",  "can you develop an interface to send purchase order data to our external logistics system"),
    ("ABAP",  "we need a new custom table to store additional vendor classification data"),
    ("HR",    "please create a new headcount report by department showing joining and leaving trends"),
    ("HR",    "we need an enhancement to show manager hierarchy on the employee master screen"),
    ("FI",    "new report required for auditors showing all reversals and the reason codes"),
    ("MM",    "we need a custom transaction to bulk update material prices from an excel upload"),
    ("SD",    "please develop a new order confirmation email that automatically sends to customer on sales order save"),
    ("ABAP",  "we need a workflow for capital expenditure approvals, current process is all manual on paper"),
]

# =============================================================================
# SEED TEMPLATES — EDGE CASES
# These are deliberately ambiguous, unusual, or tricky tickets.
# They test the model's ability to handle real-world messiness.
# Types of edge cases:
# - Ambiguous: could be incident OR development request
# - Cross-module: touches two modules at once
# - Very short: almost no information
# - Very vague: so unclear a human would need to ask follow-up questions
# - Mixed language signals: sounds like development but is actually an incident
# =============================================================================
EDGE_CASE_SEEDS = [
    # Ambiguous — could be incident or development request
    ("FI",          "Authorization", False,
     "invoice process not working as expected, need to fix or change something"),
    ("MM",          "Configuration", False,
     "purchase order system needs improvement, many users complaining"),

    # Cross-module — touches two areas at once
    ("FI",          "Integration",   True,
     "goods receipt posted in MM but {error} in FI when trying to post invoice, cross module issue"),
    ("SD",          "Integration",   False,
     "sales order delivered but finance not getting the revenue, MM and SD both checked separately seems fine"),

    # Very short tickets — almost no information
    ("BASIS",       "System Error",  False,
     "system down please help"),
    ("FI",          "Authorization", False,
     "cannot post"),
    ("MM",          "System Error",  True,
     "error {error} please urgent"),

    # Very vague
    ("SD",          "Configuration", False,
     "something wrong with customer orders since last week, users saying many things not working"),
    ("PP",          "Data Inconsistency", False,
     "production numbers not right, manager is asking questions, need to fix fast"),

    # Sounds like development but is actually an incident
    ("HR",          "Configuration", False,
     "we need the leave system to work properly, it was calculating correctly before"),
    ("ABAP",        "System Error",  False,
     "we need the custom report to run without errors like it used to"),

    # Tickets with unusual formatting
    ("FI",          "Master Data",   False,
     "VENDOR MASTER BLOCKED - URGENT - CLIENT ESCALATING - PLEASE CHECK IMMEDIATELY"),
    ("MM",          "Authorization", True,
     "error: {error} | user: john.smith | transaction: MIGO | plant: 1000 | please fix"),

    # Cross-module with error codes
    ("SD",          "Integration",   True,
     "billing run giving {error}, checked SD config seems fine but FI posting not happening"),
    ("MM",          "Integration",   True,
     "warehouse transfer order not confirming, {error} in WM, MM stock shows correct"),
]

# =============================================================================
# RESOLUTION TEMPLATES
# These are structured resolution records for the RAG knowledge base.
# When a consultant resolves a ticket in production, they write something
# like this. For the demo, we generate these synthetically.
#
# The structure is important — the LangGraph agent reads these to understand
# what fixes work for what problems. Richer structure = better suggestions.
# =============================================================================
RESOLUTION_TEMPLATES = {

    # =========================================================================
    # FI — FINANCE
    # =========================================================================

    ("FI", "Authorization"): {
        "root_cause":    "Missing authorization object F_BKPF_BUK or F_BKPF_BLA for the company code or document type. User role does not include posting authorization for this transaction.",
        "fix":           "1. Ask user to run SU53 immediately after the error and share the screenshot\n2. Identify the missing authorization object from SU53 output\n3. Open PFCG and find the user's assigned role\n4. Add the missing authorization object with correct company code and activity values\n5. Generate the authorization profile\n6. Transport role to production if needed\n7. Ask user to retest",
        "tcodes_used":   ["SU53", "PFCG", "SU01", "FB60"],
        "resolution_time_minutes": 45,
    },

    ("FI", "Configuration"): {
        "root_cause":    "Missing or incorrect configuration in financial accounting — posting period not open, document type not configured, or account determination missing for the transaction.",
        "fix":           "1. Check if posting period is open in OB52 for the company code and period\n2. If period closed, open it temporarily or ask finance manager to approve period opening\n3. Check document type configuration in OBA7\n4. Verify account determination in FBKP\n5. Check number range assignment in FBN1\n6. Save configuration and retest with user",
        "tcodes_used":   ["OB52", "OBA7", "FBKP", "FBN1", "OBY6"],
        "resolution_time_minutes": 90,
    },

    ("FI", "Master Data"): {
        "root_cause":    "Vendor, customer, or GL account master data issue — account blocked, mandatory fields incomplete, or account not extended to the relevant company code.",
        "fix":           "1. Check the account in FK03 (vendor) or FD03 (customer) or FS03 (GL)\n2. Look for blocking indicators in the General Data and Company Code tabs\n3. Check if account is extended to the correct company code\n4. Remove incorrect blocks using FK02 or FD02 or FS00\n5. Complete any mandatory fields that are missing\n6. Save and ask user to retest",
        "tcodes_used":   ["FK03", "FK02", "FD03", "FD02", "FS03", "FS00"],
        "resolution_time_minutes": 30,
    },

    ("FI", "System Error"): {
        "root_cause":    "Financial posting failure due to system-level error — often caused by incomplete configuration, missing exchange rates, or period-end locking.",
        "fix":           "1. Check SM21 system log for the exact error timestamp\n2. Review ST22 for any ABAP dumps related to the transaction\n3. Check if exchange rates are maintained in OB08 if foreign currency involved\n4. Verify that the fiscal year variant is correctly assigned in OBY6\n5. Check if any background jobs are locking the relevant tables in SM12\n6. Clear locks if present and retest",
        "tcodes_used":   ["SM21", "ST22", "OB08", "OBY6", "SM12"],
        "resolution_time_minutes": 75,
    },

    ("FI", "Data Inconsistency"): {
        "root_cause":    "Data mismatch between FI and a feeder module (MM or SD) — typically caused by incomplete postings, cancelled documents with open items, or incorrect clearing.",
        "fix":           "1. Run FBL1N or FBL3N to identify open items and inconsistencies\n2. Check if corresponding MM or SD documents are correctly posted\n3. Run F.13 for automatic clearing of matched items\n4. For GR/IR mismatches run MR11 to clear GR/IR account\n5. Document all findings and have finance team review before clearing\n6. Rerun the affected report to confirm consistency",
        "tcodes_used":   ["FBL1N", "FBL3N", "F.13", "MR11", "FS10N"],
        "resolution_time_minutes": 120,
    },

    ("FI", "User Error"): {
        "root_cause":    "User performed an incorrect action — wrong document type, wrong posting key, or attempted an operation not permitted on already-posted documents.",
        "fix":           "1. Review exactly what the user did by checking the document in FB03\n2. If document needs reversal use FB08 for individual reversal or F.80 for mass reversal\n3. Check if reversal is blocked due to period being closed — open period temporarily if needed\n4. Guide user through the correct process step by step\n5. Document the correct procedure for the user to follow next time",
        "tcodes_used":   ["FB03", "FB08", "F.80", "OB52"],
        "resolution_time_minutes": 40,
    },

    ("FI", "Performance"): {
        "root_cause":    "FI report or transaction running slowly due to missing database indexes, large data volume without selection filters, or table buffering issues.",
        "fix":           "1. Check what selection parameters the user is using — wide date ranges cause slowness\n2. Advise user to narrow selection criteria (company code, date range, account range)\n3. Run ST05 SQL trace during the slow transaction to identify the bottleneck query\n4. Check if relevant database indexes exist using SE11\n5. Escalate to BASIS team if index is missing or database optimization is needed\n6. Check if archiving is needed for old FI documents",
        "tcodes_used":   ["ST05", "SE11", "FBL1N", "SM50"],
        "resolution_time_minutes": 90,
    },

    ("FI", "Integration"): {
        "root_cause":    "Integration issue between FI and MM or SD — goods receipt not generating FI document, invoice not matched to PO, or billing document not creating accounting entry.",
        "fix":           "1. Check the document flow in the relevant MM or SD transaction\n2. Verify account determination configuration in OBYC (for MM) or VKOA (for SD)\n3. Check if the GL accounts are correctly assigned for the transaction/event key\n4. Review posting keys and document types for the integration\n5. Check if any user exit or enhancement is blocking the posting\n6. Test with a new document to confirm fix",
        "tcodes_used":   ["OBYC", "VKOA", "MR8M", "VF02", "FB03"],
        "resolution_time_minutes": 100,
    },

    # =========================================================================
    # MM — MATERIALS MANAGEMENT
    # =========================================================================

    ("MM", "Authorization"): {
        "root_cause":    "Missing authorization for movement type, plant, or storage location in MM. Common missing objects are M_MSEG_BWA (movement type) or M_MSEG_WMB (plant/storage location).",
        "fix":           "1. Ask user to run SU53 after the authorization error\n2. Identify missing authorization object — usually M_MSEG_BWA or M_MSEG_WMB\n3. Open PFCG and edit the user's warehouse or procurement role\n4. Add missing object with correct plant, storage location, and movement type values\n5. Generate authorization profile and save\n6. Reassign role to user in SU01 if needed\n7. Ask user to retest MIGO transaction",
        "tcodes_used":   ["SU53", "PFCG", "SU01", "MIGO"],
        "resolution_time_minutes": 40,
    },

    ("MM", "Configuration"): {
        "root_cause":    "Purchase order or goods receipt configuration missing — document type not configured, number range exhausted, tolerance limits not set, or release strategy not correctly defined.",
        "fix":           "1. Check document type configuration in OMH6 for PO or OMBA for GR\n2. Verify number ranges are not exhausted in OMH7\n3. Check tolerance limits in OMR6 for invoice verification\n4. Review release strategy configuration in OMGS if PO not releasing\n5. Check plant parameters in OPPQ if plant-specific issue\n6. Save and transport configuration changes\n7. Retest with user",
        "tcodes_used":   ["OMH6", "OMH7", "OMR6", "OMGS", "SPRO"],
        "resolution_time_minutes": 80,
    },

    ("MM", "Master Data"): {
        "root_cause":    "Material or vendor master data incomplete — material not extended to plant or storage location, vendor not extended to purchasing organization, or mandatory views missing.",
        "fix":           "1. Check material master in MM03 — verify all required views exist for the plant\n2. If plant view missing extend material using MM01 and select missing organizational levels\n3. Check vendor master in MK03 for purchasing organization data\n4. Extend vendor to purchasing org using MK01 if needed\n5. Verify info record exists in ME11 if price not found\n6. Check source list in ME01 if source determination failing",
        "tcodes_used":   ["MM03", "MM01", "MK03", "MK01", "ME11", "ME01"],
        "resolution_time_minutes": 35,
    },

    ("MM", "Integration"): {
        "root_cause":    "Integration failure between MM and FI or WM — goods receipt not updating stock, invoice not matched to PO, or warehouse transfer not confirming.",
        "fix":           "1. Check the document flow from PO through GR to invoice in ME23N\n2. Verify account determination in OBYC for the transaction/event key\n3. Check if GR/IR clearing account is correctly configured\n4. If WM integration issue check warehouse movement type configuration in LE-WM\n5. Check if any enhancement or user exit is blocking the integration step\n6. Review SM21 logs around the time of the failure for system messages",
        "tcodes_used":   ["ME23N", "OBYC", "MB51", "LT0A", "SM21"],
        "resolution_time_minutes": 90,
    },

    ("MM", "Data Inconsistency"): {
        "root_cause":    "Stock or value inconsistency between MM stock ledger and FI balance sheet — caused by incomplete postings, cancelled documents, or incorrect movement types used.",
        "fix":           "1. Run MB52 to check current stock levels by plant and storage location\n2. Run MB51 to see all material movements and identify the inconsistent posting\n3. Compare with FI GL balance using FS10N for the stock account\n4. Run MR21 if price differences need correction\n5. For GR/IR mismatches run MR11\n6. Document all findings before making corrections\n7. Have both MM and FI teams sign off on the correction",
        "tcodes_used":   ["MB52", "MB51", "FS10N", "MR21", "MR11"],
        "resolution_time_minutes": 110,
    },

    ("MM", "System Error"): {
        "root_cause":    "MM transaction crashing or failing at system level — often caused by ABAP dump, database lock, or incomplete transport causing inconsistent customizing.",
        "fix":           "1. Check ST22 for the ABAP dump — note the program name and line number\n2. Check SM21 system log for errors around the same timestamp\n3. Check SM12 for database locks on MM tables (EKKO, EKPO, MSEG)\n4. Clear any old locks that are blocking the transaction\n5. If dump is in a custom program check SE38 for recent changes\n6. If standard SAP program check SNOTE for relevant OSS notes\n7. Apply note if available and retest",
        "tcodes_used":   ["ST22", "SM21", "SM12", "SE38", "SNOTE"],
        "resolution_time_minutes": 85,
    },

    ("MM", "User Error"): {
        "root_cause":    "User performed incorrect action in MM — wrong movement type used, posted to wrong plant, or attempted to change a document that is already in a completed status.",
        "fix":           "1. Review what the user did by checking the material document in MB03\n2. If goods movement needs reversal use MIGO with movement type 102 (for 101 reversal)\n3. Check if the original document can still be reversed based on period\n4. If PO needs correction check if it can still be changed in ME22N\n5. Guide user through the correct process\n6. Add notes to the document explaining the correction",
        "tcodes_used":   ["MB03", "MIGO", "ME22N", "ME23N"],
        "resolution_time_minutes": 35,
    },

    ("MM", "Performance"): {
        "root_cause":    "MM report or transaction slow due to large data volume, missing database index, or inefficient selection criteria.",
        "fix":           "1. Check the selection parameters — large plant ranges or date ranges cause slowness\n2. Advise user to add more restrictive filters (plant, material group, date range)\n3. Run ST05 SQL trace to identify slow database queries\n4. Check if background processing is available for the report\n5. Schedule report as background job in SM36 if data volume is too large for online run\n6. Escalate to BASIS if database index is missing",
        "tcodes_used":   ["ST05", "SM36", "SM37", "MB52"],
        "resolution_time_minutes": 60,
    },

    # =========================================================================
    # SD — SALES AND DISTRIBUTION
    # =========================================================================

    ("SD", "Authorization"): {
        "root_cause":    "Missing authorization for sales organization, distribution channel, or division. Common missing objects are V_VBAK_AAT (order type) or V_VBAK_VKO (sales area).",
        "fix":           "1. Run SU53 for the affected user after the error\n2. Identify missing authorization object — typically V_VBAK_AAT or V_VBAK_VKO\n3. Open PFCG and edit the sales user role\n4. Add missing object with correct sales organization, distribution channel, and division\n5. Generate profile and reassign role\n6. Ask user to retest VA01 or VL01N",
        "tcodes_used":   ["SU53", "PFCG", "SU01", "VA01"],
        "resolution_time_minutes": 40,
    },

    ("SD", "Configuration"): {
        "root_cause":    "SD configuration gap — pricing procedure not assigned, shipping point not determined, output type not configured, or delivery document type missing.",
        "fix":           "1. Check pricing procedure assignment in OVKK for the sales area\n2. Verify shipping point determination in OVL2 if delivery issue\n3. Check output determination in V/40 if output not generating\n4. Review delivery document type configuration in OVLK\n5. Check copy control between order and delivery in VTLA\n6. Save and transport configuration\n7. Retest end to end",
        "tcodes_used":   ["OVKK", "OVL2", "V/40", "OVLK", "VTLA"],
        "resolution_time_minutes": 95,
    },

    ("SD", "Master Data"): {
        "root_cause":    "Customer, material, or condition master data issue — customer blocked, material not extended to sales organization, or pricing condition missing.",
        "fix":           "1. Check customer master in VD03 for sales area data and blocking indicators\n2. Remove customer block in VD02 if incorrectly set\n3. Check material master in MM03 — verify Sales Org and Sales General views exist\n4. Extend material to sales organization if missing using MM01\n5. Check condition records in VK13 if pricing not determined\n6. Create missing condition record in VK11 if needed",
        "tcodes_used":   ["VD03", "VD02", "MM03", "MM01", "VK13", "VK11"],
        "resolution_time_minutes": 40,
    },

    ("SD", "Integration"): {
        "root_cause":    "Integration failure between SD and MM or FI — delivery not reducing stock, billing not creating accounting document, or credit check blocking orders incorrectly.",
        "fix":           "1. Check document flow in VA03 to see where the chain breaks\n2. If goods issue not posting check MM plant and storage location settings\n3. If billing not creating FI document check account determination in VKOA\n4. If credit check blocking review credit limit in FD32\n5. Check copy control settings between relevant document types\n6. Review any enhancement or user exit affecting the integration",
        "tcodes_used":   ["VA03", "VF02", "VKOA", "FD32", "VL02N"],
        "resolution_time_minutes": 100,
    },

    ("SD", "System Error"): {
        "root_cause":    "SD transaction crashing with ABAP dump — often caused by missing configuration, bad master data combination, or recent transport introducing a defect.",
        "fix":           "1. Get exact error details from ST22 — note program and line number\n2. Check if error is reproducible with the same data\n3. Check SM21 logs for system-level messages\n4. If standard SAP program search for OSS notes in SNOTE\n5. Check if any transports were imported recently that might have caused regression\n6. If custom code involved check SE38 for recent changes",
        "tcodes_used":   ["ST22", "SM21", "SNOTE", "SE38", "STMS"],
        "resolution_time_minutes": 80,
    },

    ("SD", "Data Inconsistency"): {
        "root_cause":    "Data mismatch in SD — invoice amount different from order, quantity discrepancy between delivery and billing, or document status inconsistent.",
        "fix":           "1. Check document flow in VA03 to compare quantities and amounts at each step\n2. Review copy control configuration in VTFL for any price redetermination settings\n3. Check if manual changes were made to pricing in the order\n4. Run VF04 to check billing due list for inconsistencies\n5. If already billed and incorrect issue a credit memo using VA01 with order type CR\n6. Document the root cause clearly before making corrections",
        "tcodes_used":   ["VA03", "VF04", "VTFL", "VA01"],
        "resolution_time_minutes": 75,
    },

    ("SD", "User Error"): {
        "root_cause":    "User made an incorrect entry — wrong order type, wrong customer, or attempted to change a completed document.",
        "fix":           "1. Review the document in VA03 to understand what was done incorrectly\n2. If order not yet delivered it can be rejected using reason for rejection\n3. If delivery already posted goods issue needs to be reversed in VL09\n4. If billing already done issue credit memo in VA01 with order type CR\n5. Guide user through the correct process\n6. Consider adding a note in the document explaining the correction",
        "tcodes_used":   ["VA03", "VA02", "VL09", "VA01"],
        "resolution_time_minutes": 50,
    },

    ("SD", "Performance"): {
        "root_cause":    "SD report or list transaction running slowly due to large data volume or missing index on SD tables.",
        "fix":           "1. Check the selection parameters — reduce date range and add sales organization filter\n2. Run ST05 to identify the slow database query\n3. Check if the report can be run in background using SM36\n4. For VKM3 or credit reports check if credit data needs reorganization\n5. Escalate to BASIS for database analysis if query is on standard tables",
        "tcodes_used":   ["ST05", "SM36", "SM37", "VA05"],
        "resolution_time_minutes": 55,
    },

    # =========================================================================
    # PP — PRODUCTION PLANNING
    # =========================================================================

    ("PP", "Authorization"): {
        "root_cause":    "Missing authorization for production order confirmation, goods movement in PP, or MRP execution. Common missing object is C_AFKO_AWK for order type and plant.",
        "fix":           "1. Run SU53 after the authorization error\n2. Identify missing object — typically C_AFKO_AWK or C_AFKO_AUF\n3. Open PFCG and edit the production user role\n4. Add authorization object with correct plant, order type, and activity values\n5. Generate profile and reassign role\n6. Ask user to retest CO11N or CO01",
        "tcodes_used":   ["SU53", "PFCG", "SU01", "CO11N"],
        "resolution_time_minutes": 35,
    },

    ("PP", "Configuration"): {
        "root_cause":    "Production or MRP configuration gap — scheduling parameters missing, order type not configured for plant, or MRP parameters not maintained for material.",
        "fix":           "1. Check order type dependent parameters in OPL8 for the plant and order type\n2. Verify scheduling parameters in OP45\n3. Check MRP parameters in plant settings OPPQ\n4. Review work center configuration in CR01 if scheduling issue\n5. Check BOM usage and routing assignment if order creation failing\n6. Save configuration and retest",
        "tcodes_used":   ["OPL8", "OP45", "OPPQ", "CR01", "SPRO"],
        "resolution_time_minutes": 100,
    },

    ("PP", "Master Data"): {
        "root_cause":    "Work center, BOM, routing, or material master data missing or incomplete for the plant and production version.",
        "fix":           "1. Check if work center exists for the plant in CR03\n2. Verify BOM exists for the material and plant in CS03\n3. Check routing exists in CA03\n4. Verify production version in MM03 MRP views\n5. Check if material master has correct MRP type and lot sizing procedure\n6. Create or extend missing master data and retest order creation",
        "tcodes_used":   ["CR03", "CS03", "CA03", "MM03", "CO01"],
        "resolution_time_minutes": 60,
    },

    ("PP", "Integration"): {
        "root_cause":    "Integration failure between PP and MM or FI — goods issue for production order not updating stock, order confirmation not creating FI document, or MRP not reading correct inventory.",
        "fix":           "1. Check the production order status in CO03 — look for MSPT or GMPS flags\n2. Verify goods movement configuration for the order type in OPJK\n3. Check account determination in OKB9 for PP cost postings\n4. Review settlement profile in OKO7 if cost settlement failing\n5. Check MM stock in MMBE to verify inventory exists for goods issue\n6. Run CO24 to check missing parts if goods issue blocked",
        "tcodes_used":   ["CO03", "OPJK", "OKB9", "CO24", "MMBE"],
        "resolution_time_minutes": 90,
    },

    ("PP", "System Error"): {
        "root_cause":    "Production order or MRP transaction failing with system error — often database lock, ABAP dump, or MRP inconsistency in planning data.",
        "fix":           "1. Check ST22 for ABAP dump details\n2. Check SM12 for locks on PP tables (AUFK, AFKO, AFPO)\n3. Clear old locks if safe to do so\n4. For MRP inconsistency run MDRE to reset MRP data for the material\n5. Check SM21 system log for additional context\n6. If standard SAP program check SNOTE for relevant notes",
        "tcodes_used":   ["ST22", "SM12", "MDRE", "SM21", "MD04"],
        "resolution_time_minutes": 85,
    },

    ("PP", "Data Inconsistency"): {
        "root_cause":    "Production data inconsistency — planned vs actual quantities not matching, MRP results inconsistent with actual stock, or order status not reflecting actual progress.",
        "fix":           "1. Run MD04 to check current MRP situation for the material and plant\n2. Compare planned orders, production orders, and actual stock in MMBE\n3. Run MDRE to reset and recalculate MRP data if inconsistent\n4. Check if any goods movements were posted incorrectly using MB51\n5. Run MCP0 or MCNB production reports to identify the discrepancy source\n6. Correct incorrect postings and rerun MRP",
        "tcodes_used":   ["MD04", "MMBE", "MDRE", "MB51", "MD01"],
        "resolution_time_minutes": 95,
    },

    ("PP", "User Error"): {
        "root_cause":    "User performed incorrect action — confirmed wrong quantity, closed order prematurely, or used wrong movement type for goods issue.",
        "fix":           "1. Review the production order in CO03 to understand current status\n2. If confirmation was wrong use CO13 to cancel the incorrect confirmation\n3. If order was closed incorrectly use CO02 to reopen if technically possible\n4. If goods issue was wrong reverse the movement in MIGO with movement type 262\n5. Guide user through the correct confirmation process\n6. Check if any cost or stock impact needs to be corrected",
        "tcodes_used":   ["CO03", "CO13", "CO02", "MIGO"],
        "resolution_time_minutes": 45,
    },

    ("PP", "Performance"): {
        "root_cause":    "MRP run or production report taking too long — large planning scope, missing database indexes, or MRP data inconsistency causing infinite loops.",
        "fix":           "1. Check MRP run scope — reduce to specific plant and material group if possible\n2. Run MRP in background using MD01 with background processing flag\n3. Check if MDRE can clean up inconsistent MRP data before rerun\n4. Review MRP planning file entries in MDAB\n5. Check SM50 during MRP run to identify bottleneck work processes\n6. Escalate to BASIS for database analysis if consistently slow",
        "tcodes_used":   ["MD01", "MDRE", "MDAB", "SM50", "SM37"],
        "resolution_time_minutes": 80,
    },

    # =========================================================================
    # BASIS — SYSTEM ADMINISTRATION
    # =========================================================================

    ("BASIS", "Authorization"): {
        "root_cause":    "User account locked, role not assigned, authorization object missing, or profile not generated after role change.",
        "fix":           "1. Check user status in SU01 — unlock account if locked\n2. Verify roles are assigned in the Roles tab\n3. Check if profile has been generated after last role change in PFCG\n4. Run SU53 for specific missing authorization objects\n5. Update role in PFCG if object is missing\n6. Generate authorization profile\n7. If transport needed move role to production\n8. Ask user to log out completely and log back in",
        "tcodes_used":   ["SU01", "PFCG", "SU53", "SU10", "SUIM"],
        "resolution_time_minutes": 30,
    },

    ("BASIS", "Configuration"): {
        "root_cause":    "System configuration issue — incorrect profile parameter, work process configuration, or system setting causing unexpected behaviour.",
        "fix":           "1. Check relevant profile parameters in RZ11\n2. Review system profile in RZ10\n3. Check operation mode configuration in SM63 if work process issue\n4. Review background job configuration in SM36 if job scheduling issue\n5. Check logon group configuration in SMLG if load balancing issue\n6. Apply parameter changes and restart instance if required\n7. Monitor SM50 after change to confirm improvement",
        "tcodes_used":   ["RZ11", "RZ10", "SM63", "SM36", "SMLG", "SM50"],
        "resolution_time_minutes": 90,
    },

    ("BASIS", "System Error"): {
        "root_cause":    "System-level error — ABAP dump, work process crash, database error, or transport causing inconsistent state.",
        "fix":           "1. Check ST22 for ABAP dumps — note program, line number, and error type\n2. Review SM21 system log for errors and warnings\n3. Check SM50 for work process status — look for running, waiting, or stopped processes\n4. Check SM66 for global work process overview\n5. Clear old database locks in SM12 if relevant tables are locked\n6. Restart hung work processes if necessary\n7. For transport-related issues check STMS import log",
        "tcodes_used":   ["ST22", "SM21", "SM50", "SM66", "SM12", "STMS"],
        "resolution_time_minutes": 60,
    },

    ("BASIS", "Performance"): {
        "root_cause":    "System performance degradation — database growth, missing indexes, too many work processes busy, or memory shortage.",
        "fix":           "1. Check SM50 and SM66 for work process utilization\n2. Run ST05 SQL trace on a slow transaction to identify bottleneck query\n3. Check OS-level resources using OS07N\n4. Review database statistics and indexes using DB02\n5. Check AL08 for active users — look for sessions consuming excessive resources\n6. Kill any runaway processes after confirming with user\n7. Schedule archiving if database growth is the cause",
        "tcodes_used":   ["SM50", "SM66", "ST05", "OS07N", "DB02", "AL08"],
        "resolution_time_minutes": 75,
    },

    ("BASIS", "User Error"): {
        "root_cause":    "User locked themselves out, forgot password, or is trying to access something incorrectly at the system level.",
        "fix":           "1. Check user in SU01 — unlock account if locked due to failed login attempts\n2. Reset password and set 'change password on next login' flag\n3. If user deleted by mistake recreate from SU01 or restore from backup\n4. Check login parameters in RZ11 for password policy settings\n5. Guide user through correct login procedure\n6. If role-related confusion review user's assigned roles in SU01",
        "tcodes_used":   ["SU01", "SU10", "RZ11"],
        "resolution_time_minutes": 20,
    },

    ("BASIS", "Integration"): {
        "root_cause":    "RFC connection failure, ALE/IDocs error, or interface between SAP and external system not working.",
        "fix":           "1. Check RFC destination in SM59 — test connection to confirm if reachable\n2. Review IDoc status in WE02 or WE05 for ALE issues\n3. Check partner profile in WE20 if IDoc configuration issue\n4. Review SM21 log for RFC or communication errors\n5. Check if the target system is running and network connectivity exists\n6. Restart RFC connection or recreate if certificate expired",
        "tcodes_used":   ["SM59", "WE02", "WE05", "WE20", "SM21"],
        "resolution_time_minutes": 70,
    },

    # =========================================================================
    # ABAP — CUSTOM DEVELOPMENT
    # =========================================================================

    ("ABAP", "System Error"): {
        "root_cause":    "Custom ABAP program dumping at runtime — caused by system upgrade introducing incompatibility, data type mismatch, or unhandled exception in the code.",
        "fix":           "1. Check ST22 for the full dump — note the exact program name, line number, and error category\n2. Open the program in SE38 and navigate to the line causing the dump\n3. Check if the error is data-related (specific input causing dump) or always happening\n4. If system upgrade caused it check syntax using SE38 extended syntax check\n5. Apply ABAP fix — handle the exception or correct the data type issue\n6. Test fix in development system first\n7. Transport to production via STMS after testing",
        "tcodes_used":   ["ST22", "SE38", "SE80", "STMS", "SE91"],
        "resolution_time_minutes": 120,
    },

    ("ABAP", "Configuration"): {
        "root_cause":    "Custom program configuration missing — variant not created, selection screen parameters not set, or custom table entries missing that the program depends on.",
        "fix":           "1. Check if a variant is required for the program — run SE38 and check Goto > Variants\n2. Create the required variant with correct parameters\n3. Check if the program reads from a custom configuration table — verify entries in SE16N\n4. Check if any enhancement framework implementation is missing using SE19\n5. If background job run verify job definition in SM36 has correct variant assigned",
        "tcodes_used":   ["SE38", "SE16N", "SE19", "SM36", "SM37"],
        "resolution_time_minutes": 60,
    },

    ("ABAP", "Data Inconsistency"): {
        "root_cause":    "Custom report showing different data than standard SAP — caused by incorrect SELECT statement, wrong table join, missing WHERE condition, or custom table out of sync.",
        "fix":           "1. Run the standard SAP transaction alongside the custom report to identify the discrepancy\n2. Open the custom report in SE38 and review the SELECT statements\n3. Check if all relevant tables are being read and joins are correct\n4. Verify filtering conditions match what the standard report uses\n5. Check if any custom table is being used that might be out of sync\n6. If data in custom table is wrong run the synchronization report if available\n7. Fix the SELECT query or data source and retest",
        "tcodes_used":   ["SE38", "SE16N", "SE11", "ST05"],
        "resolution_time_minutes": 100,
    },

    ("ABAP", "Performance"): {
        "root_cause":    "Custom program running slowly — SELECT * without field restriction, SELECT inside a loop (classic N+1 problem), missing database index, or large data volume without parallel processing.",
        "fix":           "1. Run ST05 SQL trace while the program executes to identify the slow queries\n2. Open the program in SE38 and look for SELECT statements inside loops\n3. Refactor to use SELECT...INTO TABLE (array fetch) instead of SELECT SINGLE in loops\n4. Add WHERE conditions to restrict data volume\n5. Check if a database index exists for the fields used in WHERE clause using SE11\n6. If index missing create one after consulting BASIS team\n7. Consider using parallel processing for large volume jobs",
        "tcodes_used":   ["ST05", "SE38", "SE11", "SM50", "SE30"],
        "resolution_time_minutes": 150,
    },

    ("ABAP", "Integration"): {
        "root_cause":    "Custom RFC function module or BAPI call failing — remote system unreachable, interface parameter mismatch, or custom enhancement blocking standard BAPI.",
        "fix":           "1. Test RFC destination in SM59 — confirm connection to remote system is working\n2. Check the custom function module in SE37 — verify input and output parameters\n3. Run the function module in test mode using SE37 with sample data\n4. Check if any enhancement to the standard BAPI is causing the failure\n5. Review ST22 for any dumps during the RFC call\n6. Check SM21 for RFC communication errors\n7. Fix parameter mismatch or connection issue and retest",
        "tcodes_used":   ["SM59", "SE37", "SE38", "ST22", "SM21"],
        "resolution_time_minutes": 90,
    },

    ("ABAP", "Authorization"): {
        "root_cause":    "Custom program missing authorization checks or user lacks authorization for custom authorization object used by the program.",
        "fix":           "1. Check the program in SE38 for AUTHORITY-CHECK statements\n2. If check is present run SU53 to see which object is failing\n3. If custom authorization object is used check it exists in SU21\n4. Add missing authorization object to user role in PFCG\n5. If program has no authorization check consider adding one for security compliance\n6. Generate and reassign role after changes",
        "tcodes_used":   ["SE38", "SU53", "SU21", "PFCG", "SU01"],
        "resolution_time_minutes": 55,
    },

    # =========================================================================
    # HR — HUMAN RESOURCES
    # =========================================================================

    ("HR", "Authorization"): {
        "root_cause":    "Missing HR-specific authorization — P_ORGIN for infotype access, P_PERNR for own data access, or structural authorization not covering the employee's organizational unit.",
        "fix":           "1. Run SU53 to identify the missing authorization object\n2. For P_ORGIN — check infotype, personnel area, employee group, and subgroup values in the role\n3. For structural authorization check OOSB for the authorization profile assignment\n4. Open PFCG and add missing HR authorization object with correct infotype ranges\n5. Generate profile and reassign role\n6. Test with PA20 or PA30 using an employee in the relevant personnel area",
        "tcodes_used":   ["SU53", "PFCG", "SU01", "OOSB", "PA20"],
        "resolution_time_minutes": 50,
    },

    ("HR", "Configuration"): {
        "root_cause":    "HR feature, schema, or rule configuration incorrect — leave quota calculation wrong, payroll schema missing wage type, or time evaluation rule not covering the employee's work schedule.",
        "fix":           "1. Check the relevant HR feature in PE03 — verify the decision tree values for the employee group and subgroup\n2. If payroll issue check schema T52C1 or relevant country schema in PE01\n3. Review wage type configuration in OH11 or table V_T511\n4. Check work schedule rule assigned to employee in IT0007 using PA20\n5. Review time management settings in SPRO under Personnel Time Management\n6. Test configuration change with a test employee before applying broadly",
        "tcodes_used":   ["PE03", "PE01", "PA20", "SPRO", "PT60"],
        "resolution_time_minutes": 90,
    },

    ("HR", "Master Data"): {
        "root_cause":    "Employee master data incomplete or incorrect — missing infotype record, wrong organizational assignment, or incomplete payroll-relevant data causing processing failure.",
        "fix":           "1. Open employee master data in PA20 and review all relevant infotypes\n2. Check IT0001 (organizational assignment) for correct personnel area and cost center\n3. Check IT0008 (basic pay) if payroll issue — verify wage types and amounts\n4. Check IT0007 (planned working time) if time evaluation issue\n5. Correct missing or incorrect infotype records using PA30\n6. Run payroll simulation in PY60 to verify fix before actual payroll run",
        "tcodes_used":   ["PA20", "PA30", "IT0001", "IT0008", "PY60"],
        "resolution_time_minutes": 40,
    },

    ("HR", "System Error"): {
        "root_cause":    "Payroll or time evaluation failing with system error — schema inconsistency, function module dump, or cluster table inconsistency in payroll results.",
        "fix":           "1. Check the payroll log in PC_PAYRESULT or PY60 for the exact error message\n2. Review ST22 for any ABAP dumps during payroll execution\n3. Check if the error affects all employees or specific personnel area\n4. Run payroll for a single test employee to isolate the error\n5. Check for OSS notes related to the error in SNOTE\n6. If cluster table inconsistency — check with BASIS team for table repair",
        "tcodes_used":   ["PY60", "PC_PAYRESULT", "ST22", "SNOTE", "PA03"],
        "resolution_time_minutes": 120,
    },

    ("HR", "Data Inconsistency"): {
        "root_cause":    "HR data inconsistency — attendance data not matching payroll results, organizational data not reflecting recent changes, or time balances incorrect.",
        "fix":           "1. Compare time data in PT60 report with payroll results\n2. Check if time evaluation has been run after the data change using PT66\n3. Verify organizational changes have correct validity dates in IT0001\n4. Check if delimitation of old infotype records was done correctly\n5. Run time evaluation retroactively if needed using PT60 with earlier start date\n6. Rerun payroll simulation after correction to confirm consistency",
        "tcodes_used":   ["PT60", "PT66", "PA20", "PA30", "PY60"],
        "resolution_time_minutes": 85,
    },

    ("HR", "Performance"): {
        "root_cause":    "HR report or payroll run taking too long — large employee population, missing index, or payroll schema performing unnecessary calculations.",
        "fix":           "1. Check if payroll is running for too large a scope — narrow to specific personnel area\n2. Run payroll in parallel processing mode if available\n3. Check ST05 SQL trace during slow report to find bottleneck query\n4. Review report selection criteria — add more specific filters\n5. Schedule large payroll runs and HR reports as background jobs in SM36\n6. Consult BASIS team if database index optimization is needed",
        "tcodes_used":   ["SM36", "SM37", "ST05", "PT60", "PA03"],
        "resolution_time_minutes": 70,
    },

    ("HR", "Integration"): {
        "root_cause":    "Integration failure between HR and FI or external systems — payroll posting to FI not working, interface to third-party payroll system failing, or cost center assignment not posting correctly.",
        "fix":           "1. Check payroll posting log in PC00_M99_CIPE for errors\n2. Verify symbolic account to wage type mapping in PPMOD\n3. Check FI account assignment in table T52EK\n4. Review cost center assignment in employee IT0001 and IT0027\n5. For external interface check RFC connection in SM59\n6. Review IDoc status in WE02 if ALE-based integration",
        "tcodes_used":   ["PC00_M99_CIPE", "SM59", "WE02", "PA20", "PPMOD"],
        "resolution_time_minutes": 95,
    },

    ("HR", "User Error"): {
        "root_cause":    "HR user made incorrect data entry — wrong dates, wrong wage type, or locked an employee record incorrectly.",
        "fix":           "1. Review the incorrect infotype record in PA20\n2. Use PA30 to correct the record — delimit the incorrect entry and create the correct one\n3. If payroll already ran with incorrect data check if retroactive payroll will correct it\n4. Run payroll simulation in PY60 to verify correction\n5. Guide user through correct data entry process\n6. Check if any downstream processes were affected by the incorrect data",
        "tcodes_used":   ["PA20", "PA30", "PY60", "PT60"],
        "resolution_time_minutes": 45,
    },
}

# Default resolution for combinations that may not lie in any of above categories.
# This is a fallback to ensure that every generated ticket has a resolution.
DEFAULT_RESOLUTION = {
    "root_cause":    "Issue identified in {module} module related to {issue_type}",
    "fix":           "1. Investigated the issue in relevant SAP transactions\n2. Identified root cause through system analysis\n3. Applied configuration or master data fix\n4. Tested with affected user and confirmed resolution",
    "tcodes_used":   [],
    "resolution_time_minutes": 60,
}


# =============================================================================
# PROGRESS CHECKPOINT HELPERS
# These functions enable the generator to resume from exactly the last
# completed seed within each pass after any interruption.
# =============================================================================

def load_progress() -> dict:
    """
    Load generation progress from the checkpoint file.

    WHY THIS EXISTS: Allows the generator to resume from exactly where it
    stopped at the seed level within each pass. Without this, every restart
    would overwrite the CSVs and start from zero.

    Returns a progress dict with all counts zeroed if no checkpoint file
    exists — this means fresh start. Same code path handles both cases.
    """
    if not os.path.exists(PROGRESS_FILE):
        # No checkpoint — return zeroed structure for a fresh start
        return {
            "incidents": {
                "completed_passes": 0,
                "completed_seeds_in_current_pass": [],
                "total_count": 0,
            },
            "development": {
                "completed_passes": 0,
                "completed_seeds_in_current_pass": [],
                "total_count": 0,
            },
            "edge_cases": {
                "completed_passes": 0,
                "completed_seeds_in_current_pass": [],
                "total_count": 0,
            },
            "last_updated": None,
        }
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_progress(progress: dict) -> None:
    """
    Save current generation progress to the checkpoint file.

    WHY THIS EXISTS: Called after every seed completes and its rows are
    flushed to disk via os.fsync(). This is the finest safe checkpoint
    granularity — if the script stops mid-seed we lose at most one API
    call (20 tickets). inspect_data.ipynb removes duplicates.

    IMPORTANT: save_progress() is deliberately NOT called when a seed
    returns an empty list (API failure). This means the seed's position
    in completed_seeds_in_current_pass is never advanced, so the resume
    logic on the next run will naturally retry the failed seed and append
    its rows to the CSV. Append mode + inspect_data sorting handles order.
    """
    progress["last_updated"] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


# =============================================================================
# GRACEFUL SHUTDOWN HANDLER
# =============================================================================

def handle_shutdown(signum, frame):
    """
    Handle Ctrl+C (SIGINT) and system termination (SIGTERM) gracefully.

    WHY THIS EXISTS: Progress is saved after every successful seed, so no
    data is lost on shutdown. This handler prints a clear message so the
    user knows the script is safe to close and can be resumed.
    """
    print("\n\nShutdown signal received.")
    print("Progress is saved. Run the script again to resume from checkpoint.")
    print(f"Checkpoint: {PROGRESS_FILE}")
    sys.exit(0)

# Register shutdown handler for keyboard interrupt and system signals
signal.signal(signal.SIGINT,  handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


# =============================================================================
# GROQ API CALL
# This function calls the Groq API to generate text variations.
# It takes a prompt describing what to generate and returns the LLM response.
# =============================================================================
# call_groq() kept below for reference — replaced by call_llm()
# def call_groq(prompt: str, max_tokens: int = 800) -> str:
#     """
#     Call the Groq API with a prompt and return the generated text.
# 
#     WHY GROQ: Free tier, fast inference (1000 tokens/sec on fast model),
#     no credit card required. We use gpt-oss-20b (fast model) here because
#     we are generating 25,000 tickets and speed matters more than quality.
# 
#     Args:
#         prompt: The instruction telling the LLM what to generate
#         max_tokens: Maximum length of the response
# 
#     Returns:
#         The generated text as a string, or empty string if API call fails
#     """
#     if not GROQ_API_KEY:
#         raise ValueError(
#             "GROQ_API_KEY not set. Set it as an environment variable:\n"
#             "Windows: set GROQ_API_KEY=your_key_here\n"
#             "Mac/Linux: export GROQ_API_KEY=your_key_here"
#         )
# 
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type":  "application/json",
#     }
# 
#     payload = {
#         "model":      GROQ_FAST_MODEL,
#         "max_tokens": max_tokens,
#         "messages": [
#             {
#                 "role":    "system",
#                 "content": (
#                     "You are a data generation assistant. Generate exactly what "
#                     "is asked — nothing more, nothing less. No explanations, no "
#                     "preamble, no formatting unless asked. Output only the "
#                     "requested data in the exact format specified."
#                 ),
#             },
#             {
#                 "role":    "user",
#                 "content": prompt,
#             },
#         ],
#     }
# 
#     # Maximum 3 attempts. On all retries exhausted, return "" so the caller
#     # knows the seed failed and can skip it without saving progress.
#     max_retries = 3
#     for attempt in range(max_retries):
#         try:
#             response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
# 
#             if response.status_code == 429:
#                 # Rate limit hit — flat 20-second wait regardless of attempt number.
#                 # Flat (not exponential) because Groq's free tier resets every ~15-20s.
#                 # Exponential would overshoot the reset window unnecessarily.
#                 print(f"  Rate limit (429) — waiting 20s flat (attempt {attempt + 1}/{max_retries})...")
#                 time.sleep(20)
#                 continue
# 
#             response.raise_for_status()
#             return response.json()["choices"][0]["message"]["content"].strip()
# 
#         except requests.exceptions.Timeout:
#             # Network timeout — flat 5-second wait then retry
#             print(f"  Timeout — retrying in 5s (attempt {attempt + 1}/{max_retries})...")
#             time.sleep(5)
# 
#         except requests.exceptions.HTTPError as e:
#             # Non-429 HTTP error (e.g. 500, 503) — not retriable
#             print(f"  HTTP error: {e}")
#             break
# 
#         except Exception as e:
#             # Unexpected error — not retriable
#             print(f"  Unexpected error calling Groq: {e}")
#             break
# 
#     # All retries exhausted — return empty string so caller can skip this seed
#     return ""


def call_llm(prompt: str, max_tokens: int = 800) -> str:
    """
    Call the Google Gemini API with a prompt and return the generated text.

    WHY THIS EXISTS: Generates realistic SAP support ticket variations
    from seed templates using an LLM. The LLM produces genuinely varied
    linguistic patterns — different vocabulary, tone, length, panic level —
    which a pure template system cannot match.

    WHY GEMINI: Free tier provides 1,000 requests per day, enough to
    generate 20,000 ticket variations per day. The OpenAI-compatible
    endpoint means request and response format is identical to the
    previous provider — only the URL and API key differ.

    Retry strategy:
    - Rate limit (429): flat 40 second wait, then retry
      Flat rather than exponential because per-minute limits
      reset on a fixed window, not a rolling one
    - Timeout: 5 second wait, then retry
    - Other HTTP errors: log and break immediately, not retriable
    - Maximum 3 attempts per call before returning empty string

    Returns empty string on total failure so the caller can skip
    this seed and retry next run via the checkpoint system.

    Args:
        prompt: The instruction telling the LLM what to generate
        max_tokens: Maximum length of the response

    Returns:
        The generated text as a string, or empty string if all retries fail
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not set. Get a free key at aistudio.google.com\n"
            "Windows:   set GEMINI_API_KEY=your_key_here\n"
            "Mac/Linux: export GEMINI_API_KEY=your_key_here"
        )

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type":  "application/json",
    }

    payload = {
        "model":      GEMINI_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role":    "system",
                "content": (
                    "You are a data generation assistant. Generate exactly what "
                    "is asked — nothing more, nothing less. No explanations, no "
                    "preamble, no formatting unless asked. Output only the "
                    "requested data in the exact format specified."
                ),
            },
            {
                "role":    "user",
                "content": prompt,
            },
        ],
    }

    # Maximum 3 attempts before giving up and returning empty string
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                GEMINI_URL, headers=headers, json=payload, timeout=30
            )

            if response.status_code == 429:
                # Per-minute rate limit hit — flat 40 second wait
                # Flat rather than exponential because the rate limit
                # window resets on a fixed schedule
                print(f"  Rate limit (429) — waiting 40s flat "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(40)
                continue

            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.Timeout:
            # Network timeout — short wait then retry
            print(f"  Timeout — retrying in 5s "
                  f"(attempt {attempt + 1}/{max_retries})...")
            time.sleep(5)

        except requests.exceptions.HTTPError as e:
            # Non-429 HTTP error — not worth retrying
            print(f"  HTTP error: {e}")
            break

        except Exception as e:
            # Unexpected error — not retriable
            print(f"  Unexpected error: {e}")
            break

    # All retries exhausted — return empty string
    # Caller must NOT save progress for this seed so it retries next run
    return ""


# =============================================================================
# GENERATE INCOMING TICKET VARIATIONS
# Takes a seed text and generates multiple different ways a real user
# might write the same problem — different vocabulary, different panic level,
# different amount of detail, some with more context, some minimal.
# =============================================================================
def generate_incoming_variations(
    seed_text: str,
    module: str,
    issue_type: str,
    has_error_code: bool,
    error_code: str,
    count: int = VARIATIONS_PER_PASS,
) -> list[str]:
    """
    Generate realistic variations of a seed ticket using the LLM.

    The key instruction to the LLM is to keep the language NON-TECHNICAL
    and MESSY — like a real frustrated user would write, not a consultant.

    Args:
        seed_text: The base example to vary
        module: SAP module (FI, MM, etc.) — for context only
        issue_type: Type of issue — for context only
        has_error_code: Whether to include an SAP error code in the text
        error_code: The actual error code string to insert if has_error_code
        count: How many variations to generate

    Returns:
        List of variation strings
    """
    # Replace {error} placeholder in seed with actual error code if needed
    if has_error_code and error_code:
        seed_with_code = seed_text.replace("{error}", error_code)
    else:
        seed_with_code = seed_text

    prompt = f"""Generate {count} different ways a non-technical SAP end user might write the following support ticket.

Original ticket: "{seed_with_code}"

Rules:
- Write like a real frustrated non-technical user, not a consultant or developer
- Vary the vocabulary, sentence structure, and level of detail significantly
- Some should be very short (one sentence), some medium length (2-3 sentences)
- Some should sound panicked and urgent, some calm and factual
- Some should have typos or grammar errors (like real users make)
- Do NOT use technical SAP jargon unless it appears in the original
- {"Include the error code " + error_code + " in all variations" if has_error_code else "Do NOT include any error codes"}
- Output ONLY the {count} ticket texts, one per line, numbered 1 to {count}
- No explanations, no labels, just the ticket texts"""

    response = call_llm(prompt, max_tokens=800)

    if not response:
        # API call failed after all retries — return empty list.
        # Caller must write nothing, skip save_progress(), print a warning,
        # and let the resume logic retry this seed on the next run.
        return []

    # Robust regex line parser to strip numbered list prefixes (e.g. "1. ", "1) ")
    # without corrupting ticket texts starting with digits (e.g. "1000 plant issue")
    lines = response.strip().split("\n")
    variations = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        cleaned_text = re.sub(r"^\d+[\.\)\-]\s*", "", line).strip()
        if cleaned_text:
            variations.append(cleaned_text)

    # Accept 75% threshold (15+ variations out of 20) as a successful call
    # to prevent unnecessary retries when Groq returns slightly fewer variations.
    if not variations or len(variations) < int(count * 0.75):
        return []

    return variations[:count]


# =============================================================================
# GENERATE DEVELOPMENT REQUEST VARIATIONS
# Similar to incident variations but for new development requests.
# Language is more formal — users writing feature requests tend to be
# more structured than users reporting urgent problems.
# =============================================================================
def generate_development_variations(
    seed_text: str,
    module: str,
    count: int = VARIATIONS_PER_PASS,
) -> list[str]:
    """
    Generate variations of a new development request seed.

    Args:
        seed_text: Base development request example
        module: SAP module for context
        count: Number of variations to generate

    Returns:
        List of variation strings
    """
    prompt = f"""Generate {count} different ways someone might write the following SAP development request or enhancement request.

Original request: "{seed_text}"

Rules:
- These are requests to BUILD something new, not fix a broken thing
- Write from the perspective of a business user or manager making a request
- Vary the vocabulary and phrasing significantly
- Some should be formal business requests, some informal
- Use phrases like: "we need", "please create", "can you build", "we require", "enhancement needed", "new functionality"
- Keep them clearly as NEW requests, not complaints about broken things
- Output ONLY the {count} texts, one per line, numbered 1 to {count}
- No explanations, just the request texts"""

    response = call_llm(prompt, max_tokens=700)

    if not response:
        # API call failed — return empty list so caller skips and retries next run
        return []

    lines = response.strip().split("\n")
    variations = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip numbered list prefixes, preserving text that starts with digits
        cleaned_text = re.sub(r"^\d+[\.\)\-]\s*", "", line).strip()
        if cleaned_text:
            variations.append(cleaned_text)

    # Accept 75% threshold (15+ variations out of 20) as a successful call
    # to prevent unnecessary retries when Groq returns slightly fewer variations.
    if not variations or len(variations) < int(count * 0.75):
        return []

    return variations[:count]


# =============================================================================
# BUILD RESOLVED TICKET RECORD
# Creates the structured resolution record for a given incident.
# This is what goes into the RAG knowledge base.
# The LangGraph agent reads these structured records when generating
# resolution suggestions for new tickets.
# =============================================================================
def build_resolved_record(
    incoming_text: str,
    module: str,
    issue_type: str,
    error_code: str,
    ticket_id: str,
) -> dict:
    """
    Build a structured resolved ticket record from an incoming ticket.

    The structure matters — the LangGraph agent expects these specific fields
    when it reads retrieved resolved tickets to generate suggestions.

    Args:
        incoming_text: The original ticket text as the user wrote it
        module: SAP module
        issue_type: Type of issue
        error_code: Error code if present, empty string if not
        ticket_id: Unique identifier for this ticket

    Returns:
        Dictionary with all resolution fields
    """
    # Get the resolution template for this module/issue_type combination
    # Fall back to default if we don't have a specific template
    template = RESOLUTION_TEMPLATES.get(
        (module, issue_type),
        DEFAULT_RESOLUTION
    )

    # Get relevant T-codes for this module
    module_tcodes = TCODES.get(module, [])

    # Use template T-codes if available, otherwise pick random ones from module
    if template["tcodes_used"]:
        tcodes_used = template["tcodes_used"]
    else:
        # Pick 2-4 random T-codes from the module
        tcodes_used = random.sample(module_tcodes, min(3, len(module_tcodes)))

    # Get a relevant authorization object name for FI/MM/SD auth issues
    auth_objects = {
        "FI": "F_BKPF_BUK", "MM": "M_MSEG_BWA",
        "SD": "V_VBAK_VKO", "PP": "C_AFKO_AWK",
        "BASIS": "S_TCODE",  "ABAP": "S_DEVELOP",
        "HR": "P_ORGIN",
    }

    # Format the root cause safely with actual values
    try:
        root_cause = template["root_cause"].format(
            module     = module,
            issue_type = issue_type,
            auth_obj   = auth_objects.get(module, "S_TCODE"),
            activity   = random.choice(["01", "02", "03", "06"]),
            error      = error_code if error_code else "error",
            error_code = error_code if error_code else "error",
        )
    except Exception:
        root_cause = template["root_cause"]

    # Add some realistic variation to resolution time (±30%)
    base_time = template.get("resolution_time_minutes", 60)
    variation = random.uniform(0.7, 1.3)
    resolution_time = int(base_time * variation)

    # Generate a realistic ticket close date
    # Tickets are typically resolved within 1-5 business days
    days_to_resolve = random.randint(0, 4)
    hours_to_resolve = random.randint(1, 8)
    created_date = datetime.now() - timedelta(
        days = random.randint(30, 365)  # Created sometime in past year
    )
    resolved_date = created_date + timedelta(
        days  = days_to_resolve,
        hours = hours_to_resolve,
    )

    return {
        "ticket_id":        ticket_id,
        "module":           module,
        "issue_type":       issue_type,
        "problem":          incoming_text,
        "error_code":       error_code,
        "root_cause":       root_cause,
        "fix":              template["fix"],
        "tcodes_used":      ", ".join(tcodes_used),
        "resolution_time_minutes": resolution_time,
        "escalated":        "Yes" if resolution_time > 120 else "No",
        "created_date":     created_date.strftime("%Y-%m-%d"),
        "resolved_date":    resolved_date.strftime("%Y-%m-%d"),
        "ticket_type":      "incident",
    }


# =============================================================================
# MAIN GENERATION FUNCTION
# Orchestrates the entire generation process.
# Generates all three ticket categories and writes to CSV files.
# =============================================================================
def generate_all_tickets():
    """
    Main function that generates all 25,000 synthetic tickets.

    Generation breakdown:
    - 15,000 incident tickets  (INCIDENT_SEEDS x VARIATIONS_PER_PASS x N passes)
    -  6,500 development requests (DEVELOPMENT_SEEDS x VARIATION_PER_PASS x N passes)
    -  3,500 edge cases (EDGE_CASE_SEEDS x VARIATION_PER_PASS x N passes)

    Output files:
    - incoming_tickets.csv: all ticket texts with labels (for BERT training)
    - resolved_tickets.csv: structured resolutions (for RAG knowledge base)

    Resume behaviour:
    - If PROGRESS_FILE and INCOMING_CSV both exist with data, the script
      resumes from the last completed seed. Already-written rows are preserved.
    - If either file is absent, the script starts fresh.

    Failed seed behaviour (Task 1):
    - When a seed's API call returns [] after 3 retries, zero rows are written
      to either CSV. save_progress() is NOT called for that seed, so
      completed_seeds_in_current_pass is not advanced. The next run's resume
      skip logic will not skip that seed — it retries it and appends its rows.
      inspect_data.ipynb sorts and deduplicates, so append order does not matter.
    """

    # ── RESUME DETECTION ──────────────────────────────────────────────────
    # load_progress() returns all-zeros when no checkpoint file exists,
    # so fresh-start and resume use exactly the same code path from here.
    progress = load_progress()

    # Track progress counts and daily limit exhaustion status across all phases
    inc_count  = progress["incidents"]["total_count"]
    dev_count  = progress["development"]["total_count"]
    edge_count = progress["edge_cases"]["total_count"]
    daily_limit_exhausted = False

    # Determine if this is a resume by checking whether the CSV already
    # exists and has data in it (more than just the header row)
    is_resuming = (
        os.path.exists(INCOMING_CSV)
        and os.path.getsize(INCOMING_CSV) > 0
    )

    if is_resuming:
        # Count existing rows by type so the summary is accurate
        existing_incidents   = 0
        existing_development = 0
        existing_edge_cases  = 0
        with open(INCOMING_CSV, "r", encoding="utf-8") as check_file:
            for row in csv.DictReader(check_file):
                if   row["ticket_type"] == "incident":    existing_incidents   += 1
                elif row["ticket_type"] == "development": existing_development += 1
                elif row["ticket_type"] == "edge_case":   existing_edge_cases  += 1

        print("=" * 55)
        print("RESUME MODE — continuing from checkpoint")
        print(f"Incidents:   {existing_incidents:,} / {TARGET_INCIDENTS:,}")
        print(f"Development: {existing_development:,} / {TARGET_DEVELOPMENT:,}")
        print(f"Edge Cases:  {existing_edge_cases:,} / {TARGET_EDGE_CASES:,}")
        print(f"Last saved:  {progress['last_updated']}")
        print("=" * 55)

        # Append mode — preserve all existing rows, no headers written again
        inc_file_mode = "a"
        write_headers = False
    else:
        print("=" * 55)
        print("FRESH START — generating 25,000 tickets")
        print(f"Model:       {GEMINI_MODEL}")
        print(f"Incidents:   {TARGET_INCIDENTS:,}")
        print(f"Development: {TARGET_DEVELOPMENT:,}")
        print(f"Edge Cases:  {TARGET_EDGE_CASES:,}")
        print("=" * 55)

        # Write mode — create new files, headers written once
        inc_file_mode = "w"
        write_headers = True

    # WHY THIS EXISTS: When resuming from a checkpoint in append mode ('a'),
    # starting ticket_counter at 1 would re-issue existing ticket IDs (e.g. INC-000001).
    # We inspect existing IDs in INCOMING_CSV to resume ticket_counter from max_num + 1.
    if is_resuming:
        # Find the highest ticket number already written to avoid duplicate IDs
        # ticket_id format is INC-000001, DEV-000006, EDG-000012 etc.
        # Extract the numeric part and find the maximum across all rows
        max_ticket_num = 0
        with open(INCOMING_CSV, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tid = row.get("ticket_id", "")
                if "-" in tid:
                    try:
                        num = int(tid.split("-")[1])
                        if num > max_ticket_num:
                            max_ticket_num = num
                    except ValueError:
                        pass
        ticket_counter = max_ticket_num + 1
        print(f"Resuming ticket IDs from: {ticket_counter:,}")
    else:
        ticket_counter = 1

    # Make sure output directory exists
    os.makedirs("data/synthetic", exist_ok=True)

    # ── FIELD DEFINITIONS ─────────────────────────────────────────────────
    # incoming_tickets.csv fields:
    # - ticket_id:      unique ID (INC-XXXXXX, DEV-XXXXXX, EDG-XXXXXX)
    # - text:           the raw ticket text as the user would write it
    # - ticket_type:    incident / development / edge_case
    # - module:         FI, MM, SD, PP, BASIS, ABAP, HR
    # - issue_type:     Authorization, Configuration, etc. (empty for dev)
    # - has_error_code: True/False — whether ticket contains an SAP error code
    # - error_code:     the actual error code if present, empty string if not
    #
    # resolved_tickets.csv fields (only for incident and edge case tickets):
    # - ticket_id, module, issue_type, problem, error_code,
    #   root_cause, fix, tcodes_used, resolution_time_minutes,
    #   escalated, created_date, resolved_date, ticket_type

    incoming_fields = [
        "ticket_id", "text", "ticket_type", "module",
        "issue_type", "has_error_code", "error_code",
    ]

    resolved_fields = [
        "ticket_id", "module", "issue_type", "problem", "error_code",
        "root_cause", "fix", "tcodes_used", "resolution_time_minutes",
        "escalated", "created_date", "resolved_date", "ticket_type",
    ]

    # ── OPEN CSV FILES ────────────────────────────────────────────────────
    with open(INCOMING_CSV, inc_file_mode, newline="", encoding="utf-8") as inc_file, \
         open(RESOLVED_CSV, inc_file_mode, newline="", encoding="utf-8") as res_file:

        inc_writer = csv.DictWriter(inc_file, fieldnames=incoming_fields)
        res_writer = csv.DictWriter(res_file, fieldnames=resolved_fields)

        # Write headers only on fresh start — resuming would duplicate the header row
        if write_headers:
            inc_writer.writeheader()
            res_writer.writeheader()

        # ── PHASE 1: INCIDENT TICKETS ──────────────────────────────────────
        # PASS-FIRST ordering: Pass 1 processes all 58 seeds, then Pass 2
        # processes all 58 seeds again, and so on. This is intentional —
        # pass-first produces better linguistic diversity because the LLM
        # generates fresh variations on each pass. Do not change to seed-first.
        #
        # Math: 58 seeds × 20 variations/pass = 1,160 per pass
        # ceil(15,000 / 1,160) = 13 passes required

        INCIDENT_PASSES = math.ceil(
            TARGET_INCIDENTS / (len(INCIDENT_SEEDS) * VARIATIONS_PER_PASS)
        )
        print(f"\nPHASE 1: Generating {TARGET_INCIDENTS:,} incident tickets")
        print(f"Seeds: {len(INCIDENT_SEEDS)} | Passes: {INCIDENT_PASSES} | "
              f"Variations per call: {VARIATIONS_PER_PASS}")
        print("-" * 40)

        # Restore count from checkpoint — 0 on fresh start
        inc_count = progress["incidents"]["total_count"]

        for pass_idx in range(INCIDENT_PASSES):
            if inc_count >= TARGET_INCIDENTS:
                break

            completed_passes = progress["incidents"]["completed_passes"]
            if pass_idx < completed_passes:
                continue   # Entire pass already done in a previous run

            # ── PASS RETRY LOOP ──────────────────────────────────────────────
            # WHY THIS EXISTS: Tracks consecutive empty/failed iterations across all seeds in a pass.
            # If rate limits or API outages cause 5 consecutive iterations where zero seeds succeed,
            # we break gracefully to avoid infinite loops and allow resuming later.
            consecutive_empty_passes = 0
            while (len(progress["incidents"]["completed_seeds_in_current_pass"]) < len(INCIDENT_SEEDS)
                   and inc_count < TARGET_INCIDENTS):

                missing_indices = [
                    i for i in range(len(INCIDENT_SEEDS))
                    if i not in progress["incidents"]["completed_seeds_in_current_pass"]
                ]

                seed_succeeded_this_iter = False
                for seed_idx in missing_indices:
                    if inc_count >= TARGET_INCIDENTS:
                        break

                    module, issue_type, has_error_code, seed_text = INCIDENT_SEEDS[seed_idx]

                    # Pick a random error code for seeds that require one
                    error_code = ""
                    if has_error_code:
                        error_code = random.choice(ERROR_CODES.get(module, ["SY 001"]))

                    print(f"  Pass {pass_idx + 1}/{INCIDENT_PASSES} | "
                          f"Seed {seed_idx + 1}/{len(INCIDENT_SEEDS)} | "
                          f"{module} - {issue_type}", end="")

                    # One API call generates VARIATIONS_PER_PASS (20) variations
                    variations = generate_incoming_variations(
                        seed_text      = seed_text,
                        module         = module,
                        issue_type     = issue_type,
                        has_error_code = has_error_code,
                        error_code     = error_code,
                    )

                    print(f" -> {len(variations)} tickets")

                    if not variations or len(variations) < int(VARIATIONS_PER_PASS * 0.75):
                        # API returned [] or fewer than 75% threshold (15) variations.
                        # Do NOT write to CSV, do NOT save progress. Seed remains
                        # in missing_indices and will be retried in the next while-loop iteration.
                        print(f"  WARNING: Seed {seed_idx + 1} in Pass {pass_idx + 1} "
                              f"did not return sufficient variations (>=75%) — retrying seed")
                        continue

                    # Write all variations to both CSVs
                    for ticket_text in variations:
                        if inc_count >= TARGET_INCIDENTS:
                            break

                        ticket_id = f"INC-{ticket_counter:06d}"
                        ticket_counter += 1

                        # Write incoming ticket row (raw user text, for BERT training)
                        inc_writer.writerow({
                            "ticket_id":      ticket_id,
                            "text":           ticket_text,
                            "ticket_type":    "incident",
                            "module":         module,
                            "issue_type":     issue_type,
                            "has_error_code": has_error_code,
                            "error_code":     error_code,
                        })

                        # Write resolved record side-by-side — incidents always
                        # get a resolved record because they feed the RAG DB
                        resolved = build_resolved_record(
                            incoming_text = ticket_text,
                            module        = module,
                            issue_type    = issue_type,
                            error_code    = error_code,
                            ticket_id     = ticket_id,
                        )
                        res_writer.writerow(resolved)
                        inc_count += 1

                    # Flush both files to disk (physical write via os.fsync)
                    inc_file.flush()
                    os.fsync(inc_file.fileno())
                    res_file.flush()
                    os.fsync(res_file.fileno())

                    # Record seed as completed for this pass ONLY after all rows are written
                    if seed_idx not in progress["incidents"]["completed_seeds_in_current_pass"]:
                        progress["incidents"]["completed_seeds_in_current_pass"].append(seed_idx)
                    progress["incidents"]["total_count"] = inc_count
                    save_progress(progress)

                    seed_succeeded_this_iter = True

                    # 5 second delay between API calls — respects 15 RPM free tier limit
                    time.sleep(5)

                if seed_succeeded_this_iter:
                    consecutive_empty_passes = 0
                else:
                    consecutive_empty_passes += 1

                if consecutive_empty_passes >= 5:
                    print("\n" + "=" * 55)
                    print("DAILY RATE LIMIT EXHAUSTED")
                    print("All seeds returned empty 5 consecutive times.")
                    print("Progress saved. Run script again tomorrow.")
                    print("Current progress:")
                    print(f"  Incidents:   {inc_count:,} / {TARGET_INCIDENTS:,}")
                    print(f"  Development: {dev_count:,} / {TARGET_DEVELOPMENT:,}")
                    print(f"  Edge Cases:  {edge_count:,} / {TARGET_EDGE_CASES:,}")
                    print("=" * 55)
                    daily_limit_exhausted = True
                    break

                # If missing seeds remain, print status before retrying them
                remaining = len(INCIDENT_SEEDS) - len(progress["incidents"]["completed_seeds_in_current_pass"])
                if remaining > 0 and inc_count < TARGET_INCIDENTS:
                    print(f"  [Pass {pass_idx + 1}] Retrying {remaining} uncompleted seed(s)...")
                    time.sleep(3)

            if daily_limit_exhausted:
                break

            # ── END OF PASS ────────────────────────────────────────────────
            # Increment completed_passes ONLY when ALL seeds in this pass are done
            if (len(progress["incidents"]["completed_seeds_in_current_pass"]) >= len(INCIDENT_SEEDS)
                    or inc_count >= TARGET_INCIDENTS):
                if pass_idx >= progress["incidents"]["completed_passes"]:
                    progress["incidents"]["completed_passes"]                = pass_idx + 1
                    progress["incidents"]["completed_seeds_in_current_pass"] = []
                    save_progress(progress)

            print(f"  Pass {pass_idx + 1} done | Incidents so far: {inc_count:,}")

        if daily_limit_exhausted:
            print("Generation paused. Rerun tomorrow to continue.")
            return

        print(f"\nPhase 1 complete: {inc_count:,} incident tickets generated")

        # ── PHASE 2: DEVELOPMENT REQUEST TICKETS ──────────────────────────
        # Development tickets go to incoming_tickets.csv only.
        # They are NOT written to resolved_tickets.csv because they are
        # feature requests that DistilBERT routes out at Step 0.
        #
        # Math: 18 seeds × 20 variations/pass = 360 per pass
        # ceil(6,500 / 360) = 19 passes required

        DEV_PASSES = math.ceil(
            TARGET_DEVELOPMENT / (len(DEVELOPMENT_SEEDS) * VARIATIONS_PER_PASS)
        )
        print(f"\nPHASE 2: Generating {TARGET_DEVELOPMENT:,} development request tickets")
        print(f"Seeds: {len(DEVELOPMENT_SEEDS)} | Passes: {DEV_PASSES} | "
              f"Variations per call: {VARIATIONS_PER_PASS}")
        print("-" * 40)

        # Restore count from checkpoint
        dev_count = progress["development"]["total_count"]

        for pass_idx in range(DEV_PASSES):
            if dev_count >= TARGET_DEVELOPMENT:
                break

            completed_passes = progress["development"]["completed_passes"]
            if pass_idx < completed_passes:
                continue

            # ── PASS RETRY LOOP ──────────────────────────────────────────────
            # WHY THIS EXISTS: Tracks consecutive empty/failed iterations across all seeds in a pass.
            # If rate limits or API outages cause 5 consecutive iterations where zero seeds succeed,
            # we break gracefully to avoid infinite loops and allow resuming later.
            consecutive_empty_passes = 0
            while (len(progress["development"]["completed_seeds_in_current_pass"]) < len(DEVELOPMENT_SEEDS)
                   and dev_count < TARGET_DEVELOPMENT):

                missing_indices = [
                    i for i in range(len(DEVELOPMENT_SEEDS))
                    if i not in progress["development"]["completed_seeds_in_current_pass"]
                ]

                seed_succeeded_this_iter = False
                for seed_idx in missing_indices:
                    if dev_count >= TARGET_DEVELOPMENT:
                        break

                    module, seed_text = DEVELOPMENT_SEEDS[seed_idx]

                    print(f"  Pass {pass_idx + 1}/{DEV_PASSES} | "
                          f"Seed {seed_idx + 1}/{len(DEVELOPMENT_SEEDS)} | "
                          f"{module}", end="")

                    # Generate development request variations via Groq API
                    variations = generate_development_variations(
                        seed_text = seed_text,
                        module    = module,
                    )

                    print(f" -> {len(variations)} tickets")

                    if not variations or len(variations) < int(VARIATIONS_PER_PASS * 0.75):
                        print(f"  WARNING: Seed {seed_idx + 1} in Pass {pass_idx + 1} "
                              f"did not return sufficient variations (>=75%) — retrying seed")
                        continue

                    for ticket_text in variations:
                        if dev_count >= TARGET_DEVELOPMENT:
                            break

                        ticket_id = f"DEV-{ticket_counter:06d}"
                        ticket_counter += 1

                        # Development requests go to incoming CSV only (no resolved record)
                        inc_writer.writerow({
                            "ticket_id":      ticket_id,
                            "text":           ticket_text,
                            "ticket_type":    "development",
                            "module":         module,
                            "issue_type":     "",      # No issue type for dev requests
                            "has_error_code": False,
                            "error_code":     "",
                        })
                        dev_count += 1

                    # Flush incoming only — resolved not written for dev requests
                    inc_file.flush()
                    os.fsync(inc_file.fileno())

                    # Save progress after confirmed disk write
                    if seed_idx not in progress["development"]["completed_seeds_in_current_pass"]:
                        progress["development"]["completed_seeds_in_current_pass"].append(seed_idx)
                    progress["development"]["total_count"] = dev_count
                    save_progress(progress)

                    seed_succeeded_this_iter = True

                    # 5 second delay between API calls — respects 15 RPM free tier limit
                    time.sleep(5)

                if seed_succeeded_this_iter:
                    consecutive_empty_passes = 0
                else:
                    consecutive_empty_passes += 1

                if consecutive_empty_passes >= 5:
                    print("\n" + "=" * 55)
                    print("DAILY RATE LIMIT EXHAUSTED")
                    print("All seeds returned empty 5 consecutive times.")
                    print("Progress saved. Run script again tomorrow.")
                    print("Current progress:")
                    print(f"  Incidents:   {inc_count:,} / {TARGET_INCIDENTS:,}")
                    print(f"  Development: {dev_count:,} / {TARGET_DEVELOPMENT:,}")
                    print(f"  Edge Cases:  {edge_count:,} / {TARGET_EDGE_CASES:,}")
                    print("=" * 55)
                    daily_limit_exhausted = True
                    break

                remaining = len(DEVELOPMENT_SEEDS) - len(progress["development"]["completed_seeds_in_current_pass"])
                if remaining > 0 and dev_count < TARGET_DEVELOPMENT:
                    print(f"  [Pass {pass_idx + 1}] Retrying {remaining} uncompleted development seed(s)...")
                    time.sleep(3)

            if daily_limit_exhausted:
                break

            # ── END OF PASS ────────────────────────────────────────────────
            if (len(progress["development"]["completed_seeds_in_current_pass"]) >= len(DEVELOPMENT_SEEDS)
                    or dev_count >= TARGET_DEVELOPMENT):
                if pass_idx >= progress["development"]["completed_passes"]:
                    progress["development"]["completed_passes"]                = pass_idx + 1
                    progress["development"]["completed_seeds_in_current_pass"] = []
                    save_progress(progress)

            print(f"  Pass {pass_idx + 1} done | Development so far: {dev_count:,}")

        if daily_limit_exhausted:
            print("Generation paused. Rerun tomorrow to continue.")
            return

        print(f"\nPhase 2 complete: {dev_count:,} development request tickets generated")

        # ── PHASE 3: EDGE CASE TICKETS ────────────────────────────────────
        # Edge cases use generate_incoming_variations() — NOT generate_development_variations().
        # They are deliberate ambiguous/vague incidents, not dev requests.
        # They DO get resolved records, same as Phase 1.
        #
        # Math: 15 seeds × 20 variations/pass = 300 per pass
        # ceil(3,500 / 300) = 12 passes required

        EDGE_PASSES = math.ceil(
            TARGET_EDGE_CASES / (len(EDGE_CASE_SEEDS) * VARIATIONS_PER_PASS)
        )
        print(f"\nPHASE 3: Generating {TARGET_EDGE_CASES:,} edge case tickets")
        print(f"Seeds: {len(EDGE_CASE_SEEDS)} | Passes: {EDGE_PASSES} | "
              f"Variations per call: {VARIATIONS_PER_PASS}")
        print("-" * 40)

        # Restore count from checkpoint
        edge_count = progress["edge_cases"]["total_count"]

        for pass_idx in range(EDGE_PASSES):
            if edge_count >= TARGET_EDGE_CASES:
                break

            completed_passes = progress["edge_cases"]["completed_passes"]
            if pass_idx < completed_passes:
                continue

            # ── PASS RETRY LOOP ──────────────────────────────────────────────
            # WHY THIS EXISTS: Tracks consecutive empty/failed iterations across all seeds in a pass.
            # If rate limits or API outages cause 5 consecutive iterations where zero seeds succeed,
            # we break gracefully to avoid infinite loops and allow resuming later.
            consecutive_empty_passes = 0
            while (len(progress["edge_cases"]["completed_seeds_in_current_pass"]) < len(EDGE_CASE_SEEDS)
                   and edge_count < TARGET_EDGE_CASES):

                missing_indices = [
                    i for i in range(len(EDGE_CASE_SEEDS))
                    if i not in progress["edge_cases"]["completed_seeds_in_current_pass"]
                ]

                seed_succeeded_this_iter = False
                for seed_idx in missing_indices:
                    if edge_count >= TARGET_EDGE_CASES:
                        break

                    module, issue_type, has_error_code, seed_text = EDGE_CASE_SEEDS[seed_idx]

                    error_code = ""
                    if has_error_code:
                        error_code = random.choice(ERROR_CODES.get(module, ["SY 001"]))

                    print(f"  Pass {pass_idx + 1}/{EDGE_PASSES} | "
                          f"Seed {seed_idx + 1}/{len(EDGE_CASE_SEEDS)} | "
                          f"{module} - {issue_type}", end="")

                    # Edge cases use generate_incoming_variations() — the seeds are
                    # deliberately ambiguous/vague/cross-module incident-style tickets
                    variations = generate_incoming_variations(
                        seed_text      = seed_text,
                        module         = module,
                        issue_type     = issue_type,
                        has_error_code = has_error_code,
                        error_code     = error_code,
                    )

                    print(f" -> {len(variations)} tickets")

                    if not variations or len(variations) < int(VARIATIONS_PER_PASS * 0.75):
                        print(f"  WARNING: Seed {seed_idx + 1} in Pass {pass_idx + 1} "
                              f"did not return sufficient variations (>=75%) — retrying seed")
                        continue

                    for ticket_text in variations:
                        if edge_count >= TARGET_EDGE_CASES:
                            break

                        ticket_id = f"EDG-{ticket_counter:06d}"
                        ticket_counter += 1

                        # Write to incoming tickets CSV
                        inc_writer.writerow({
                            "ticket_id":      ticket_id,
                            "text":           ticket_text,
                            "ticket_type":    "edge_case",
                            "module":         module,
                            "issue_type":     issue_type,
                            "has_error_code": has_error_code,
                            "error_code":     error_code,
                        })

                        # Edge cases also get resolved ticket records
                        # because the RAG knowledge base needs to handle them
                        resolved = build_resolved_record(
                            incoming_text = ticket_text,
                            module        = module,
                            issue_type    = issue_type,
                            error_code    = error_code,
                            ticket_id     = ticket_id,
                        )
                        resolved["ticket_type"] = "edge_case"
                        res_writer.writerow(resolved)
                        edge_count += 1

                    # Flush both files then save progress
                    inc_file.flush()
                    os.fsync(inc_file.fileno())
                    res_file.flush()
                    os.fsync(res_file.fileno())

                    # Save progress after confirmed disk write
                    if seed_idx not in progress["edge_cases"]["completed_seeds_in_current_pass"]:
                        progress["edge_cases"]["completed_seeds_in_current_pass"].append(seed_idx)
                    progress["edge_cases"]["total_count"] = edge_count
                    save_progress(progress)

                    seed_succeeded_this_iter = True

                    # 5 second delay between API calls — respects 15 RPM free tier limit
                    time.sleep(5)

                if seed_succeeded_this_iter:
                    consecutive_empty_passes = 0
                else:
                    consecutive_empty_passes += 1

                if consecutive_empty_passes >= 5:
                    print("\n" + "=" * 55)
                    print("DAILY RATE LIMIT EXHAUSTED")
                    print("All seeds returned empty 5 consecutive times.")
                    print("Progress saved. Run script again tomorrow.")
                    print("Current progress:")
                    print(f"  Incidents:   {inc_count:,} / {TARGET_INCIDENTS:,}")
                    print(f"  Development: {dev_count:,} / {TARGET_DEVELOPMENT:,}")
                    print(f"  Edge Cases:  {edge_count:,} / {TARGET_EDGE_CASES:,}")
                    print("=" * 55)
                    daily_limit_exhausted = True
                    break

                remaining = len(EDGE_CASE_SEEDS) - len(progress["edge_cases"]["completed_seeds_in_current_pass"])
                if remaining > 0 and edge_count < TARGET_EDGE_CASES:
                    print(f"  [Pass {pass_idx + 1}] Retrying {remaining} uncompleted edge case seed(s)...")
                    time.sleep(3)

            if daily_limit_exhausted:
                break

            # ── END OF PASS ────────────────────────────────────────────────
            if (len(progress["edge_cases"]["completed_seeds_in_current_pass"]) >= len(EDGE_CASE_SEEDS)
                    or edge_count >= TARGET_EDGE_CASES):
                if pass_idx >= progress["edge_cases"]["completed_passes"]:
                    progress["edge_cases"]["completed_passes"]                = pass_idx + 1
                    progress["edge_cases"]["completed_seeds_in_current_pass"] = []
                    save_progress(progress)

            print(f"  Pass {pass_idx + 1} done | Edge cases so far: {edge_count:,}")

        if daily_limit_exhausted:
            print("Generation paused. Rerun tomorrow to continue.")
            return

        print(f"\nPhase 3 complete: {edge_count:,} edge case tickets generated")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────
    # total_resolved = incidents + edge_cases only — dev requests have no resolved records
    total_incoming = inc_count + dev_count + edge_count
    total_resolved = inc_count + edge_count
    print("\n" + "=" * 55)
    print("GENERATION COMPLETE")
    print("=" * 55)
    print(f"Total incoming tickets:  {total_incoming:,}")
    print(f"  Incident tickets:      {inc_count:,}")
    print(f"  Development requests:  {dev_count:,}")
    print(f"  Edge cases:            {edge_count:,}")
    print(f"Total resolved records:  {total_resolved:,}")
    print(f"\nFiles written:")
    print(f"  {INCOMING_CSV}")
    print(f"  {RESOLVED_CSV}")
    print("=" * 55)


# =============================================================================
# RECONCILIATION FUNCTION
# Safety net to patch any gaps between incoming and resolved CSVs.
# Gaps can occur from mid-write interruptions even with os.fsync().
# =============================================================================

def reconcile_resolved_tickets():
    """
    Compare incoming_tickets.csv against resolved_tickets.csv by ticket_id
    and generate any missing resolved records.

    WHY THIS EXISTS: Even with side-by-side generation and os.fsync(), edge
    cases like mid-write interruption between the two CSV writes can leave
    gaps. This function finds any incident or edge_case ticket in incoming
    that has no matching record in resolved, then generates and appends them.

    Development tickets are excluded — they never get resolved records.

    Can be run standalone:
        python generate_tickets.py --reconcile-only
    """
    print("\n" + "=" * 55)
    print("RECONCILIATION — checking resolved vs incoming")
    print("=" * 55)

    if not os.path.exists(INCOMING_CSV):
        print("incoming_tickets.csv not found — nothing to reconcile.")
        return

    # Collect all ticket_ids from incoming that need resolved records
    incoming_ids  = {}
    with open(INCOMING_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ticket_type"] in ("incident", "edge_case"):
                incoming_ids[row["ticket_id"]] = row

    # Collect all ticket_ids already in resolved
    resolved_ids = set()
    if os.path.exists(RESOLVED_CSV):
        with open(RESOLVED_CSV, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                resolved_ids.add(row["ticket_id"])

    missing_ids = set(incoming_ids.keys()) - resolved_ids

    print(f"Incoming incident + edge tickets: {len(incoming_ids):,}")
    print(f"Existing resolved records:        {len(resolved_ids):,}")
    print(f"Missing resolved records:         {len(missing_ids):,}")

    if not missing_ids:
        print("No gaps found. resolved_tickets.csv is complete.")
        return

    resolved_fields = [
        "ticket_id", "module", "issue_type", "problem", "error_code",
        "root_cause", "fix", "tcodes_used", "resolution_time_minutes",
        "escalated", "created_date", "resolved_date", "ticket_type",
    ]

    with open(RESOLVED_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=resolved_fields)
        for i, ticket_id in enumerate(sorted(missing_ids)):
            row = incoming_ids[ticket_id]
            resolved = build_resolved_record(
                incoming_text = row["text"],
                module        = row["module"],
                issue_type    = row["issue_type"],
                error_code    = row.get("error_code", ""),
                ticket_id     = ticket_id,
            )
            resolved["ticket_type"] = row["ticket_type"]
            writer.writerow(resolved)

            # Print progress every 500 records for large reconciliation runs
            if (i + 1) % 500 == 0:
                print(f"  Reconciled {i + 1:,} / {len(missing_ids):,}...")

    print(f"Reconciliation complete. Added {len(missing_ids):,} records to resolved_tickets.csv.")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Usage:
    #   python generate_tickets.py                  full generation with auto-resume
    #   python generate_tickets.py --reconcile-only  fix resolved gaps only

    if len(sys.argv) > 1 and sys.argv[1] == "--reconcile-only":
        # Run only reconciliation — does not touch incoming_tickets.csv
        reconcile_resolved_tickets()
    else:
        # Full generation — safe to run multiple times, auto-resumes from checkpoint
        generate_all_tickets()
        # Always reconcile after generation to catch any gaps from interruptions
        reconcile_resolved_tickets()

    