"""Synthetic IT ticket generator (fallback dataset when no real data is supplied).

Run:  python data/generate.py [--n 1800]
Writes data/tickets.json. Deterministic (seed 42). Labels come from the templates; priority comes from
the same rules the app uses (app/taxonomy.rule_priority), which is stated in the README.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.taxonomy import DEFAULT_TAXONOMY, rule_priority  # noqa: E402

# subcategory -> (assignment group, applications, KB ids used in resolution notes)
META = {
    "Password Reset": ("Identity & Access Management", ["Active Directory", "Workday", "ServiceNow"], ["KB-0001"]),
    "Account Lockout": ("Identity & Access Management", ["Active Directory", "Outlook"], ["KB-0002", "KB-0033"]),
    "MFA Issue": ("Identity & Access Management", ["Microsoft Authenticator", "Okta"], ["KB-0003"]),
    "Access Request": ("Identity & Access Management", ["SharePoint", "Salesforce", "Jira", "Workday"], ["KB-0004"]),
    "VPN": ("Network Operations", ["GlobalProtect"], ["KB-0005", "KB-0006"]),
    "WiFi": ("Network Operations", ["CORP-WIFI"], ["KB-0007", "KB-0036"]),
    "DNS/Connectivity": ("Network Operations", ["Intranet", "File Server"], ["KB-0008"]),
    "Proxy/Firewall": ("Network Operations", ["Web Proxy", "Salesforce"], ["KB-0009"]),
    "Laptop": ("End-User Computing", ["Windows 11"], ["KB-0010", "KB-0011"]),
    "Printer": ("End-User Computing", ["Print Server"], ["KB-0012", "KB-0034"]),
    "Peripheral": ("End-User Computing", ["Docking Station"], ["KB-0013"]),
    "Email/Outlook": ("Messaging Team", ["Outlook"], ["KB-0014", "KB-0015"]),
    "SAP": ("SAP Basis", ["SAP"], ["KB-0016", "KB-0035"]),
    "Software Installation": ("Desktop Engineering", ["Software Center"], ["KB-0017"]),
    "Application Crash": ("Application Support", ["Teams", "Salesforce", "ServiceNow", "Excel"], ["KB-0018", "KB-0019"]),
    "DB Performance": ("DBA Team", ["Oracle DB", "SQL Server"], ["KB-0020"]),
    "DB Connection": ("DBA Team", ["Oracle DB", "SQL Server"], ["KB-0021"]),
    "Backup/Restore": ("DBA Team", ["SQL Server", "Oracle DB"], ["KB-0022"]),
    "How-To": ("Service Desk L1", ["Outlook", "Windows 11", "SharePoint"], ["KB-0023"]),
    "License Request": ("Software Asset Management", ["Visio", "MS Project", "Adobe Acrobat"], ["KB-0024"]),
    "Mobile Device": ("Mobile Device Management", ["Intune Company Portal", "Outlook Mobile"], ["KB-0025", "KB-0026"]),
    "OneDrive/SharePoint Sync": ("Collaboration Services", ["OneDrive", "SharePoint", "Teams"], ["KB-0027"]),
    "Web Application Error": ("Application Support", ["HR Portal", "Expense Portal", "Intranet", "ServiceNow"], ["KB-0032"]),
    "Phishing": ("Security Operations", ["Outlook"], ["KB-0028"]),
    "Malware/Antivirus": ("Security Operations", ["Microsoft Defender"], ["KB-0029"]),
    "Server/Storage": ("Infrastructure Operations", ["Windows Server", "Linux VM", "File Server"], ["KB-0030", "KB-0031"]),
}

# subcategory -> list of (title, body). Slots: {app} {code} {when} {loc}
T = {
    "Password Reset": [
        ("Forgot password", "I forgot my password and cannot log in to my laptop {when}."),
        ("Password expired", "My password expired and the login screen says the password is incorrect even after I changed it."),
        ("Need password reset", "Please reset my password for {app}, it keeps saying invalid credentials."),
        ("Password change not working", "Changed my password this morning but the new password is not accepted on {app}."),
        ("Reset password", "Cannot remember my Windows password after returning from leave, need a reset."),
        ("Self-service reset fails", "The self-service password reset portal says I am not enrolled, I need my password reset."),
    ],
    "Account Lockout": [
        ("Account locked", "My account has been locked {when}. Error 'Your account has been locked' on every application."),
        ("Keeps getting locked out", "I get locked out every few minutes even after the service desk unlocked me."),
        ("Locked out of {app}", "Locked out of {app} after several failed sign-ins, please unlock."),
        ("Account lockout repeated", "Repeated account lockouts {when}, I think my phone still has the old password."),
        ("Cannot log in, account locked", "Login fails everywhere at the same time, message says account locked."),
    ],
    "MFA Issue": [
        ("MFA push not arriving", "The authenticator push notification never arrives so I cannot complete sign in."),
        ("New phone, lost authenticator", "I changed my phone and lost the authenticator app, cannot pass MFA."),
        ("MFA code invalid", "The MFA code from {app} is always rejected as invalid {when}."),
        ("Two-factor not working", "Two-factor authentication fails, the verification code is not accepted."),
        ("Stuck at MFA prompt", "Sign in is stuck at the multi-factor authentication prompt, no notification on my phone."),
    ],
    "Access Request": [
        ("Access to shared folder", "I need access to the Finance shared folder on {app}, getting access denied."),
        ("New joiner access", "New joiner in our team needs access to {app} and the team SharePoint site."),
        ("Permission denied", "Getting 'You don't have permission' when opening the project site on {app}."),
        ("Need role in {app}", "Please grant me the reporting role in {app}, my manager has approved."),
        ("Access request for {app}", "Requesting read access to the {app} sales dashboard for my new role."),
    ],
    "VPN": [
        ("VPN not connecting", "GlobalProtect VPN is stuck on connecting {when}, error {code}."),
        ("VPN gateway unreachable", "VPN says gateway unreachable, I am working from home and cannot reach internal sites."),
        ("VPN keeps disconnecting", "VPN drops every few minutes and my Teams calls freeze {when}."),
        ("VPN very slow", "Very slow file transfers over the VPN, the file server is almost unusable."),
        ("VPN connected but no access", "VPN shows connected but intranet pages do not open."),
        ("VPN certificate expired", "VPN client reports the certificate has expired and refuses to connect."),
    ],
    "WiFi": [
        ("Cannot connect to Wi-Fi", "My laptop cannot connect to CORP-WIFI on {loc}, it keeps asking for credentials."),
        ("Wi-Fi no internet", "Wi-Fi is connected but there is no internet {when}."),
        ("Wi-Fi network not visible", "The laptop does not see the CORP-WIFI network at all on {loc}."),
        ("Wireless keeps dropping", "Wireless connection drops every few minutes in the meeting rooms on {loc}."),
        ("Wi-Fi authentication failed", "Wi-Fi authentication failed after the laptop was reimaged."),
    ],
    "DNS/Connectivity": [
        ("Internal site not resolving", "Intranet site does not open by name but works by IP address, error DNS_PROBE_FINISHED_NXDOMAIN."),
        ("Network drive not found", "Mapped network drive shows path not found {when}."),
        ("Cannot reach server", "Cannot reach the file server by host name, ping by IP works."),
        ("No network connectivity", "No network connectivity on my desk port on {loc}, cable is plugged in."),
        ("DNS error", "Websites fail with a DNS error while other colleagues can browse normally."),
    ],
    "Proxy/Firewall": [
        ("Website blocked", "A vendor website is blocked by corporate policy, I need it for a supplier contract."),
        ("Proxy prompt keeps appearing", "The proxy authentication prompt keeps appearing every time I open the browser."),
        ("Firewall blocking partner system", "Connection to the partner SFTP system times out, looks like a firewall block on port 22."),
        ("Need URL whitelisted", "Please whitelist a training website, it shows 'Blocked by corporate policy'."),
        ("API calls blocked", "Outbound calls from our test server to the payment provider sandbox are blocked by the firewall."),
    ],
    "Laptop": [
        ("Laptop very slow", "My laptop is extremely slow {when}, takes over 10 minutes to start and the disk is at 100%."),
        ("Laptop overheating", "Laptop fan is loud and the laptop gets very hot, then freezes."),
        ("Blue screen", "Laptop shows a blue screen CRITICAL_PROCESS_DIED and restarts {when}."),
        ("Laptop not powering on", "My laptop does not turn on at all, charger light is on."),
        ("Black screen after logo", "Laptop shows a black screen after the manufacturer logo."),
        ("Battery drains quickly", "Battery drains from full to empty in one hour, battery health warning shown."),
    ],
    "Printer": [
        ("Printer not printing", "The printer on {loc} is not printing, jobs stay in the queue."),
        ("Printer offline", "Printer shows offline on my laptop although it is switched on."),
        ("Paper jam", "Paper jam on the {loc} printer, the panel shows an error and nothing prints."),
        ("Cannot add printer", "Cannot add the new printer from the print server, driver install fails."),
        ("Print jobs stuck", "Print jobs are stuck in the queue and cannot be deleted."),
    ],
    "Peripheral": [
        ("External monitor no signal", "External monitor shows no signal when connected through the docking station."),
        ("Docking station not working", "Docking station does not detect my keyboard, mouse or monitors {when}."),
        ("Keyboard not responding", "Wireless keyboard stopped responding, replaced batteries but still nothing."),
        ("Second screen not detected", "Second screen is not detected after the Windows update."),
        ("Mouse lagging", "Mouse pointer lags and freezes when connected to the dock."),
    ],
    "Email/Outlook": [
        ("Outlook not opening", "Outlook is stuck on 'Loading profile' {when} and never opens."),
        ("Outlook crashes on attachments", "Outlook crashes every time I open a PDF attachment."),
        ("Emails stuck in outbox", "Emails are stuck in the Outbox and do not send."),
        ("Mailbox full", "I get a warning that my mailbox is almost full and cannot send emails."),
        ("Not receiving external emails", "I am not receiving emails from external customers {when}."),
        ("Outlook not responding", "Outlook is not responding and freezes when searching emails."),
    ],
    "SAP": [
        ("SAP login failed", "SAP GUI shows 'Partner not reached' when I try to log in {when}."),
        ("SAP user locked", "SAP logon fails with user locked after password attempts."),
        ("SAP short dump", "Transaction VA01 ends with a short dump in SAP, cannot create sales orders."),
        ("SAP very slow", "SAP transactions are very slow and time out when posting invoices."),
        ("SAP GUI error", "SAP GUI throws an error on startup, the system entry seems missing."),
    ],
    "Software Installation": [
        ("Install software", "Please install Python and VS Code on my laptop, I need it for a project."),
        ("Installation needs admin", "Installation fails with administrator rights required."),
        ("Software Center stuck", "Software Center install is stuck at 0% for the Zoom package."),
        ("Need {app} installed", "Need Power BI Desktop installed from {app} for monthly reporting."),
        ("Update failed", "The update for Adobe Reader fails with error {code} in {app}."),
    ],
    "Application Crash": [
        ("{app} crashes", "{app} closes unexpectedly when I open a large file {when}."),
        ("Teams white screen", "Microsoft Teams shows a white screen and keeps crashing on start."),
        ("Teams audio not working", "Others cannot hear me in Teams calls, microphone not detected."),
        ("Application has stopped working", "Error 'Application has stopped working' when launching {app}."),
        ("{app} hangs after update", "{app} hangs after the latest update and must be killed in Task Manager."),
    ],
    "DB Performance": [
        ("Reports very slow", "Monthly reports from {app} take much longer than usual {when}."),
        ("Query timeout", "The application shows query timeout errors when loading the customer list."),
        ("Database high CPU", "High CPU on the {app} database server, users complain about slowness."),
        ("Slow queries", "Queries on the orders table are slow after the data load."),
        ("Blocking sessions", "Batch job is blocked by another session on the {app} database."),
    ],
    "DB Connection": [
        ("Cannot connect to database", "Application cannot connect to the {app} database, error ORA-12541: TNS:no listener."),
        ("Login failed for service account", "Error 'Login failed for user' for the app service account after password rotation."),
        ("Connection pool exhausted", "Connection pool exhausted errors in the application logs {when}."),
        ("DB unreachable", "The {app} database is unreachable from the application server."),
        ("Database connection timeout", "Connection to the {app} database times out from the reporting server."),
    ],
    "Backup/Restore": [
        ("Backup failed", "Nightly backup job for the {app} database failed with error {code}."),
        ("Backup disk full", "Backup disk is full and the backup job did not run."),
        ("Restore request", "Please restore the {app} database to yesterday 6 PM, some records were deleted by mistake."),
        ("Restore a table", "Need a single table restored from last week's backup."),
        ("Backup verification", "Backups have failed two nights in a row on the {app} server."),
    ],
    "How-To": [
        ("How to set out of office", "How do I set up an out of office reply in Outlook?"),
        ("How to map a drive", "How can I map a network drive on my new laptop?"),
        ("Share file securely", "How do I share a large file securely with an external partner?"),
        ("Change display language", "Is it possible to change the Windows display language to French?"),
        ("Add shared mailbox", "How do I add a shared mailbox to Outlook?"),
    ],
    "License Request": [
        ("Need Visio licence", "I need a Visio licence for process diagrams, manager approved."),
        ("Licence expired", "{app} says the licence has expired and opens in read-only mode."),
        ("Adobe Acrobat Pro request", "Requesting an Adobe Acrobat Pro licence to edit PDFs."),
        ("MS Project licence", "Please provide an MS Project licence for the new programme manager."),
        ("Trial ended", "The {app} trial ended, I need a full licence to continue."),
    ],
    "Mobile Device": [
        ("Company phone email not syncing", "Email stopped syncing on my company phone {when}, the Outlook app shows 'cannot connect'."),
        ("Cannot enrol phone", "Cannot enrol my new phone in {app}, it says the device does not meet requirements."),
        ("Lost company phone", "I lost my company phone on the train, please wipe it."),
        ("Phone not compliant", "My phone shows 'not compliant' in {app} and blocks Teams and email."),
        ("Phone update blocked", "Company iPhone cannot install the required iOS update, storage is full and {app} warns me."),
        ("Phone stolen", "My company phone was stolen {when}, it has my work email on it."),
    ],
    "OneDrive/SharePoint Sync": [
        ("OneDrive not syncing", "OneDrive is not syncing {when}, files show a red cross."),
        ("Sync conflict copies", "OneDrive keeps creating duplicate files with my laptop name added, sync conflict."),
        ("SharePoint library not syncing", "The SharePoint library I synced to File Explorer stopped updating {when}."),
        ("Files missing after sync", "Some files disappeared from my OneDrive folder after the sync, I need them back."),
        ("OneDrive storage full", "OneDrive says my storage is full and stopped uploading."),
        ("Synced file locked", "Cannot open a synced Excel file from {app}, it says the file is locked by another user."),
    ],
    "Web Application Error": [
        ("{app} shows HTTP 500", "{app} shows 'HTTP 500 Internal Server Error' when I submit the form {when}."),
        ("502 Bad Gateway", "Getting 502 Bad Gateway on {app} {when}."),
        ("Login loop on {app}", "{app} keeps redirecting me back to the sign-in page in a loop."),
        ("{app} not loading", "{app} page loads forever and then shows 'service unavailable'."),
        ("Form submit fails", "The submit button on {app} does nothing and the page shows error ID {code}."),
    ],
    "Phishing": [
        ("Suspicious email", "I received a suspicious email asking me to confirm my password through a link."),
        ("Clicked phishing link", "I clicked a link in an email that looked like it came from IT and entered my password {when}."),
        ("Fake invoice email", "Got an invoice email from an unknown sender with a zip attachment, I did not open it."),
        ("CEO fraud email", "An email that looks like it is from our CEO asks me to buy gift cards urgently."),
        ("Report phishing", "How do I report a phishing email? Several colleagues got the same message."),
    ],
    "Malware/Antivirus": [
        ("Antivirus alert", "Microsoft Defender shows a threat detected alert on my laptop {when}."),
        ("Pop-ups and strange programs", "My browser keeps opening pop-up ads and a program I did not install starts at login."),
        ("Files encrypted", "My files have strange extensions and a ransom note appeared on the desktop."),
        ("Defender turned off", "Defender says real-time protection is turned off and I cannot turn it on."),
        ("Virus warning from USB", "Got a virus warning after plugging in a USB stick from a supplier."),
    ],
    "Server/Storage": [
        ("Server disk full", "Disk C on the {app} application server is 98% full, the app stopped writing logs."),
        ("Server not responding", "The {app} server is not responding to ping or RDP {when}."),
        ("VM will not start", "Our test VM is powered off and will not start, error {code}."),
        ("File share out of space", "The department file share is out of space, users cannot save files."),
        ("High memory on server", "The {app} server is using 99% memory and services keep restarting."),
    ],
}

# a few non-English tickets (multilingual embeddings demo)
FOREIGN = {
    "VPN": ["La VPN non si connette da stamattina, errore {code}.", "La VPN si disconnette continuamente durante le chiamate."],
    "Password Reset": ["Ho dimenticato la password e non riesco ad accedere al portatile."],
    "Printer": ["La stampante del terzo piano non stampa, i lavori restano in coda."],
    "Email/Outlook": ["Outlook non si apre, resta bloccato su 'Caricamento profilo'."],
    "Account Lockout": ["Mi account está bloqueado después de varios intentos fallidos."],
    "Phishing": ["Ho ricevuto un'email sospetta che mi chiede di confermare la password."],
    "Mobile Device": ["No recibo el correo en mi teléfono de empresa desde ayer."],
}

CODES = ["809", "0x80070005", "0x80004005", "720", "1603", "ERR-5021", "0x8024402c", "E_FAIL 0x80004005"]
WHENS = ["since this morning", "after the latest update", "since yesterday", "for the last two hours",
         "after I changed my password", "intermittently this week", "since Monday", "today"]
LOCS = ["floor 3", "floor 5", "the Pune office", "the Chennai office", "building B", "the London office"]
INTROS = ["", "", "", "Hi team, ", "Hello, ", "Good morning, ", "Hi, "]
SCOPES = ["", "", "", "", " Only affects me.", " It only happens on my laptop.",
          " Several users in my team have the same problem.", " The whole team is blocked.",
          " This is blocking month-end close, please treat as urgent.",
          " All users in the office are affected, looks like an outage.",
          " Production is down for customers."]
# realism: sentences that mention OTHER topics (keyword confusion), vague tickets, agent mislabelling
DISTRACTORS = ["I am working from home on VPN today.", "I already restarted Outlook and the laptop.",
               "My laptop got a Windows update last week.", "The printer next to me works fine.",
               "I reset my password yesterday for another reason.", "Teams and email are working normally.",
               "This happened right after the SAP release.", "I am connected to the office Wi-Fi.",
               "Our database team said nothing changed.", "I have admin rights on this machine.",
               "OneDrive was syncing slowly yesterday too.", "My phone gets email without problems."]
VAGUE = ["Not working since this morning, please help.", "Getting an error, cannot continue my work.",
         "Something is wrong again, same as last week.", "Please call me, the issue is back.",
         "It is very slow and keeps failing."]
P_DISTRACT, P_VAGUE, P_MISLABEL = 0.25, 0.04, 0.03
NAMES = ["Priya Sharma", "John Miller", "Ananya Rao", "Marco Rossi", "Fatima Khan", "Liam Chen", "Sofia Garcia"]
RES_NOTES = [
    "Resolved by following {kb}: {step} User confirmed the issue is fixed.",
    "Applied {kb}. {step} Closed after user confirmation.",
    "Fixed per {kb}. {step}",
]


def pii(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    email = name.lower().replace(" ", ".") + "@corp.example.com"
    choice = rng.random()
    if choice < 0.35:
        return f" Contact me at {email} or +91 98{rng.randint(10, 99)}{rng.randint(100000, 999999)}."
    if choice < 0.6:
        return f" My laptop IP is 10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}."
    if choice < 0.8:
        return f" Employee ID EMP-{rng.randint(10000, 99999)}, {name}."
    return f" Asset tag AST-{rng.randint(100000, 999999)}."


def fill(s: str, rng: random.Random, app: str) -> str:
    return s.format(app=app, code=rng.choice(CODES), when=rng.choice(WHENS), loc=rng.choice(LOCS))


def generate(n: int = 1800, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    kb = {a["kb_id"]: a for a in json.loads((ROOT / "data" / "kb" / "articles.json").read_text(encoding="utf-8"))}
    sub_to_cat = {s: c for c, subs in DEFAULT_TAXONOMY.items() for s in subs}
    subs = list(T)
    # mild class imbalance, like real service desks
    weights = [1.6 if s in {"Password Reset", "VPN", "Email/Outlook", "Laptop", "Printer", "Account Lockout"} else 1.0
               for s in subs]
    end = datetime(2026, 11, 1)
    tickets = []
    for i in range(1, n + 1):
        sub = rng.choices(subs, weights)[0]
        group, apps, kb_ids = META[sub]
        app = rng.choice(apps)
        roll = rng.random()
        if roll < P_VAGUE:
            title, body = None, rng.choice(VAGUE) + f" ({app})"
        elif sub in FOREIGN and roll < P_VAGUE + 0.06:
            title, body = None, fill(rng.choice(FOREIGN[sub]), rng, app)
        else:
            title, body = rng.choice(T[sub])
            title, body = fill(title, rng, app), rng.choice(INTROS) + fill(body, rng, app) + rng.choice(SCOPES)
            if rng.random() < P_DISTRACT:
                body += " " + rng.choice(DISTRACTORS)
        label = sub
        if rng.random() < P_MISLABEL:                 # an agent picked the wrong subcategory
            label = rng.choice([s for s in subs if s != sub])
        if rng.random() < 0.3:
            body += pii(rng)
        env = rng.choices(["prod", "uat", "dev"], [0.8, 0.12, 0.08])[0]
        status = rng.choices(["resolved", "closed", "open"], [0.55, 0.25, 0.2])[0]
        resolution = None
        if status != "open":
            k = kb[rng.choice(kb_ids)]
            resolution = rng.choice(RES_NOTES).format(kb=k["kb_id"], step=rng.choice(k["resolution_steps"]) + ".")
        text = (title + ". " if title else "") + body
        tickets.append({
            "id": f"TKT-{i:05d}",
            "title": title,
            "text": body,
            "category": sub_to_cat[label],
            "subcategory": label,
            "priority": rule_priority(text, env),
            "status": status,
            "application": app,
            "environment": env,
            "assignment_group": group,
            "created_at": (end - timedelta(minutes=rng.randint(0, 365 * 24 * 60))).isoformat(timespec="minutes"),
            "resolution": resolution,
            "synthetic_fields": ["all (synthetic dataset)"],
        })
    return tickets


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1800)
    args = ap.parse_args()
    out = ROOT / "data" / "tickets.json"
    data = generate(args.n)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(data)} tickets to {out}")
