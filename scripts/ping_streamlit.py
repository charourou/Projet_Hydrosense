"""Visite l'app Streamlit Community Cloud et la réveille si elle est en veille.

Streamlit Cloud met les apps en veille après ~12h d'inactivité et sert alors
une page "Zzzz... this app has gone to sleep" avec un bouton
"Yes, get this app back up!". Un simple curl ne suffit donc pas :
ce script simule un vrai visiteur avec Playwright (Chromium headless).

Codes de sortie : 0 = app éveillée (déjà ou après réveil), 1 = échec.
"""

import sys

from playwright.sync_api import sync_playwright

APP_URL = "https://hydro-sense.streamlit.app"
WAKE_BUTTON_TEXT = "Yes, get this app back up!"
SLEEP_MARKERS = ("has gone to sleep", "zzz")

PAGE_LOAD_TIMEOUT_MS = 60_000
RENDER_WAIT_MS = 8_000
WAKE_UP_TIMEOUT_MS = 180_000
POLL_INTERVAL_MS = 5_000


def is_asleep(page) -> bool:
    """Détecte la page de veille via son texte ou le bouton de réveil."""
    if page.get_by_text(WAKE_BUTTON_TEXT).count() > 0:
        return True
    body_text = page.locator("body").inner_text().lower()
    return any(marker in body_text for marker in SLEEP_MARKERS)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Chargement de {APP_URL} ...")
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT_MS)
        # Laisse le temps à la page (app ou page de veille) de se rendre
        page.wait_for_timeout(RENDER_WAIT_MS)

        if not is_asleep(page):
            print("✅ L'app était déjà éveillée, rien à faire.")
            browser.close()
            return 0

        print("😴 Page de veille détectée, clic sur le bouton de réveil ...")
        wake_button = page.get_by_text(WAKE_BUTTON_TEXT)
        if wake_button.count() == 0:
            print("❌ Page de veille détectée mais bouton de réveil introuvable.")
            browser.close()
            return 1
        wake_button.first.click()

        # Attend que la page de veille disparaisse (le redémarrage peut être long)
        elapsed = 0
        while elapsed < WAKE_UP_TIMEOUT_MS:
            page.wait_for_timeout(POLL_INTERVAL_MS)
            elapsed += POLL_INTERVAL_MS
            if not is_asleep(page):
                print(f"🌅 L'app a été réveillée avec succès (après ~{elapsed // 1000}s).")
                browser.close()
                return 0
            print(f"   ... toujours en cours de réveil ({elapsed // 1000}s)")

        print("❌ L'app est toujours en veille après le délai d'attente.")
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
