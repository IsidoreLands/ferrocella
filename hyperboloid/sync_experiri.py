import git
import os
import shutil
import schedule
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
REPO_URL = "https://github.com/IsidoreLands/aether.git"
LOCAL_AETHER_PATH = "/tmp/aether_clone"
EXPERIRI_SRC = os.path.join(LOCAL_AETHER_PATH, "experiri")
EXPERIRI_DEST = "experiri"

def sync_experiri():
    logging.info("Starting sync of aether/experiri...")
    try:
        if os.path.exists(LOCAL_AETHER_PATH):
            shutil.rmtree(LOCAL_AETHER_PATH)
        repo = git.Repo.clone_from(REPO_URL, LOCAL_AETHER_PATH)
        logging.info(f"Cloned aether repo to {LOCAL_AETHER_PATH}")
        
        if os.path.exists(EXPERIRI_DEST):
            shutil.rmtree(EXPERIRI_DEST)
        shutil.copytree(EXPERIRI_SRC, EXPERIRI_DEST)
        logging.info(f"Copied {EXPERIRI_SRC} to {EXPERIRI_DEST}")
        
        shutil.rmtree(LOCAL_AETHER_PATH)
        logging.info("Cleaned up temporary clone")
    except Exception as e:
        logging.error(f"Sync failed: {e}")

if __name__ == "__main__":
    sync_experiri()  # Run once on start
    schedule.every(1).hours.do(sync_experiri)  # Schedule hourly sync
    logging.info("Starting sync scheduler...")
    while True:
        schedule.run_pending()
        time.sleep(60)
