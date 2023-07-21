# pip install plyer
import time
from plyer import notification

if __name__ == '__main__':
    while True:
        notification.notify(
            title="Alert",
            message="Alert from python",
            timeout=10
        )
        time.sleep(2000)
