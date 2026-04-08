import time, subprocess
for i in range(5):
    subprocess.run(["sudo", "tee", "/sys/class/leds/ACT/brightness"], input=b"1", stdout=subprocess.DEVNULL)
    time.sleep(0.3)
    subprocess.run(["sudo", "tee", "/sys/class/leds/ACT/brightness"], input=b"0", stdout=subprocess.DEVNULL)
    time.sleep(0.3)
print("Done!")
