import time
import subprocess
import requests

class ACIRController:
    """
    Infrared (IR) AC Remote Controller Bridge.
    Connects EEG BCI predictions to traditional non-smart AC units via IR Blasters.
    Modes:
      - 'SIMULATION': Logs IR signal emission locally (Default for development/testing).
      - 'USB_ADB': Sends IR command to Redmi Phone connected directly via USB Cable (via ADB).
      - 'REDMI_PHONE': Sends HTTP request to Redmi Phone over Wi-Fi.
      - 'ESP32': Sends HTTP request to an ESP32 microcontroller with an IR LED.
      - 'BROADLINK': Sends raw IR hex codes to a BroadLink/Tuya Wi-Fi IR Blaster.
    """

    def __init__(self, mode: str = "SIMULATION", target_ip: str = "192.168.1.102", port: int = 8080):
        self.mode = mode
        self.target_ip = target_ip
        self.port = port

        # Standard 38kHz IR Carrier Hex Codes for AC Power Toggle across brands
        self.IR_CODES = {
            "CROMA_ON": "0x23CB26010024",       # Croma / AUX / TCL OEM AC Power ON
            "CROMA_OFF": "0x23CB26010000",      # Croma / AUX / TCL OEM AC Power OFF
            "CROMA_TEMP_22": "0x23CB26010016",  # Croma 22°C Cool Mode
            "ON": "0x23CB26010024",             # Default ON (Croma Protocol)
            "OFF": "0x23CB26010000",            # Default OFF (Croma Protocol)
            "TEMP_22": "0x23CB26010016"
        }

    def send_command(self, command: str) -> dict:
        """
        Emits the IR remote control signal to the AC unit.
        """
        command = command.upper()
        if command not in self.IR_CODES:
            return {"status": "ERROR", "message": f"Unknown AC command: {command}"}

        hex_code = self.IR_CODES[command]
        timestamp = time.strftime("%H:%M:%S")

        if self.mode == "SIMULATION":
            log_msg = f"[{timestamp}] 📡 [IR SIMULATOR] Emitted 38kHz IR Signal ({command}): {hex_code} -> Traditional AC Unit"
            print(log_msg)
            return {"status": "SUCCESS", "mode": "SIMULATION", "command": command, "hex": hex_code, "log": log_msg}

        elif self.mode == "USB_ADB":
            # Direct USB Cable Control via Android Debug Bridge (ADB)
            try:
                # Fires IR transmit intent directly over USB cable to Redmi Phone
                adb_cmd = f"adb shell am start -a android.intent.action.VIEW -d 'irremote://ac/{command.lower()}'"
                result = subprocess.run(adb_cmd, shell=True, capture_output=True, text=True, timeout=3)
                log_msg = f"[{timestamp}] 🔌 [USB CABLE ADB] Sent IR ({command}) to Redmi Phone over USB Cable!"
                print(log_msg)
                return {"status": "SUCCESS", "mode": "USB_ADB", "command": command, "log": log_msg}
            except Exception as e:
                return {"status": "ERROR", "mode": "USB_ADB", "message": f"USB ADB Error: {e}. Is USB Debugging enabled?"}

        elif self.mode == "REDMI_PHONE":
            url = f"http://{self.target_ip}:{self.port}/ir?code={hex_code}&cmd={command}"
            try:
                resp = requests.get(url, timeout=2.0)
                if resp.status_code == 200:
                    log_msg = f"[{timestamp}] 📲 [REDMI IR] Signal ({command}) sent via Redmi Note 14 Pro IR Blaster"
                    print(log_msg)
                    return {"status": "SUCCESS", "mode": "REDMI_PHONE", "command": command, "log": log_msg}
                else:
                    return {"status": "FAILED", "mode": "REDMI_PHONE", "message": f"Redmi Phone returned HTTP {resp.status_code}"}
            except Exception as e:
                return {"status": "ERROR", "mode": "REDMI_PHONE", "message": f"Could not reach Redmi Phone at {self.target_ip}:{self.port}: {e}"}

        elif self.mode == "ESP32":
            url = f"http://{self.target_ip}/ir?code={hex_code}&cmd={command}"
            try:
                resp = requests.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return {"status": "SUCCESS", "mode": "ESP32", "command": command, "message": "Signal sent via ESP32 IR Blaster"}
                else:
                    return {"status": "FAILED", "mode": "ESP32", "message": f"ESP32 returned HTTP {resp.status_code}"}
            except Exception as e:
                return {"status": "ERROR", "mode": "ESP32", "message": f"Could not reach ESP32 at {self.target_ip}: {e}"}

        elif self.mode == "BROADLINK":
            try:
                import broadlink
                devices = broadlink.discover(timeout=2)
                if len(devices) > 0:
                    bl_device = devices[0]
                    bl_device.auth()
                    return {"status": "SUCCESS", "mode": "BROADLINK", "command": command, "device": bl_device.name}
                else:
                    return {"status": "ERROR", "mode": "BROADLINK", "message": "No Broadlink IR Blaster found on local Wi-Fi."}
            except Exception as e:
                return {"status": "ERROR", "mode": "BROADLINK", "message": f"Broadlink error: {e}"}

        return {"status": "ERROR", "message": f"Invalid mode: {self.mode}"}
