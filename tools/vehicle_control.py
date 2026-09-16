"""Auto-generated VehicleControl implementation."""

import math
import hashlib
import copy
from typing import List, Dict, Any, Optional, Tuple

class VehicleControl:
    """Vehicle control system class to manage various aspects of the car such as engine, doors, climate control, lights, and more."""
    
    def __init__(self, initial_config: dict) -> None:
        """Initializes the vehicle control system with the provided configuration."""
        self.remainingUnlockedDoors: int = initial_config.get("remainingUnlockedDoors", 0)
        self.fuelLevel: float = initial_config.get("fuelLevel", 15.0)
        self.batteryVoltage: float = initial_config.get("batteryVoltage", 12.8)
        self.engineState: str = initial_config.get("engineState", "stopped")
        self.doorStatus: Dict[str, str] = copy.deepcopy(initial_config.get("doorStatus", {
            "driver": "locked",
            "passenger": "locked",
            "rear_left": "locked",
            "rear_right": "locked"
        }))
        self.acTemperature: float = initial_config.get("acTemperature", 22.0)
        self.fanSpeed: int = initial_config.get("fanSpeed", 60)
        self.acMode: str = initial_config.get("acMode", "auto")
        self.humidityLevel: float = initial_config.get("humidityLevel", 45.0)
        self.headLightStatus: str = initial_config.get("headLightStatus", "off")
        self.parkingBrakeStatus: str = initial_config.get("parkingBrakeStatus", "released")
        self.parkingBrakeForce: float = initial_config.get("parkingBrakeForce", 0.0)
        self.slopeAngle: float = initial_config.get("slopeAngle", 0.0)
        self.distanceToNextVehicle: float = initial_config.get("distanceToNextVehicle", 100.0)
        self.cruiseStatus: str = initial_config.get("cruiseStatus", "inactive")
        self.destination: str = initial_config.get("destination", "Grand Canyon")
        self.frontLeftTirePressure: float = initial_config.get("frontLeftTirePressure", 35.0)
        self.frontRightTirePressure: float = initial_config.get("frontRightTirePressure", 35.0)
        self.rearLeftTirePressure: float = initial_config.get("rearLeftTirePressure", 35.0)
        self.rearRightTirePressure: float = initial_config.get("rearRightTirePressure", 35.0)
        self.brakePedalStatus: str = initial_config.get("brakePedalStatus", "released")
        self.brakePedalForce: float = float(initial_config.get("brakePedalForce", 0.0))
        self.cruiseSpeed: float = float(initial_config.get("cruiseSpeed", 0.0))
        self.currentSpeed: float = float(
            initial_config.get(
                "currentSpeed",
                self.cruiseSpeed if self.cruiseStatus == "active" else 0.0,
            )
        )
        self.outsideTemperatureC: float = float(
            initial_config.get("outsideTemperatureC", 22.0)
        )

    def activateParkingBrake(self, mode: str) -> Dict[str, Any]:
        """Activates the parking brake of the vehicle."""
        if mode not in ["engage", "release"]:
            return {"parkingBrakeStatus": self.parkingBrakeStatus, "_parkingBrakeForce": self.parkingBrakeForce, "_slopeAngle": self.slopeAngle}
        if mode == "engage":
            self.parkingBrakeStatus = "engaged"
            self.parkingBrakeForce = 500.0 * max(0.1, abs(math.sin(math.radians(self.slopeAngle))) + 0.1)
        else:
            self.parkingBrakeStatus = "released"
            self.parkingBrakeForce = 0.0
        return {
            "parkingBrakeStatus": self.parkingBrakeStatus,
            "_parkingBrakeForce": self.parkingBrakeForce,
            "_slopeAngle": self.slopeAngle
        }

    def adjustClimateControl(self, temperature: float, unit: str = 'celsius', fanSpeed: int = 50, mode: str = 'auto') -> Dict[str, Any]:
        """Adjusts the climate control of the vehicle."""
        if mode not in ["auto", "cool", "heat", "defrost"]:
            mode = "auto"
        if fanSpeed < 0:
            fanSpeed = 0
        elif fanSpeed > 100:
            fanSpeed = 100
        self.fanSpeed = fanSpeed
        self.acMode = mode
        
        if unit == "fahrenheit":
            self.acTemperature = (temperature - 32.0) * 5.0 / 9.0
        else:
            self.acTemperature = temperature
            
        if mode == "defrost":
            self.humidityLevel = max(30.0, self.humidityLevel - 15.0)
        elif mode == "cool":
            self.humidityLevel = max(30.0, self.humidityLevel - 10.0)
        elif mode == "heat":
            self.humidityLevel = min(60.0, self.humidityLevel + 5.0)
            
        return {
            "currentTemperature": self.acTemperature,
            "climateMode": self.acMode,
            "humidityLevel": self.humidityLevel
        }

    def displayCarStatus(self, option: str) -> Dict[str, Any]:
        """Displays the status of the vehicle based on the provided display option."""
        status: Dict[str, Any] = {}
        if option == "fuel":
            status["fuelLevel"] = self.fuelLevel
        elif option == "battery":
            status["batteryVoltage"] = self.batteryVoltage
        elif option == "doors":
            status["doorStatus"] = dict(self.doorStatus)
        elif option == "climate":
            status["currentACTemperature"] = self.acTemperature
            status["fanSpeed"] = self.fanSpeed
            status["climateMode"] = self.acMode
            status["humidityLevel"] = self.humidityLevel
        elif option == "headlights":
            status["headlightStatus"] = self.headLightStatus
        elif option == "parkingBrake":
            status["parkingBrakeStatus"] = self.parkingBrakeStatus
            status["parkingBrakeForce"] = self.parkingBrakeForce
            status["slopeAngle"] = self.slopeAngle
        elif option == "brakePadle":
            status["brakePedalStatus"] = self.brakePedalStatus
            status["brakePedalForce"] = self.brakePedalForce
        elif option == "engine":
            status["engineState"] = self.engineState
        return {"status": status}

    def display_log(self, messages: list) -> Dict[str, Any]:
        """Displays the log messages."""
        return {"log": list(messages)}

    def check_tire_pressure(self) -> Dict[str, Any]:
        """Return all tire pressures and classify the overall state."""
        tire_pressure = {
            "front_left": float(self.frontLeftTirePressure),
            "front_right": float(self.frontRightTirePressure),
            "rear_left": float(self.rearLeftTirePressure),
            "rear_right": float(self.rearRightTirePressure),
        }
        low_tires = sorted(name for name, value in tire_pressure.items() if value < 30.0)
        high_tires = sorted(name for name, value in tire_pressure.items() if value > 36.0)
        if low_tires:
            status = "low"
        elif high_tires:
            status = "high"
        else:
            status = "normal"
        return {
            "tire_pressure": tire_pressure,
            "unit": "PSI",
            "status": status,
            "low_tires": low_tires,
            "high_tires": high_tires,
        }

    def find_nearest_tire_shop(self) -> Dict[str, Any]:
        """Return a deterministic synthetic tire shop for the current route."""
        shops = [
            ("RoadReady Tire Center", "100 Service Road"),
            ("AllWheel Auto & Tire", "245 Market Street"),
            ("QuickTread Garage", "18 Highway Plaza"),
            ("SafeMiles Tire Shop", "730 Main Avenue"),
        ]
        seed = self.destination if self.destination and self.destination != "None" else "current_location"
        index = self._stable_int(seed.casefold()) % len(shops)
        shop_name, address = shops[index]
        distance = round(0.5 + (self._stable_int(seed + "|tire_shop") % 95) / 10.0, 1)
        return {
            "shop_name": shop_name,
            "address": address,
            "distance": distance,
            "unit": "miles",
        }

    def get_current_speed(self) -> Dict[str, Any]:
        """Return the current vehicle speed."""
        speed = self.currentSpeed if self.engineState == "running" else 0.0
        return {"current_speed": float(speed), "unit": "mph"}

    def get_outside_temperature_from_google(self) -> Dict[str, Any]:
        """Return a deterministic Google weather reading."""
        return {
            "temperature": round(self.outsideTemperatureC, 1),
            "unit": "celsius",
            "source": "google",
        }

    def get_outside_temperature_from_weather_com(self) -> Dict[str, Any]:
        """Return a deterministic weather.com reading.

        A small fixed provider offset creates realistic but replayable readings.
        """
        return {
            "temperature": round(self.outsideTemperatureC + 0.4, 1),
            "unit": "celsius",
            "source": "weather.com",
        }

    def releaseBrakePedal(self) -> Dict[str, Any]:
        """Release the brake pedal and clear applied force."""
        self.brakePedalStatus = "released"
        self.brakePedalForce = 0.0
        return {
            "brakePedalStatus": self.brakePedalStatus,
            "brakePedalForce": self.brakePedalForce,
        }

    @staticmethod
    def _stable_int(value: str) -> int:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    def estimate_distance(self, cityA: str, cityB: str) -> Dict[str, Any]:
        """Estimate a deterministic, symmetric distance in miles."""
        endpoints = sorted([str(cityA).strip(), str(cityB).strip()])
        if endpoints[0] == endpoints[1]:
            distance = 0.0
        else:
            hash_val = self._stable_int("|".join(endpoints))
            distance = round(100.0 + (hash_val % 4000) / 10.0, 1)
        return {
            "distance": distance,
            "unit": "miles",
            "intermediaryCities": []
        }

    def estimate_drive_feasibility_by_mileage(self, distance: float) -> Dict[str, Any]:
        """Estimates the mileage of the vehicle given the distance needed to drive."""
        mileage_per_gallon = 25.0
        can_drive = (self.fuelLevel * mileage_per_gallon) >= distance
        return {"canDrive": can_drive}

    def fillFuelTank(self, fuelAmount: float) -> Dict[str, Any]:
        """Fills the fuel tank of the vehicle. The fuel tank can hold up to 50 gallons."""
        max_tank_capacity = 50.0
        if fuelAmount < 0:
            fuelAmount = 0.0
        self.fuelLevel = min(max_tank_capacity, self.fuelLevel + fuelAmount)
        return {"fuelLevel": self.fuelLevel}

    def gallon_to_liter(self, gallon: float) -> Dict[str, Any]:
        """Converts the gallon to liter."""
        liter = gallon * 3.78541
        return {"liter": liter}

    def get_zipcode_based_on_city(self, city: str) -> Dict[str, Any]:
        """Get a deterministic synthetic ZIP code for a city name."""
        normalized_city = str(city).strip().casefold()
        hash_val = self._stable_int(normalized_city)
        zipcode = str(10000 + (hash_val % 90000))
        return {"zipcode": zipcode}

    def liter_to_gallon(self, liter: float) -> Dict[str, Any]:
        """Converts the liter to gallon."""
        gallon = liter / 3.78541
        return {"gallon": gallon}

    def lockDoors(self, unlock: bool, door: list) -> Dict[str, Any]:
        """Locks the doors of the vehicle."""
        valid_doors = ["driver", "passenger", "rear_left", "rear_right"]
        for d in door:
            if d in valid_doors:
                self.doorStatus[d] = "unlocked" if unlock else "locked"
        self.remainingUnlockedDoors = sum(1 for status in self.doorStatus.values() if status == "unlocked")
        lock_status = "unlocked" if unlock else "locked"
        return {
            "lockStatus": lock_status,
            "remainingUnlockedDoors": self.remainingUnlockedDoors
        }

    def pressBrakePedal(self, pedalPosition: float) -> Dict[str, Any]:
        """Presses the brake pedal based on pedal position."""
        if pedalPosition < 0:
            pedalPosition = 0.0
        elif pedalPosition > 1:
            pedalPosition = 1.0
            
        if pedalPosition > 0:
            self.brakePedalStatus = "pressed"
            self.brakePedalForce = pedalPosition * 500.0
            self.cruiseStatus = "inactive"
            self.cruiseSpeed = 0.0
        else:
            self.brakePedalStatus = "released"
            self.brakePedalForce = 0.0
            
        return {
            "brakePedalStatus": self.brakePedalStatus,
            "brakePedalForce": self.brakePedalForce
        }

    def setCruiseControl(self, speed: float, activate: bool, distanceToNextVehicle: float) -> Dict[str, Any]:
        """Sets the cruise control of the vehicle."""
        if activate:
            if 0 <= speed <= 120 and speed % 5 == 0:
                self.cruiseStatus = "active"
                self.cruiseSpeed = speed
                self.currentSpeed = speed
                self.distanceToNextVehicle = distanceToNextVehicle
            else:
                self.cruiseStatus = "inactive"
                self.cruiseSpeed = 0.0
        else:
            self.cruiseStatus = "inactive"
            self.cruiseSpeed = 0.0
            
        return {
            "cruiseStatus": self.cruiseStatus,
            "currentSpeed": self.cruiseSpeed,
            "distanceToNextVehicle": self.distanceToNextVehicle
        }

    def setHeadlights(self, mode: str) -> Dict[str, Any]:
        """Sets the headlights of the vehicle."""
        if mode in ["on", "auto"]:
            self.headLightStatus = "on"
        elif mode == "off":
            self.headLightStatus = "off"
        return {"headlightStatus": self.headLightStatus}

    def set_navigation(self, destination: str) -> Dict[str, Any]:
        """Navigates to the destination."""
        self.destination = destination
        return {"status": "navigating to " + self.destination}

    def startEngine(self, ignitionMode: str) -> Dict[str, Any]:
        """Starts the engine of the vehicle."""
        if ignitionMode == "START":
            self.engineState = "running"
        elif ignitionMode == "STOP":
            self.engineState = "stopped"
            self.currentSpeed = 0.0
            self.cruiseSpeed = 0.0
            self.cruiseStatus = "inactive"
        return {
            "engineState": self.engineState,
            "fuelLevel": self.fuelLevel,
            "batteryVoltage": self.batteryVoltage
        }
