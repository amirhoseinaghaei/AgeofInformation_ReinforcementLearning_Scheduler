import json 
class SimulationParameters():
    def __init__(self, JsonFile) -> None:
        with open(JsonFile) as f:
            self.SimulationParameters = json.load(f).get("SimulationParameters")
    def Configure(self):
        self.NumberOfBits = self.SimulationParameters.get("NumberOfBits")
        self.NumberOfCPUCycles = self.SimulationParameters.get("NumberOfCPUCycles")
        self.IntervalBetweenArrivals = self.SimulationParameters.get("IntervalBetweenArrivals")
        self.WindowSize = self.SimulationParameters.get("WindowSize")
        self.CPUSpeed = self.SimulationParameters.get("CPUSpeed")
        self.NumberOfTransmissionChannels = self.SimulationParameters.get("NumberOfTransmissionChannels")
        self.Deadline = self.SimulationParameters.get("Deadline")
