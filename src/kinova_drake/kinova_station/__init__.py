from .common import EndEffectorTarget, GripperTarget
from .simulation_station import KinovaStation

__all__ = [
    'KinovaStation', 'EndEffectorTarget', 'GripperTarget',
]

try:
    from kinova_drake.kinova_station.hardware_station import KinovaStationHardwareInterface
    __all__.append('KinovaStationHardwareInterface')
except ImportError as error:
    # Don't import the hardware interface if the kortex API isn't installed
    print(error)
    print("Kortex API not detected: disabling hardware interface")
    pass