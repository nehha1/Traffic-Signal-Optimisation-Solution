import time
from enum import Enum


class TrafficStatus(Enum):
    DENSE = 1
    SPARSE = 2
    FIRE = 3
    ACCIDENT = 4
    FIRE_BURST = 5
    FIRE_BRIGADE = 6
    AMBULANCE = 7
    TRUCK = 8


class LaneDirection(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


class Lane:
    def __init__(self, direction, status):
        self.direction = direction
        self.status = status


class Signal:
    def __init__(self, lane):
        self.lane = lane
        self.green = False
        self.straight = False
        self.left = False
        self.right = False
        self.duration = 0  # Duration in seconds


class TrafficSignal:
    def __init__(self):
        self.lanes = [
            Lane(LaneDirection.NORTH, TrafficStatus.SPARSE),
            Lane(LaneDirection.SOUTH, TrafficStatus.SPARSE),
            Lane(LaneDirection.EAST, TrafficStatus.SPARSE),
            Lane(LaneDirection.WEST, TrafficStatus.SPARSE)
        ]
        self.signals = [Signal(lane) for lane in self.lanes]

    def update_lane_status(self, lane_direction, status):
        for lane in self.lanes:
            if lane.direction == lane_direction:
                lane.status = status
                break

    def get_signal(self, lane_direction):
        for signal in self.signals:
            if signal.lane.direction == lane_direction:
                return signal

    def emergency_phase(self):
        emergency_priority = {
            TrafficStatus.ACCIDENT: 1,
            TrafficStatus.FIRE_BURST: 1,
            TrafficStatus.AMBULANCE: 2,
            TrafficStatus.FIRE_BRIGADE: 2,
            TrafficStatus.TRUCK: 3,
            TrafficStatus.DENSE: 4,
            TrafficStatus.SPARSE: 5
        }

        # Sort lanes by emergency priority
        emergency_lanes = sorted(
            [lane for lane in self.lanes if lane.status in emergency_priority],
            key=lambda lane: emergency_priority[lane.status]
        )

        phases = []
        for lane in emergency_lanes:
            signal = self.get_signal(lane.direction)
            signal.green = True
            signal.straight = True
            signal.left = True
            signal.right = True
            signal.duration = 30  # Emergency phase lasts for 30 seconds for high-priority emergencies
            phases.append([signal])

        return phases

    def calculate_signals(self):
        emergency_signals = self.emergency_phase()
        if emergency_signals:
            return emergency_signals

        dense_lanes = [lane for lane in self.lanes if lane.status == TrafficStatus.DENSE]
        sparse_lanes = [lane for lane in self.lanes if lane.status == TrafficStatus.SPARSE]

        phases = []
        timer_values = {
            'DENSE_GREEN': 45,
            'SPARSE_GREEN': 30,
            'YELLOW': 5,
            'RED': 30
        }

        if len(dense_lanes) == 3:  # All lanes dense
            for lane in self.lanes:
                signal = self.get_signal(lane.direction)
                signal.green = True
                signal.duration = timer_values['DENSE_GREEN']
                signal.straight = True
                signal.left = True
            phases.append([signal for lane in self.lanes])

        elif len(dense_lanes) == 2:  # 2 Dense and 1 Sparse
            for lane in dense_lanes:
                signal = self.get_signal(lane.direction)
                signal.green = True
                signal.duration = timer_values['DENSE_GREEN']
                signal.straight = True
                signal.left = True

            sparse_signal = self.get_signal((set(self.lanes) - set(dense_lanes)).pop().direction)
            sparse_signal.green = False
            sparse_signal.duration = timer_values['RED']
            phases.append([signal for signal in dense_lanes] + [sparse_signal])

        elif len(dense_lanes) == 1:  # 1 Dense and 2 Sparse
            dense_signal = self.get_signal(dense_lanes[0].direction)
            dense_signal.green = True
            dense_signal.duration = timer_values['DENSE_GREEN']
            dense_signal.straight = True
            phases.append([dense_signal])

            for lane in sparse_lanes:
                sparse_signal = self.get_signal(lane.direction)
                sparse_signal.green = False
                sparse_signal.duration = timer_values['RED']
                phases[-1].append(sparse_signal)

        elif len(dense_lanes) == 0:  # All Sparse
            for lane in self.lanes:
                signal = self.get_signal(lane.direction)
                signal.green = True
                signal.duration = timer_values['SPARSE_GREEN']
                signal.straight = True
            phases.append([signal for lane in self.lanes])

        return phases

    def display_phase(self, phase_num, signals):
        print(f"\nPhase {phase_num}:")
        print("=" * 30)

        for signal in signals:
            status = "Red"
            if signal.green:
                status = "Green"
                directions = []
                if signal.straight:
                    directions.append("Straight")
                if signal.left:
                    directions.append("Left")
                if signal.right:
                    directions.append("Right")
                direction_str = ", ".join(directions)
                print(f"Lane {signal.lane.direction.name}: {status} ({direction_str}), Duration: {signal.duration} seconds")
            else:
                print(f"Lane {signal.lane.direction.name}: {status}")
        print("=" * 30)

    def simulate_signals(self):
        phases = self.calculate_signals()

        for idx, phase in enumerate(phases):
            self.display_phase(idx + 1, phase)
            for signal in phase:
                if signal.green:
                    print(f"Turning green light on for {signal.lane.direction.name} lane for {signal.duration} seconds...")
                    time.sleep(signal.duration)  # Simulate time spent in green
                else:
                    print(f"{signal.lane.direction.name} lane is Red.")


# Example usage
traffic_signal = TrafficSignal()

# Scenario Setup
traffic_signal.update_lane_status(LaneDirection.NORTH, TrafficStatus.DENSE)
traffic_signal.update_lane_status(LaneDirection.SOUTH, TrafficStatus.AMBULANCE)
traffic_signal.update_lane_status(LaneDirection.EAST, TrafficStatus.FIRE_BURST)
traffic_signal.update_lane_status(LaneDirection.WEST, TrafficStatus.SPARSE)

# Simulate the signal phases
traffic_signal.simulate_signals()
