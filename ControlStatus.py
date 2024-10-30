class TrafficSignal:
    def __init__(self):
        # Define the traffic light status
        self.lights = {
            'Lane 1 (North)': 'Red',
            'Lane 2 (West)': 'Red',
            'Lane 3 (South)': 'Red',
            'Lane 4 (East)': 'Red'
        }

    def classify_lanes(self, lane_status):
        # Classify lanes based on their status
        lane_classes = {
            'Lane 1 (North)': lane_status[0],
            'Lane 2 (West)': lane_status[1],
            'Lane 3 (South)': lane_status[2],
            'Lane 4 (East)': lane_status[3]
        }
        return lane_classes

    def determine_signal(self, lane_classes):
        # Check for priority conditions
        priority_lanes = [lane for lane, status in lane_classes.items() if status in ['fire', 'accident']]
        dense_lanes = [lane for lane, status in lane_classes.items() if status == 'dense']
        sparse_lanes = [lane for lane, status in lane_classes.items() if status == 'sparse']

        # Case 1: 3 Lanes are classified dense
        if len(dense_lanes) == 3:
            if priority_lanes:
                # Give priority to the lane with fire or accident
                self.give_priority(priority_lanes[0])
            else:
                self.manage_dense_lanes(dense_lanes)

        # Case 2: 2 lanes are dense
        elif len(dense_lanes) == 2:
            if priority_lanes:
                self.give_priority(priority_lanes[0])
            else:
                self.manage_two_dense_lanes(dense_lanes)

        # Case 3: 1 lane is dense
        elif len(dense_lanes) == 1:
            self.give_priority(dense_lanes[0])

        # Case 4: No lane is dense
        else:
            self.manage_all_red()

    def give_priority(self, lane):
        # Set green light for the prioritized lane and red for others
        self.reset_lights()
        self.lights[lane] = 'Green'
        print(f"Priority given to {lane}. Lights updated.")

    def manage_dense_lanes(self, dense_lanes):
        # Set a 30-second timer for dense lanes
        self.reset_lights()

        # For demonstration, we will allow one lane to go straight while others can turn
        if 'Lane 3 (South)' in dense_lanes:
            self.lights['Lane 3 (South)'] = 'Green'
            self.lights['Lane 2 (West)'] = 'Green (Right Turn)'
            self.lights['Lane 4 (East)'] = 'Green (Left/Right Turn)'
            print("Lane 3 (South) goes straight. Lanes 2 and 4 can turn.")
        elif 'Lane 1 (North)' in dense_lanes:
            self.lights['Lane 1 (North)'] = 'Green'
            self.lights['Lane 4 (East)'] = 'Green (Right Turn)'
            self.lights['Lane 2 (West)'] = 'Green (Left/Right Turn)'
            print("Lane 1 (North) goes straight. Lanes 4 and 2 can turn.")
        elif 'Lane 2 (West)' in dense_lanes:
            self.lights['Lane 2 (West)'] = 'Green'
            self.lights['Lane 1 (North)'] = 'Green (Left/Right Turn)'
            self.lights['Lane 3 (South)'] = 'Green (Left/Right Turn)'
            print("Lane 2 (West) goes straight. Lanes 1 and 3 can turn.")
        elif 'Lane 4 (East)' in dense_lanes:
            self.lights['Lane 4 (East)'] = 'Green'
            self.lights['Lane 1 (North)'] = 'Green (Right Turn)'
            self.lights['Lane 3 (South)'] = 'Green (Left Turn)'
            print("Lane 4 (East) goes straight. Lanes 1 and 3 can turn.")

        print("All dense lanes are given green light for 30 seconds.")

    def manage_two_dense_lanes(self, dense_lanes):
        # Handle two dense lanes with priority
        self.reset_lights()
        if 'Lane 1 (North)' in dense_lanes and 'Lane 3 (South)' in dense_lanes:
            self.lights['Lane 1 (North)'] = 'Green'
            self.lights['Lane 3 (South)'] = 'Green'
            print("Lanes 1 and 3 go straight simultaneously.")
        elif 'Lane 2 (West)' in dense_lanes and 'Lane 4 (East)' in dense_lanes:
            self.lights['Lane 2 (West)'] = 'Green'
            self.lights['Lane 4 (East)'] = 'Green'
            print("Lanes 2 and 4 go straight simultaneously.")
        else:
            self.lights[dense_lanes[0]] = 'Green'
            print(f"{dense_lanes[0]} goes straight while allowing turns.")

    def manage_all_red(self):
        # Set all lanes to red if no traffic
        self.reset_lights()
        print("All lights are set to red.")

    def reset_lights(self):
        # Reset all lights to red
        for lane in self.lights:
            self.lights[lane] = 'Red'

    def display_signals(self):
        # Display the current signal status
        for lane, status in self.lights.items():
            print(f"{lane}: {status}")


# Example usage
def main():
    # Input status of lanes: (North, West, South, East)
    # Each lane can be 'dense', 'sparse', 'fire', or 'accident'
    lane_status = ['dense', 'dense', 'fire', 'sparse']  # You can change this input to test different scenarios

    traffic_signal = TrafficSignal()
    lane_classes = traffic_signal.classify_lanes(lane_status)
    traffic_signal.determine_signal(lane_classes)
    traffic_signal.display_signals()

if __name__ == "__main__":
    main()
