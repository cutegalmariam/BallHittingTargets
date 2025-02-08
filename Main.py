import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
from PIL import Image
from TestCases import *


# Force matplotlib to use a backend that opens a new window
matplotlib.use('TkAgg')


class CustomCircleDetector:
    def __init__(self, min_radius=20, max_radius=100, threshold=0.4):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.threshold = threshold

    def sobel_operator(self, img):
        """Apply Sobel operator for edge detection"""
        kernel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

        kernel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

        height, width = img.shape
        gradient_x = np.zeros_like(img, dtype=float)
        gradient_y = np.zeros_like(img, dtype=float)

        for y in range(1, height-1):
            for x in range(1, width-1):
                neighborhood = img[y-1:y+2, x-1:x+2]
                gradient_x[y, x] = np.sum(neighborhood * kernel_x)
                gradient_y[y, x] = np.sum(neighborhood * kernel_y)

        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)

        return gradient_magnitude, gradient_direction

    def non_max_suppression(self, img, gradient_direction):
        """Apply non-maximum suppression to thin edges"""
        height, width = img.shape
        output = np.zeros_like(img)

        angle = gradient_direction * 180 / np.pi
        angle[angle < 0] += 180

        for y in range(1, height-1):
            for x in range(1, width-1):
                if (0 <= angle[y,x] < 22.5) or (157.5 <= angle[y,x] <= 180):
                    neighbors = [img[y,x-1], img[y,x+1]]
                elif (22.5 <= angle[y,x] < 67.5):
                    neighbors = [img[y-1,x+1], img[y+1,x-1]]
                elif (67.5 <= angle[y,x] < 112.5):
                    neighbors = [img[y-1,x], img[y+1,x]]
                else:
                    neighbors = [img[y-1,x-1], img[y+1,x+1]]

                if img[y,x] >= max(neighbors):
                    output[y,x] = img[y,x]

        return output

    def hough_circle_transform(self, edge_image, gradient_direction):
        """Custom implementation of Hough Transform for circles"""
        height, width = edge_image.shape
        accumulator = {}

        for y in range(height):
            for x in range(width):
                if edge_image[y,x] > 0:
                    theta = gradient_direction[y,x]

                    for r in range(self.min_radius, self.max_radius):
                        # Forward direction
                        a = int(x - r * np.cos(theta))
                        b = int(y - r * np.sin(theta))

                        if 0 <= a < width and 0 <= b < height:
                            key = (a, b, r)
                            accumulator[key] = accumulator.get(key, 0) + 1

                        # Backward direction
                        a = int(x + r * np.cos(theta))
                        b = int(y + r * np.sin(theta))

                        if 0 <= a < width and 0 <= b < height:
                            key = (a, b, r)
                            accumulator[key] = accumulator.get(key, 0) + 1

        return accumulator

    def find_circle_centers(self, accumulator, edge_image):
        """Find circle centers from accumulator array"""
        height, width = edge_image.shape
        max_votes = max(accumulator.values()) if accumulator else 0
        threshold_votes = max_votes * self.threshold

        circles = []
        sorted_points = sorted(accumulator.items(), key=lambda x: x[1], reverse=True)

        for (x, y, r), votes in sorted_points:
            if votes < threshold_votes:
                break

            is_maximum = True
            for detected_x, detected_y, detected_r in circles:
                distance = np.sqrt((x - detected_x)**2 + (y - detected_y)**2)
                if distance < self.min_radius:
                    is_maximum = False
                    break

            if is_maximum:
                circles.append((x, y, r))

        return circles

    def detect_circles(self, image_path):
        """Main method to detect circles in image"""
        # Load and preprocess image
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)

        # Store dimensions for scaling
        self.image_height, self.image_width = img_array.shape

        # Apply Gaussian blur
        img_array = np.array([[np.mean(img_array[i-1:i+2, j-1:j+2])
                             for j in range(1, img_array.shape[1]-1)]
                             for i in range(1, img_array.shape[0]-1)])

        # Detect edges
        gradient_magnitude, gradient_direction = self.sobel_operator(img_array)
        edge_image = self.non_max_suppression(gradient_magnitude, gradient_direction)

        # Threshold
        edge_threshold = np.max(edge_image) * 0.2
        edge_image = (edge_image > edge_threshold).astype(np.uint8) * 255

        # Detect circles
        accumulator = self.hough_circle_transform(edge_image, gradient_direction)
        circles = self.find_circle_centers(accumulator, edge_image)

        # Scale circles to simulation space (120x100)
        scaled_circles = []
        for x, y, r in circles:
            scaled_x = (x / self.image_width) * 120
            scaled_y = ((self.image_height - y) / self.image_height) * 100
            scaled_r = (r / self.image_width) * 120
            scaled_circles.append((scaled_x, scaled_y, scaled_r))

        return scaled_circles



def main():
    print("...starting...")
    print("1: m=0.5kg, b=0.07kg/s")
    print("2: m=0.5kg, b=0.5kg/s")
    print("3: m=3kg, b=0.07kg/s")
    print("4: m=3kg, b=0.5kg/s")
    print("5: m=1kg, b=0.2kg/s")
    print("6: m=0.5kg, b=0kg/s  NO DRAG(Vacuum)")

    while True:
        try:
            test_case_num = int(input("which testcase would you like to run? (1-6): "))
            if 1 <= test_case_num <= 6:
                break
            else:
                print("Please enter a number between 1 and 6")
        except ValueError:
            print("Please enter a valid number")

    m, b = get_mass_drag(test_case_num)
    print("...calculating... :)")
    g = 9.81
    radius = 1  # radius of the ball (m)
    initial_height = 20  # initial height of the shooter ball (m)
    time_step = 0.01  # time step for RK4 and animation (s)
    total_time = 10  # total animation time (s)

    image_path = "TestPhoto.png"
    detector = CustomCircleDetector(min_radius=20, max_radius=100, threshold=0.4)
    targets = detector.detect_circles(image_path)

    # Calculate the number of frames
    num_frames = int(total_time / time_step)

    # Define the system of equations for RK4
    def derivatives(t, state):
        x, vx, y, vy = state
        dxdt = vx
        dvxdt = -(b / m) * vx
        dydt = vy
        dvydt = -g - (b / m) * vy
        return np.array([dxdt, dvxdt, dydt, dvydt])

    # Runge-Kutta 4th order method
    def rk4_step(t, state, dt):
        k1 = derivatives(t, state)
        k2 = derivatives(t + dt / 2, state + dt * k1 / 2)
        k3 = derivatives(t + dt / 2, state + dt * k2 / 2)
        k4 = derivatives(t + dt, state + dt * k3)
        return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def compute_trajectory(initial_vx, initial_vy, initial_height):
        state = np.array([0, initial_vx, initial_height, initial_vy])
        trajectory = []
        t = 0
        for _ in range(num_frames):
            trajectory.append(state.copy())
            state = rk4_step(t, state, time_step)
            t += time_step
            if state[2] - radius <= 0:  # If the ball hits the ground
                state[2] = radius
                break
        return np.array(trajectory)

    def hits_target(trajectory, target_x, target_y, target_radius):
        x_data, _, y_data, _ = trajectory.T
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            distance = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
            if distance <= target_radius + radius:  # Ball radius + target radius
                return True, i  # Return both hit status and frame index of hit
        return False, len(trajectory)

    # Find trajectories for each target
    successful_trajectories = []
    all_failed_trajectories = []
    hit_frames = []  # Store the frame where each successful trajectory hits its target

    for target_x, target_y, target_radius in targets:
        target_successful = False
        target_failed_trajectories = []

        for _ in range(1000):
            initial_vx = np.random.uniform(15, 100)
            initial_vy = np.random.uniform(15, 100)
            trajectory = compute_trajectory(initial_vx, initial_vy, initial_height)

            hit, hit_frame = hits_target(trajectory, target_x, target_y, target_radius)
            if hit:
                successful_trajectories.append(trajectory[:hit_frame + 1])  # Only keep frames until hit
                hit_frames.append(hit_frame)
                target_successful = True
                break

            target_failed_trajectories.append(trajectory)

        # if not target_successful:
        #     print(f"No successful trajectory found for target at ({target_x:.1f}, {target_y:.1f})")
        #     successful_trajectories.append(None)
        #     hit_frames.append(0)

        all_failed_trajectories.append(target_failed_trajectories)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 120)
    ax.set_aspect('equal')
    ax.set_title("Ball Motion with Multiple Targets")
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Height (m)")

    # Draw the ground
    ground = plt.Rectangle((0, 0), 120, radius, color='brown')
    ax.add_patch(ground)

    # Draw all targets
    target_patches = []
    for i, (target_x, target_y, target_radius) in enumerate(targets):
        color = plt.cm.Set3(i / len(targets))
        target = plt.Circle((target_x, target_y), target_radius, color=color,
                            label=f'Target {i + 1}')
        target_patches.append(target)
        ax.add_patch(target)

    # Create the ball
    ball = plt.Circle((0, initial_height), radius, color='blue')
    ax.add_patch(ball)

    # Initialize the trail line
    trail, = ax.plot([], [], 'b-', lw=2)

    # Store trajectory lines
    failed_lines = []
    success_line = None

    current_target = 0
    current_frame = 0
    total_frames_shown = 0

    def update(frame):
        nonlocal current_target, current_frame, total_frames_shown, failed_lines, success_line

        if successful_trajectories[current_target] is None:
            current_target = (current_target + 1) % len(targets)
            current_frame = 0
            total_frames_shown += 1
            return ball, trail

        trajectory = successful_trajectories[current_target]

        # Reset animation if we've reached the end of current trajectory
        if current_frame >= len(trajectory):
            current_target = (current_target + 1) % len(targets)
            current_frame = 0
            total_frames_shown += 1
            # Clear the trail when starting new target
            trail.set_data([], [])
            ball.set_center((0, initial_height))

            # Clear all trajectory lines when moving to next target
            for line in failed_lines:
                line.remove()
            failed_lines.clear()
            if success_line is not None:
                success_line.remove()
                success_line = None

            return ball, trail

        # Update current position
        x, _, y, _ = trajectory[current_frame]
        ball.set_center((x, max(y, radius)))

        # Update trail
        x_data, _, y_data, _ = trajectory.T
        trail.set_data(x_data[:current_frame + 1], y_data[:current_frame + 1])

        current_frame += 1
        return ball, trail

    # Plot failed trajectories for current target
    def plot_failed_trajectories(target_idx):
        # Clear previous failed trajectory lines
        for line in failed_lines:
            line.remove()
        failed_lines.clear()

        print("Found successful trajectory in ", len(all_failed_trajectories[target_idx]), " tries!")

        # Plot new failed trajectories
        for trajectory in all_failed_trajectories[target_idx]:
            x_data, _, y_data, _ = trajectory.T
            line, = ax.plot(x_data, y_data, 'r--', alpha=0.1, label="Failed Trajectory on current target")
            failed_lines.append(line)

    # Plot successful trajectory for current target
    def plot_successful_trajectory(target_idx):
        nonlocal success_line

        if successful_trajectories[target_idx] is not None:
            x_data, _, y_data, _ = successful_trajectories[target_idx].T
            success_line, = ax.plot(x_data, y_data, 'g-', alpha=0.5, label="Successful Trajectory")

    def on_animation_frame(frame):
        nonlocal current_target, current_frame, total_frames_shown, all_failed_trajectories

        # Clear previous trajectories when starting new target
        if current_frame == 0:
            plot_failed_trajectories(current_target)
            plot_successful_trajectory(current_target)

            # Update title to show current target
            ax.set_title(f"Ball Motion - Targeting Target {current_target + 1}. Found successful trajectory in {len(all_failed_trajectories[current_target]) + 1} tries!")

        artists = update(frame)

        # Stop animation after one complete cycle
        if total_frames_shown >= len(targets):
            ani.event_source.stop()

        return artists

    ax.legend()

    # Create the animation
    ani = FuncAnimation(fig, on_animation_frame, frames=num_frames * len(targets),
                        interval=time_step * 1000, blit=False)

    plt.show()


if __name__ == "__main__":
    main()
