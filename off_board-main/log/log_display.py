# log/log_display.py
from pymavlink import mavutil          # reads .bin via DFReader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import io

def load_gps_path(bin_path):
    """Returns list of (lat, lon, timestamp) from GPS messages."""
    mlog = mavutil.mavlink_connection(bin_path)
    points = []
    while True:
        msg = mlog.recv_match(type='GPS', blocking=False)
        if msg is None: break
        points.append((msg.Lat, msg.Lng, msg.TimeUS))
    return points

def load_waypoints(bin_path):
    """Returns list of (lat, lon) from CMD messages (uploaded mission)."""
    mlog = mavutil.mavlink_connection(bin_path)
    waypoints = []
    while True:
        msg = mlog.recv_match(type='CMD', blocking=False)
        if msg is None: break
        if msg.Lat != 0:  # skip home/meta commands
            waypoints.append((msg.Lat, msg.Lng))
    return waypoints

def create_path_gif(bin_path, output_path, step=10):
    path = load_gps_path(bin_path)
    waypoints = load_waypoints(bin_path)
    lats = [p[0] for p in path]
    lons = [p[1] for p in path]

    fig, ax = plt.subplots(figsize=(8, 8))
    # plot waypoints as fixed markers
    if waypoints:
        ax.scatter([w[1] for w in waypoints], [w[0] for w in waypoints],
                   c='red', s=80, zorder=5, label='Waypoints')

    line, = ax.plot([], [], 'b-', linewidth=1.5)
    dot,  = ax.plot([], [], 'bo', markersize=6)

    ax.set_xlim(min(lons)-0.0002, max(lons)+0.0002)
    ax.set_ylim(min(lats)-0.0002, max(lats)+0.0002)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend()

    frames = range(0, len(path), step)

    def update(i):
        line.set_data(lons[:i], lats[:i])
        dot.set_data([lons[i-1]], [lats[i-1]])
        return line, dot

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    ani.save(output_path, writer='pillow', fps=15)
    plt.close()

if __name__ == "__main__":
    create_path_gif("flights/flight_1/pixhawk_log.bin", "flights/flight_1/result.gif")