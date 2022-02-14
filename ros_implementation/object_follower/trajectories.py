import numpy
from scipy.interpolate import BSpline, splrep

class TrajectoryShape(object):
	def generate(self, t_count = 1000):
		t = numpy.linspace(0, 1, t_count)
		x = numpy.zeros(t.shape)# + 2 * t - 1
		y = numpy.zeros(t.shape)
		vals = numpy.vstack([x, y])
		return vals.T

class SinusoidalY(TrajectoryShape):
	def __init__(self, degree):
		self.degree = int(degree)
		
	def generate(self, t_count = 1000):
		k = self.degree
		t = numpy.linspace(0, 1, t_count)
		x = 2 * t - 1
		y = 1 / k * numpy.sin(2 * numpy.pi * t * k)
		return numpy.vstack([x,y]).T

class SinusoidalYOnly(TrajectoryShape):
	def __init__(self, magnitude, degree):
		self.magnitude = float(magnitude)
		self.degree = int(degree)
		
	def generate(self, t_count = 1000):
		k = self.degree
		t = numpy.linspace(0, 1, t_count)
		x = 0 * t
		y = self.magnitude * numpy.sin(2 * numpy.pi * t * k)
		return numpy.vstack([x,y]).T

class SinusoidalX(TrajectoryShape):
	def __init__(self, degree):
		self.degree = int(degree)
		
	def generate(self, t_count = 1000):
		k = self.degree
		t = numpy.linspace(0, 1, t_count)
		x = 2 * t - 1
		y = 1 / k * numpy.sin(2 * numpy.pi * t * k)
		return numpy.vstack([y,x]).T
	
def get_vals_from_constants(consts, t, func = numpy.sin):
	term_count = len(consts)
	t = numpy.tile(t, (term_count, 1))
	freq_mult = numpy.arange(1, term_count + 1).astype(numpy.float32).reshape(-1, 1) * 2 * numpy.pi
	vals = func(t * freq_mult) * numpy.array(consts).reshape(-1, 1)
	vals = numpy.sum(vals, axis=0)
	return vals

class PeriodicTrajectory(TrajectoryShape):
	sin_x_consts = None
	cos_x_consts = None
	sin_y_consts = None
	cos_y_consts = None

	def generate(self, t_count = 1000):
		sin_x_consts = self.sin_x_consts
		cos_x_consts = self.cos_x_consts
		sin_y_consts = self.sin_y_consts
		cos_y_consts = self.cos_y_consts

		t = numpy.linspace(0, 1, t_count)
		x = numpy.zeros(t.shape)# + 2 * t - 1
		y = numpy.zeros(t.shape)

		if sin_x_consts is not None:
			x += get_vals_from_constants(sin_x_consts, t, numpy.sin)

		if cos_x_consts is not None:
			x += get_vals_from_constants(cos_x_consts, t, numpy.cos)


		if sin_y_consts is not None:
			y += get_vals_from_constants(sin_y_consts, t, numpy.sin)

		if cos_y_consts is not None:
			y += get_vals_from_constants(cos_y_consts, t, numpy.cos)

		vals = numpy.vstack([x, y])
		return vals.T

class Circle(PeriodicTrajectory):
	sin_x_consts = [1]
	cos_y_consts = [1]

class DoubleInfinity(PeriodicTrajectory):
	sin_x_consts = [1]
	sin_y_consts = [0, 0, 0, 1]

class DiagonalDoubleInifinity(PeriodicTrajectory):
	sin_x_consts = [0, -1, -1]
	sin_y_consts = [0, 1, -1]

class Flower(PeriodicTrajectory):
	def __init__(self, n_lobes=4):
		n_lobes = int(n_lobes)
		if n_lobes % 2 == 0:
			n = n_lobes/2 - 2
			n = int(n)
			self.cos_x_consts = [0] * n + [1, 0, -1]
			self.sin_y_consts = [0] * n + [1, 0, 1]
		else:
			n = (n_lobes - 1) / 2 - 1
			n = int(n)
			self.sin_x_consts = [0] * n + [-1, 1]
			self.cos_y_consts = [0] * n + [1, 1]

class InscribedCircle(PeriodicTrajectory):
	cos_x_consts = [1, 0, -1]
	sin_y_consts = [-1./3, 0, 1]

class Infinity(PeriodicTrajectory):
	def __init__(self, degree=1):
		self.sin_x_consts = [1]
		self.sin_y_consts = [0] * (2 * (degree - 1) + 1) + [1]

class Rectangle(TrajectoryShape):
	def __init__(self, x_length, y_length):
		self.x_length = x_length
		self.y_length = y_length
	
	def generate(self, t_count=1000):
		t_count = 100
		points = numpy.zeros((6, 2), dtype=numpy.float32)
		points[0, :] = numpy.array([0, 140])
		points[1, :] = numpy.array([self.x_length / 2, 140])		
		points[2, :] = numpy.array([self.x_length / 2, 140 -self.y_length])		
		points[3, :] = numpy.array([-self.x_length / 2, 140 -self.y_length])
		points[4, :] = numpy.array([-self.x_length / 2, 140])			
		points[5, :] = numpy.array([0, 140])

		# print(points)
		t_points = numpy.array([0, 1, 3, 5, 7, 8])

		t = numpy.linspace(0, t_points[-1], t_count)
		x = numpy.interp(t, t_points, points[:, 0])
		y = numpy.interp(t, t_points, points[:, 1])
		return numpy.vstack([x,y]).T


class SplineY(TrajectoryShape):
	def __init__(self, n_key_points=1, std_dev=0.1, force_direction_change=1, seed=None):
		if seed is not None:
			self.seed = int(seed)
		else:
			self.seed = seed
		self.n_key_points = int(n_key_points)
		self.std_dev = std_dev
		self.force_direction_change = force_direction_change > 0
	
	def generate(self, t_count=1000):
		if self.seed is not None:
			numpy.random.seed(self.seed)
		t = numpy.linspace(0, 1, t_count)
		key_points = numpy.zeros((self.n_key_points + 2, 2), dtype=numpy.float32)
		key_points[:, 0] = numpy.linspace(-1, 1, self.n_key_points + 2)
		
		while True:
			for i in range(1, key_points.shape[0]):
				# while True:
				change = numpy.random.normal(0, self.std_dev)
				if self.force_direction_change and i >= 2:
					if numpy.sign(key_points[i-1, 1] - key_points[i-2, 1]) == numpy.sign(change):
						change *= -1

				new_val = key_points[i-1, 1] + change
					
					# if new_val >= -1 and new_val <= 1:
					# 	break
				key_points[i, 1] = new_val

			spline_consts = splrep(key_points[:, 0], key_points[:, 1], s=0, k=3)
			spl_y = BSpline(*spline_consts)
			x = 2 * t - 1
			y = spl_y(x)

			if not numpy.any(numpy.abs(y) > 1):
				break
		
		return numpy.vstack([x,y]).T

class SplineX(TrajectoryShape):
	def __init__(self, *args):
		self.spline_y = SplineY(*args)

	def generate(self, t_count=1000):
		y_spline = self.spline_y.generate(t_count)
		return y_spline[:, ::-1]

class RandomWalk(TrajectoryShape):
	def __init__(self, n_direction_changes=10, dist_between_changes=50./140., seed=None):
		self.n_direction_changes = int(n_direction_changes)
		self.dist_between_changes = dist_between_changes
		if seed is not None:
			self.seed = int(seed)
		else:
			self.seed = seed
	
	def generate(self, t_count=1000):
		points = numpy.zeros((self.n_direction_changes, 2), dtype=numpy.float32)
		dist = self.dist_between_changes
		prev_angle = None

		if self.seed is not None:
			numpy.random.seed(self.seed)

		for i in range(1, self.n_direction_changes):
			while 1:
				angle = (numpy.random.rand() * 180) + 45
				if angle > 135:
					angle += 90
					
				if prev_angle is not None:
					angle += prev_angle
				prev_angle = angle
				# print(angle)
				angle  = numpy.deg2rad(angle)
				points[i, :] = points[i-1, :] + numpy.array([numpy.cos(angle), numpy.sin(angle)], dtype=numpy.float32) * dist
				if numpy.all(numpy.abs(points[i, :]) < 1):
					break 
		
		t_points = numpy.arange(0, self.n_direction_changes)
		# return points
		# print(t_points)
		t = numpy.linspace(0, t_points[-1], t_count)
		x = numpy.interp(t, t_points, points[:, 0])
		y = numpy.interp(t, t_points, points[:, 1])
		return numpy.vstack([x,y]).T

def calculate_radius_points(angle, prev_angle, prev_point, radius ):
	if angle > 0:
		# turning left
		center_angle = prev_angle + 90
	else:
		# turning right
		center_angle = prev_angle - 90
	center_angle_rad = numpy.deg2rad(center_angle)
	center = prev_point + numpy.array([numpy.cos(center_angle_rad), numpy.sin(center_angle_rad)], dtype=numpy.float32) * radius

	# print(angle) 
	inv_center_angle = center_angle + 180
	theta = numpy.deg2rad(numpy.linspace(inv_center_angle, inv_center_angle + angle, 50))

	new_corner_points = numpy.vstack([numpy.cos(theta), numpy.sin(theta)]).astype(numpy.float32).T * radius + center
	return new_corner_points
	
def interpolate_trajectory(traj, max_distance = 0.01):
	new_traj = traj[0, :].reshape(-1, 2)
	for i in range(1, traj.shape[0]):
		dist_to_previous = numpy.linalg.norm(new_traj[-1, :] - traj[i, :])
		# print(dist_to_previous)
		if dist_to_previous > max_distance:
			# print(dist_to_previous)
			# insert interpolated points
			n_required = int(numpy.ceil(dist_to_previous / max_distance))
			vals = numpy.tile(numpy.linspace(0, 1, n_required + 1)[1:].reshape(-1,1), 2)
			# print(n_required, vals)
			# print(traj[i, :]* vals)
			new_traj = numpy.vstack([new_traj, (new_traj[-1, :] * (1-vals) + traj[i, :] * vals)])
			
		else:
			# insert current point
			new_traj = numpy.vstack([new_traj, traj[i, :]])

	return new_traj

class RadiusedRandomWalk(TrajectoryShape):
	def __init__(self, n_direction_changes=10, dist_between_changes=50./140., radius=10./140., seed=None):
		self.n_direction_changes = int(n_direction_changes)
		self.dist_between_changes = dist_between_changes
		self.radius = radius

		if seed is not None:
			self.seed = int(seed)
		else:
			self.seed = seed
	
	def generate(self, t_count=1000):
		points = numpy.zeros(2, dtype=numpy.float32).reshape(1, 2)
		dist = self.dist_between_changes
		prev_angle = None

		if self.seed is not None:
			numpy.random.seed(self.seed)

		for i in range(1, self.n_direction_changes):
			# print(i)
			while 1:
				# angle = (numpy.random.rand() * 180) + 45
				# if angle > 135:
				# 	angle += 90
				angle = (numpy.random.rand() * 270) + 45

				if angle > 180:
					angle -= 360
				# print(angle)
				# print(angle, prev_angle)
				new_corner_points = None
				corner_points_out_of_bounds = False
				if prev_angle is not None:
					new_corner_points = calculate_radius_points(angle, prev_angle, points[-1, :], self.radius)
					corner_points_out_of_bounds = numpy.any(numpy.abs(new_corner_points) > 1)

					angle += prev_angle

				
				angle = numpy.deg2rad(angle)

				new_point_offset = numpy.array([numpy.cos(angle), numpy.sin(angle)], dtype=numpy.float32) * dist
				
				if new_corner_points is None:
					new_point = points[-1, :] + new_point_offset
				else:
					new_point = new_corner_points[-1, :] + new_point_offset


				next_corner_points_in_bounds = True
				if new_corner_points is not None:
					next_corner_points = calculate_radius_points(360, numpy.rad2deg(angle), new_point, self.radius)
					next_corner_points_in_bounds_1 = numpy.all(numpy.abs(next_corner_points) < 1)

					next_corner_points = calculate_radius_points(-360, numpy.rad2deg(angle), new_point, self.radius)
					# print(new_point)
					# print(next_corner_points)
					next_corner_points_in_bounds_2 = numpy.all(numpy.abs(next_corner_points) < 1)

					# print(next_corner_points_in_bounds_1, next_corner_points_in_bounds_2)
					next_corner_points_in_bounds = next_corner_points_in_bounds_1 or next_corner_points_in_bounds_2
					

				if numpy.all(numpy.abs(new_point) < 1) and not corner_points_out_of_bounds and next_corner_points_in_bounds:
					
					prev_angle = numpy.rad2deg(angle)
					if new_corner_points is not None:
						points = numpy.vstack([points, new_corner_points])
						
					points = numpy.vstack([points, new_point])
					break 
		
		# t_points = numpy.arange(0, self.n_direction_changes)
		# print(points)
		return numpy.vstack(points)

def scale_trajectory(traj, x_limit, y_limit):
	x_max = numpy.max(numpy.abs(traj[:, 0]))
	y_max = numpy.max(numpy.abs(traj[:, 1]))
	if x_max != 0:
		x_factor = x_limit / x_max
	else:
		x_factor = numpy.inf
	
	if y_max != 0:
		y_factor = y_limit / y_max
	else:
		y_factor = numpy.inf
	factor = min(x_factor, y_factor)
	# print(factor)
	if factor == numpy.inf:
		print("Trajectory scale factor is infinity! Resetting to 1")
		factor = 1

	return traj * factor

if __name__=="__main__":
	from matplotlib import pyplot as plt
	import sys 

	TRAJECTORIES = [
		("Circle", []), 
		("Flower", []),
		("DoubleInfinity", []),
		("DiagonalDoubleInifinity", []),
		("InscribedCircle", []),
		("RadiusedRandomWalk", [100, 50.0/140, 10.0/140, 0]),
		("RadiusedRandomWalk", [100, 50.0/140, 10.0/140, 1]),
		("Rectangle", [200.0, 200.0]),
		("Rectangle", [50.0, 200.0]),
		("Rectangle", [200.0, 50.0])
	]


	plt.style.use(['science', "grid"])
	
	# plt.rcParams.update({'axes.titlesize': 'x-large'})
	# plt.rcParams.update({'axes.labelsize': 13})
	plt.rcParams.update({'savefig.dpi': 1800})

	
	# ax.grid()
	styles = ["k-", "b:", "r--"]
	workspace_bounds = (140, 140)
	for i, t in enumerate(TRAJECTORIES):
		class_ = getattr(sys.modules[__name__], t[0])
		args = t[1]
		if len(args) == 0:
			shape = class_()
		else:
			shape = class_(*args)

		traj = shape.generate()

		max_val = numpy.max(traj)
		min_val = numpy.min(traj)
		if max_val <= 1 and min_val >= -1:
			# print("Applying constant scaling of x{}".format(workspace_bounds[0]))
			traj = traj * workspace_bounds[0]
		else:
			# print("Applying custom scaling")
			traj = scale_trajectory(traj, workspace_bounds[0], workspace_bounds[1])  # Max 140, 140


        # Plot the trajectory
		plt.figure(figsize=(4,4))
		ax = plt.axes()

		ax.set_aspect(1)
		plt.plot(traj[:, 0], traj[:, 1])

		plt.xlim([-150, 150])
		plt.ylim([-150, 150])

		plt.xlabel("$x$ (mm)")
		plt.ylabel("$y$ (mm)")
		plt.title(f"Workspace Trajectory {i+1}")
		plt.tight_layout()

		# plt.savefig("../trajectory_images/test.png")
		plt.show()
