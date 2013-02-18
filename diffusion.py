# diffusion.py

"""Solves the diffusion equation on a 1D linear or radial grid.
Allows multiple media with different diffusion constants and 
different spacing on grid. Test with diffusion_unittest.py.
See class DiffusionProblem1D for details on solving method.

"""

import numpy
from scipy import sparse
from scipy.sparse.linalg.dsolve import linsolve

__author__ = "John Koschwanez"
__copyright__ = "Copyright 2010"
__credits__ = ["John Koschwanez"]
__license__ = "GPL"
__version__ = "3.0"
__maintainer__ = "John Koschwanez"
__email__ = "john.koschwanez@gmail.com"
__status__ = "Production"

class Boundary:
	"""Data structure of conditions for a boundary"""

	def __init__(self, init_value, t1_value):
		self.old_value = init_value
		self.new_value = t1_value
	
	def set_new_value(self, input_value = None):
		self.old_value = self.new_value
		if input_value:
			self.new_value = input_value
		
class ConcentrationBoundary(Boundary):
	"""Data structure of boundary conditions for given concentration"""
	
	def __init__(self, t0_concentration, t1_concentration):
		Boundary.__init__(self, t0_concentration, t1_concentration)
		
class FluxBoundary(Boundary):
	"""Data structure of boundary conditions for a given flux"""
	
	def __init__(self, init_flux, t1_flux):
		Boundary.__init__(self, init_flux, t1_flux)
		
class EvapFluxBoundary(Boundary):
	"""Data structure of boundary conditions for evaporative flux"""
	
	def __init__(self, alpha):
		Boundary.__init__(self, None, None)
		self.alpha = alpha

class Substrate:
	"""Description of the medium through which diffusion occurs
	
	num_x_points includes the boundaries"""
	
	def __init__(self, x_l, x_r, num_x_points):
		self.x_l = x_l
		self.x_r = x_r
		self.num_x_points = float(num_x_points)
		self.delta_x = (x_r-x_l)/(self.num_x_points-1)
		self.xloc_list = numpy.arange(x_l, x_r+0.000001, self.delta_x, \
							dtype = numpy.float32)

class Water(Substrate):
	"""Description of diffusion coefficients through water"""
	
	def __init__(self, x_l, x_r, num_x_points):
		Substrate.__init__(self, x_l, x_r, num_x_points)
		self.diff_coeff = {
			'glucose': 670,
			'sucrose': 520,
			'invertase_octamer': 28,
			'invertase_dimer': 40,
			'no_diffusion': 1E-12,
			'phosphate': 824,
			'organic_phosphate': 600, 
			'phosphatase_octamer': 28,
			'phosphatase_dimer': 39
		}
		
class MovingWater(Substrate):
	"""Description of diffusion coefficients through moving water"""
	
	def __init__(self, x_l, x_r, num_x_points):
		Substrate.__init__(self, x_l, x_r, num_x_points)
		self.diff_coeff = {
			'glucose': 1E5,
			'sucrose': 1E5,
			'invertase_octamer': 1E5,
			'invertase_dimer': 1E5,
			'no_diffusion': 1E-12,
			'phosphate': 1E5,
			'organic_phosphate': 1E5,
			'phosphatase_octamer': 1E5,
			'phosphatase_dimer': 1E5
		}		
		
class Agar(Substrate):
	"""Description of diffusion coefficients through agar"""
	
	def __init__(self, x_l, x_r, num_x_points):
		Substrate.__init__(self, x_l, x_r, num_x_points)
		self.diff_coeff = {
			'glucose': 670,
			'sucrose': 520,
			'invertase_octamer': 28,
			'invertase_dimer': 40,
			'no_diffusion': 1E-12,
			'phosphate': 824,
			'organic_phosphate': 600,
			'phosphatase_octamer': 28,
			'phosphatase_dimer': 39
		}		
		
class SubstrateStructure(list):
	"""List of all substrates"""
	
	def __init__(self, *substrate_list):
		self.extend(substrate_list)
		self.boundary_index = [0]
		self.boundary_xloc = [substrate_list[0].x_l]
		self.total_xloc_list = numpy.empty(0, dtype = numpy.float32)
		for i, substrate in enumerate(substrate_list):
			self.boundary_index.append(
				self.boundary_index[i]+substrate.num_x_points-1)
			self.boundary_xloc.append(substrate.x_r)
			self.total_xloc_list = numpy.append(self.total_xloc_list,
									substrate.xloc_list[:-1])
			if substrate.x_l != self.boundary_xloc[i]:
				print "boundaries do not match"
				print substrate
				#TODO: make this an exception
		self.total_xloc_list = numpy.append(self.total_xloc_list,
										substrate_list[-1].x_r)
		self.total_num_x_points = self.total_xloc_list.shape[0]
		self.overall_xloc = {'left': self.total_xloc_list[0],
								'right': self.total_xloc_list[-1]}
#		print "total list", self.total_xloc_list

class DiffusionProblem1D:
	"""Propose and solve a 1D linear or radial diffusion problem.
	
	Uses Crank-Nicolson (C-N) implicit method (Crank, 1975, Chapter 8).
	A set of n simultaneous equations (for N-2 internal grid points and 2
		"fictitious" (p.147) external grid points at each boundary)
		are solved for each time point. The t=0 values (concentrations)
		at each of the N-2 internal grid points must be given, 
		and boundary conditions must be given at all time points.
	r = diff_const * delta_t / delta_x**2 is a dimensionless constant used    
		in the calculations.
	The following matrices are set up and solved for c_new:
		A * c_new = B * c_old + C + R
	where 
		A -- sparse diagonal n x n matrix
			2 + 2*r on diagonal
			-r off-1 diagonal on both sides
			A_ul and A_lr corners correspond to external grid points and
				depend on boundary conditions
		c_new -- n x 1 vector of new concentration values
		B -- sparse diagonal N x N matrix
			2 - 2*r on diagonal
			r off-1 diagonal on both sides
			B_ul and B_lr corners correspond to external grid points and
				depend on boundary conditions
		c_old -- n x 1 vector of old concentration values
		C -- n x 1 vector accounting for boundary conditions
			(usually only first and last values filled in)
		R -- n x 1 vector of reaction values
	
	"""
	
	def __init__(self, molecule, linear_or_radial,
					left_boundary, right_boundary,
					init_conc, substrates, delta_t,
					enzyme_limited_breakdown_Vmax = None,
					breakdown_k = None):
		self.molecule = molecule
		self.linear_or_radial = linear_or_radial
		self.boundaries = {'left': left_boundary, 'right': right_boundary}
		self.conc = init_conc
		self.substrates = substrates
		self.delta_t = delta_t
		self.enzyme_limited_breakdown_Vmax = enzyme_limited_breakdown_Vmax
		self.breakdown_k = breakdown_k
#		print "boundary_index", self.substrates.boundary_index[-1]
#		print "shape", numpy.shape(self.conc)[0]-1
		if self.substrates.boundary_index[-1] !=(numpy.shape(self.conc)[0]-1):
			print "points in substrate and init concentrations do not match"
		self.n = numpy.shape(self.conc)[0]
		
		# Making a list of diffusion coefficients and r
		# will make calculations more succinct later
		# r is a dimensionless coefficient that describes diffusion in
		# the substrate for a delta time and delta x

		self.set_constants()
		self.construct_A_B_matrices()
		self.C = numpy.zeros(self.n, dtype = numpy.float32)
		self.R = numpy.zeros(self.n, dtype = numpy.float32)
		
	def set_constants(self):
		self.diff_coeff = []
		self.r = []
		for i, substrate in enumerate(self.substrates):
			self.diff_coeff.append(substrate.diff_coeff[self.molecule])
			self.r.append((self.diff_coeff[i] * self.delta_t)/
					substrate.delta_x**2)
		self.r_boundary = {'left': self.r[0], 'right': self.r[-1]}
					
		self.nu = {'left': self.delta_t/self.substrates[0].delta_x,
					'right': self.delta_t/self.substrates[-1].delta_x}
					
		if self.linear_or_radial == 'linear':
			self.phi = {'left': 0, 'right': 0}
		elif self.linear_or_radial == 'radial':
			self.phi = {'left': self.substrates[0].delta_x \
									/self.substrates.overall_xloc['left'],
						'right': -self.substrates[-1].delta_x \
									/self.substrates.overall_xloc['right']}
		
	def construct_A_B_matrices(self):
		"""Construct sparse diagonal matricies. Boundaries filled later"""
		A_diag = numpy.zeros(self.n)
		A_off_diag = numpy.zeros(self.n)
		B_diag = numpy.zeros(self.n)
		B_off_diag = numpy.zeros(self.n)
		num_substrates = len(self.substrates)
		
		#Fill in diagonals except for boundaries between substrates.
		for i in range(num_substrates):
			left_index = self.substrates.boundary_index[i]
			right_index = self.substrates.boundary_index[i+1]
			A_diag[left_index+1 : right_index] = 2+2*self.r[i]
			A_off_diag[left_index+1 : right_index] = -self.r[i]
			B_diag[left_index+1 : right_index] = 2-2*self.r[i]
			B_off_diag[left_index+1 : right_index] = self.r[i]
		
		#Fill out sparse matrices
		A = sparse.lil_matrix((self.n, self.n))
		A.setdiag(A_diag)
		A.setdiag(A_off_diag[:-1],1)
		A.setdiag(A_off_diag[1:],-1)
		B = sparse.lil_matrix((self.n, self.n))
		B.setdiag(B_diag)
		B.setdiag(B_off_diag[:-1],1)
		B.setdiag(B_off_diag[1:],-1)
		
		#Fill in entries in diagonals in boundaries between substrates
		if num_substrates > 1:
			for i in range(num_substrates-1):
				boundary_index = self.substrates.boundary_index[i+1]
				boundary_xloc = self.substrates.boundary_xloc[i+1]
				r_left = self.r[i]
				r_right = self.r[i+1]
				delta_x_left = self.substrates[i].delta_x
				delta_x_right = self.substrates[i+1].delta_x
				delta_x_ratio = delta_x_right/delta_x_left
				
				if self.linear_or_radial == 'linear':
					gamma = 0
				elif self.linear_or_radial == "radial":
					gamma = (1/boundary_xloc)*(-r_left*delta_x_left + \
								r_right * delta_x_ratio * delta_x_right)
				
				A[boundary_index, boundary_index-1] = -r_left
				A[boundary_index, boundary_index] = \
					1 + r_left + (1 + r_right) * delta_x_ratio + gamma
				A[boundary_index, boundary_index+1] = -r_right * delta_x_ratio
				B[boundary_index, boundary_index-1] = r_left
				B[boundary_index, boundary_index] = \
					1 - r_left + (1 - r_right) * delta_x_ratio - gamma
				B[boundary_index, boundary_index+1] = r_right * delta_x_ratio
		
		self.A = A
		self.B = B
		
	def set_boundary_values(self):
		A_bounds = {}
		B_bounds = {}
		C_bounds = {}
		
		for side in ('left', 'right'):
			if self.boundaries[side].__class__ == ConcentrationBoundary:
				A_bounds[side] = (2,0)
				B_bounds[side] = (0,0)
				if self.linear_or_radial == "linear":
					C_bounds[side] = 2* self.boundaries[side].new_value
				elif self.linear_or_radial == "radial":
					C_bounds[side] = 2*self.boundaries[side].new_value \
						* self.substrates.overall_xloc[side]
			elif self.boundaries[side].__class__ == FluxBoundary:
				A_bounds[side]=(2+2*self.r_boundary[side]*(1+self.phi[side]),\
									-2*self.r_boundary[side])
				B_bounds[side] = (2-2*self.r_boundary[side]*(1+self.phi[side]),
									2*self.r_boundary[side])
				C_bounds[side] = 2*self.nu[side] \
					*(self.boundaries[side].new_value \
					+ self.boundaries[side].old_value)
				if self.linear_or_radial == 'radial':
					C_bounds[side] = C_bounds[side] \
										* self.substrates.overall_xloc[side]
					#TODO: test on right side.
		
		self.A[0,0], self.A[0,1] = A_bounds['left']
		self.A[-1,-1], self.A[-1,-2] = A_bounds['right']
		
		self.B[0,0], self.B[0,1] = B_bounds['left']
		self.B[-1,-1], self.B[-1,-2] = B_bounds['right']
		
		self.C[0] = C_bounds['left']
		self.C[-1] = C_bounds['right']
		
	def set_reaction_vector_breakdown(self):
		self.R = self.breakdown_k * self.delta_t * self.conc
			
	def set_reaction_vector_enzyme_limited_breakdown(
				self, enzyme_concentration):
		self.R = self.enzyme_limited_breakdown_Vmax * \
					self.delta_t * enzyme_concentration

	def set_reaction_vector_external_add_subtract(
				self, external_add_subtract):
		self.R = external_add_subtract		

	def diffuse_one_time_step(self, enzyme_concentration = None,
								external_add_subtract = None):
		"""Solve a single time step in the diffusion problem"""

		self.set_boundary_values()
		
		if self.breakdown_k:
			self.set_reaction_vector_breakdown()
		if self.enzyme_limited_breakdown_Vmax:
			self.set_reaction_vector_enzyme_limited_breakdown(
				enzyme_concentration)
		if external_add_subtract.__class__ == numpy.ndarray:
			self.set_reaction_vector_external_add_subtract(
											external_add_subtract)
		
		A = self.A.tocsr()
		B = self.B.tocsr()
		if self.linear_or_radial == "linear":
			c_old = self.conc
		elif self.linear_or_radial == "radial":
			c_old = self.convert_radial_to_linear(self.conc)
			self.R = self.convert_radial_to_linear(self.R)
		c_new = linsolve.spsolve(A, ((B * c_old) + self.C + 2*self.R))
		if self.linear_or_radial == "linear":
			self.conc = c_new
		elif self.linear_or_radial == "radial":
			self.conc = self.convert_linear_to_radial(c_new)
		self.conc[self.conc<0] = 0	#needed if enzyme brings conc<0
		self.R = numpy.zeros(self.n, dtype = numpy.float32) #Reset reaction matrix
			

	def convert_radial_to_linear(self, c_radial):
		return (c_radial * self.substrates.total_xloc_list)

	def convert_linear_to_radial(self, c_linear):
		return (c_linear * (self.substrates.total_xloc_list**(-1)))
		
	def diffuse_multiple_time_steps(self, num_time_steps,
										enzyme_concentration = None,
										external_add_subtract = None):
		for i in range(num_time_steps):
			for side in ('left', 'right'):
				self.boundaries[side].set_new_value()
			self.diffuse_one_time_step(
				enzyme_concentration, external_add_subtract)
