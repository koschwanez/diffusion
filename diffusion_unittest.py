# diffusion_uittest.py

"""Unit testing for diffusion.py

"""

import unittest

import numpy

from diffusion import *

__author__ = "John Koschwanez"
__copyright__ = "Copyright 2010"
__credits__ = ["John Koschwanez"]
__license__ = "GPL"
__version__ = "3.0"
__maintainer__ = "John Koschwanez"
__email__ = "john.koschwanez@gmail.com"
__status__ = "Production"

pi = numpy.pi

class diffusion_1D_uniform_linear_test(unittest.TestCase):
	def setUp(self):
		self.HALF_LENGTH = float(100)
		self.INDICES_PER_SIDE = 10
		self.SURF_CONC = numpy.random.randint(1,100)
		print "surface concentration", self.SURF_CONC
		self.INIT_CONC = numpy.random.randint(1,100)
		print "initial concentration", self.INIT_CONC
		self.DELTA_T = 0.1
		self.TIME_STEPS = 50
		self.breakdown_k = -1
		
		self.SURF_FLUX = numpy.random.randint(1,10)
		print "surface flux", self.SURF_FLUX
		
		self.points = self.INDICES_PER_SIDE*2+1
		self.left_x = -self.HALF_LENGTH
		self.right_x = self.HALF_LENGTH
		self.substrate_struct = SubstrateStructure(
			Water(self.left_x, self.right_x, self.points))
		self.init_conc = (self.INIT_CONC * 
			numpy.ones(shape = (self.points), dtype=numpy.float32))
		self.molecule = "glucose"
		self.diff_coeff = self.substrate_struct[0].diff_coeff[self.molecule]
		
	def test_substrate_setup(self):
		self.assertEqual(self.substrate_struct.boundary_index,[0,self.points-1])
		self.assertEqual(self.substrate_struct.boundary_xloc, 
									[self.left_x,self.right_x])
		
	def test_concentration_boundaries(self):
		self.left_bound = ConcentrationBoundary(self.SURF_CONC,self.SURF_CONC)
		self.right_bound = ConcentrationBoundary(self.SURF_CONC,self.SURF_CONC)
		diff_prob = DiffusionProblem1D(molecule = self.molecule, 
									linear_or_radial = "linear",
									left_boundary = self.left_bound, 
									right_boundary = self.right_bound,
									init_conc = self.init_conc, 
									substrates = self.substrate_struct, 
									delta_t = self.DELTA_T)
		diff_prob.diffuse_multiple_time_steps(self.TIME_STEPS)
		print "concentration linear simulation"
		print diff_prob.conc

#		save_plot.save_plot(self.substrate_struct.total_xloc_list,
#					diff_prob.conc, "diff.png")
		self.solve_concentration_boundaries_analytical()
		print "concentration linear analytical"
		print self.anal_conc_solution
#		diffusion_interface.compare_sim_anal(
#			self.substrate_struct.total_xloc_list,
#			diff_prob.conc, self.anal_conc_solution)
		error = (self.anal_conc_solution - diff_prob.conc) \
					/ self.anal_conc_solution
		print "concentration linear error"
		print error
		for error_value in error:
			self.failIf(abs(error_value) > 0.01)
		
		
	def solve_concentration_boundaries_analytical(self):
		"""from Crank equation 4.17"""
		length = self.HALF_LENGTH
		spacing = self.HALF_LENGTH / self.INDICES_PER_SIDE
		D = self.diff_coeff
		x = self.substrate_struct.total_xloc_list
		t = self.TIME_STEPS * self.DELTA_T
		summation = 0
		for i in range(10):
			summation += (((-1)**i)/(2*i+1)) \
				*numpy.exp(-D*((2*i+1)**2)*(numpy.pi**2)*t/(4*(length**2))) \
				*numpy.cos((2*i+1)*numpy.pi*x/(2*length))
		self.anal_conc_solution = self.INIT_CONC + \
			(self.SURF_CONC-self.INIT_CONC) * (1-(4/numpy.pi)*summation)
			
	def test_flux_boundaries(self):
		self.left_bound_flux = FluxBoundary(self.SURF_FLUX,self.SURF_FLUX)
		self.right_bound_flux = FluxBoundary(self.SURF_FLUX,self.SURF_FLUX)
		diff_prob_flux = DiffusionProblem1D(molecule = self.molecule, 
									linear_or_radial = "linear",
									left_boundary = self.left_bound_flux, 
									right_boundary = self.right_bound_flux,
									init_conc = self.init_conc, 
									substrates = self.substrate_struct, 
									delta_t = self.DELTA_T)
		diff_prob_flux.diffuse_multiple_time_steps(self.TIME_STEPS)
		print "flux linear simulation"
		print diff_prob_flux.conc
		self.solve_flux_boundaries_analytical()
		print "flux linear analytical"
		print self.anal_flux_solution
		error = (self.anal_flux_solution - diff_prob_flux.conc) \
					/ self.anal_flux_solution
		print "flux linear error"
		print error
		for error_value in error:
			self.failIf(abs(error_value) > 0.01)

	def solve_flux_boundaries_analytical(self):
		"""From Crank equation 4.55"""
		length = self.HALF_LENGTH
		spacing = self.HALF_LENGTH / self.INDICES_PER_SIDE
		D = self.diff_coeff
		x = self.substrate_struct.total_xloc_list
		t = self.TIME_STEPS * self.DELTA_T
		summation = 0
		for i in range(1,10):
			summation += (((-1)**i)/i**2) \
				* numpy.exp(-D*(i**2)*(numpy.pi**2)*t/length**2) \
				* numpy.cos(i*numpy.pi*x/length)
		self.anal_flux_solution = self.INIT_CONC \
			+ (self.SURF_FLUX * length/D) \
			* ((D*t/length**2) + ((3*x**2-length**2)/(6*length**2)) \
			- ((2/numpy.pi**2)*summation))
			
	def test_breakdown(self):
		self.left_bound_flux = FluxBoundary(0,0)
		self.right_bound_flux = FluxBoundary(0,0)
		diff_prob_flux = DiffusionProblem1D(molecule = self.molecule, 
									linear_or_radial = "linear",
									left_boundary = self.left_bound_flux, 
									right_boundary = self.right_bound_flux,
									init_conc = self.init_conc, 
									substrates = self.substrate_struct, 
									delta_t = self.DELTA_T,
									breakdown_k = self.breakdown_k)
		diff_prob_flux.diffuse_multiple_time_steps(self.TIME_STEPS)
		anal_enz_depl_solution = self.init_conc
		for ts in range(self.TIME_STEPS):
			print anal_enz_depl_solution
			anal_enz_depl_solution = anal_enz_depl_solution \
				+ anal_enz_depl_solution*self.breakdown_k*self.DELTA_T
		print "enzyme depletion linear simulation"
		print diff_prob_flux.conc	
		print "enzyme depletion linear analytical"	
		print anal_enz_depl_solution
		error = (anal_enz_depl_solution - diff_prob_flux.conc) \
					/ anal_enz_depl_solution
		print "enzyme depletion linear error"
		print error
		for error_value in error:
			self.failIf(abs(error_value) > 0.01)
			
	def test_addition(self):
		self.left_bound = ConcentrationBoundary(self.SURF_CONC,self.SURF_CONC)
		self.right_bound = ConcentrationBoundary(self.SURF_CONC,self.SURF_CONC)
		diff_prob = DiffusionProblem1D(molecule = self.molecule, 
									linear_or_radial = "linear",
									left_boundary = self.left_bound, 
									right_boundary = self.right_bound,
									init_conc = self.init_conc, 
									substrates = self.substrate_struct, 
									delta_t = self.DELTA_T)
		other = (50 *numpy.ones(shape = (self.points), dtype=numpy.float32))
		diff_prob.diffuse_multiple_time_steps(self.TIME_STEPS, 
									external_add_subtract=other)
		print "enzyme linear simulation"
		print diff_prob.conc
	
	def test_zero_flux_boundaries(self):
		self.left_bound_flux = FluxBoundary(0,0)
		self.right_bound_flux = FluxBoundary(0,0)
		diff_prob_flux = DiffusionProblem1D(molecule = self.molecule, 
									linear_or_radial = "linear",
									left_boundary = self.left_bound_flux, 
									right_boundary = self.right_bound_flux,
									init_conc = self.init_conc, 
									substrates = self.substrate_struct, 
									delta_t = self.DELTA_T)
		diff_prob_flux.diffuse_multiple_time_steps(self.TIME_STEPS)
		print "no flux linear simulation"
		print diff_prob_flux.conc
		print "no flux analytical"
		print self.init_conc
		error = (self.init_conc - diff_prob_flux.conc) \
					/ self.init_conc
		print "no flux linear error"
		print error
		for error_value in error:
			self.failIf(abs(error_value) > 0.01)
		
class diffusion_1D_uniform_linear_test_composite_media(unittest.TestCase):
	def setUp(self):
		self.HALF_LENGTH = float(100)
		self.INDICES_PER_SIDE = 10
		self.SURF_CONC = numpy.random.randint(1,100)
		print "surface concentration", self.SURF_CONC
		self.INIT_CONC = numpy.random.randint(1,100)
		print "initial concentration", self.INIT_CONC
		self.DELTA_T = 0.1
		self.TIME_STEPS = 50
		self.breakdown_k = -1
		
		self.SURF_FLUX = numpy.random.randint(1,10)
		print "surface flux", self.SURF_FLUX
		
		self.points = self.INDICES_PER_SIDE*2+1
		self.left_x = -self.HALF_LENGTH
		self.mid_x = 0
		self.right_x = self.HALF_LENGTH
		self.substrate_struct = SubstrateStructure(
			Water(self.left_x, self.mid_x, self.INDICES_PER_SIDE+1),
			Water(self.mid_x, self.right_x, self.INDICES_PER_SIDE+1))
		self.init_conc = (self.INIT_CONC * 
			numpy.ones(shape = (self.points), dtype=numpy.float32))
		self.molecule = "glucose"
		self.diff_coeff = self.substrate_struct[0].diff_coeff[self.molecule]
	
	def test_concentration_boundaries(self):
		self.left_bound = ConcentrationBoundary(self.SURF_CONC,self.SURF_CONC)
		self.right_bound = ConcentrationBoundary(self.SURF_CONC,self.SURF_CONC)
		diff_prob = DiffusionProblem1D(molecule = self.molecule, 
									linear_or_radial = "linear",
									left_boundary = self.left_bound, 
									right_boundary = self.right_bound,
									init_conc = self.init_conc, 
									substrates = self.substrate_struct, 
									delta_t = self.DELTA_T)
		diff_prob.diffuse_multiple_time_steps(self.TIME_STEPS)
		print "concentration linear simulation - composite"
		print diff_prob.conc
		self.solve_concentration_boundaries_analytical()
		print "concentration linear analytical - composite"
		print self.anal_conc_solution
		error = (self.anal_conc_solution - diff_prob.conc) \
					/ self.anal_conc_solution
		print "concentration linear error - composite"
		print error
		for error_value in error:
			self.failIf(abs(error_value) > 0.01)
		
		
	def solve_concentration_boundaries_analytical(self):
		"""from Crank equation 4.17"""
		length = self.HALF_LENGTH
		spacing = self.HALF_LENGTH / self.INDICES_PER_SIDE
		D = self.diff_coeff
		x = self.substrate_struct.total_xloc_list
		t = self.TIME_STEPS * self.DELTA_T
		summation = 0
		for i in range(10):
			summation += (((-1)**i)/(2*i+1)) \
				*numpy.exp(-D*((2*i+1)**2)*(numpy.pi**2)*t/(4*(length**2))) \
				*numpy.cos((2*i+1)*numpy.pi*x/(2*length))
		self.anal_conc_solution = self.INIT_CONC + \
			(self.SURF_CONC-self.INIT_CONC) * (1-(4/numpy.pi)*summation)
			
	def test_flux_boundaries(self):
		self.left_bound_flux = FluxBoundary(self.SURF_FLUX,self.SURF_FLUX)
		self.right_bound_flux = FluxBoundary(self.SURF_FLUX,self.SURF_FLUX)
		diff_prob_flux = DiffusionProblem1D(molecule = self.molecule, 
									linear_or_radial = "linear",
									left_boundary = self.left_bound_flux, 
									right_boundary = self.right_bound_flux,
									init_conc = self.init_conc, 
									substrates = self.substrate_struct, 
									delta_t = self.DELTA_T)
		diff_prob_flux.diffuse_multiple_time_steps(self.TIME_STEPS)
		print "flux linear simulation - composite"
		print diff_prob_flux.conc
		self.solve_flux_boundaries_analytical()
		print "flux linear analytical - composite"
		print self.anal_flux_solution
		error = (self.anal_flux_solution - diff_prob_flux.conc) \
					/ self.anal_flux_solution
		print "flux linear error - composite"
		print error
		for error_value in error:
			self.failIf(abs(error_value) > 0.01)

	def solve_flux_boundaries_analytical(self):
		"""From Crank equation 4.55"""
		length = self.HALF_LENGTH
		spacing = self.HALF_LENGTH / self.INDICES_PER_SIDE
		D = self.diff_coeff
		x = self.substrate_struct.total_xloc_list
		t = self.TIME_STEPS * self.DELTA_T
		summation = 0
		for i in range(1,10):
			summation += (((-1)**i)/i**2) \
				* numpy.exp(-D*(i**2)*(numpy.pi**2)*t/length**2) \
				* numpy.cos(i*numpy.pi*x/length)
		self.anal_flux_solution = self.INIT_CONC \
			+ (self.SURF_FLUX * length/D) \
			* ((D*t/length**2) + ((3*x**2-length**2)/(6*length**2)) \
			- ((2/numpy.pi**2)*summation))

class diffusion_uniform_radial_test(unittest.TestCase):
	def setUp(self):
		self.INDICES = 51
		self.INNER_RADIUS = 10
		self.OUTER_RADIUS = 100
		self.SURF_CONC = numpy.random.randint(1,100)
		self.INIT_CONC = numpy.random.randint(1,100)
		self.DELTA_T = 0.01
		self.TIME_STEPS = 20
		
		self.SURF_FLUX = numpy.random.randint(10,1000)
		
		self.points = self.INDICES
		self.left_x = self.INNER_RADIUS
		self.right_x = self.OUTER_RADIUS
		self.substrate_struct = SubstrateStructure(
			Water(self.left_x, self.right_x, self.points))
		self.init_conc = (self.INIT_CONC * 
			numpy.ones(shape = (self.points), dtype=numpy.float32))
		self.molecule = "glucose"
		self.diff_coeff = self.substrate_struct[0].diff_coeff[self.molecule]
		
	def test_concentration_boundaries(self):
		self.left_bound = ConcentrationBoundary(self.SURF_CONC,self.SURF_CONC)
		self.right_bound = ConcentrationBoundary(self.SURF_CONC,self.SURF_CONC)
		diff_prob = DiffusionProblem1D(molecule = self.molecule, 
									linear_or_radial = "radial",
									left_boundary = self.left_bound, 
									right_boundary = self.right_bound,
									init_conc = self.init_conc, 
									substrates = self.substrate_struct, 
									delta_t = self.DELTA_T)
		diff_prob.diffuse_multiple_time_steps(self.TIME_STEPS)
		print "concentration radial simulation"
		print diff_prob.conc
		self.solve_concentration_radial_analytical()
		print "concentration radial analytical"
		print self.anal_conc_solution
		error = (self.anal_conc_solution - diff_prob.conc) \
					/ self.anal_conc_solution
		print "concentration radial error"
		print error
		for error_value in error:
			self.failIf(abs(error_value) > 0.02)

	def solve_concentration_radial_analytical(self):
		a = self.INNER_RADIUS
		b = self.OUTER_RADIUS
		D = self.diff_coeff
		t = self.TIME_STEPS * self.DELTA_T
		r = self.substrate_struct.total_xloc_list
		summation = 0
		for i in range(1,10):
			summation += ((b*numpy.cos(i*pi)-a)/i) \
				* numpy.sin(i*pi*(r-a)/(b-a)) \
				* numpy.exp(-D*i**2*pi**2*t/(b-a)**2)
		self.anal_conc_solution = self.INIT_CONC \
			+ ((self.SURF_CONC - self.INIT_CONC) \
			* (1+(2/(pi*r))*summation))
	
	def test_flux_boundaries(self):
		self.left_bound = FluxBoundary(self.SURF_FLUX,self.SURF_FLUX)
		self.right_bound = \
			ConcentrationBoundary(self.SURF_CONC,self.SURF_CONC)
		self.init_conc = (numpy.zeros(shape = (self.points), \
			dtype=numpy.float32))
		self.DELTA_T = 0.1
		self.TIME_STEPS = 50
		diff_prob_flux = DiffusionProblem1D(molecule = self.molecule, 
									linear_or_radial = "radial",
									left_boundary = self.left_bound, 
									right_boundary = self.right_bound,
									init_conc = self.init_conc, 
									substrates = self.substrate_struct, 
									delta_t = self.DELTA_T)
		diff_prob_flux.diffuse_multiple_time_steps(self.TIME_STEPS)
		print "concentration radial flux simulation"
		print diff_prob_flux.conc
		self.solve_ss_flux_radial()
		print "concentration radial flux ss analytical"
		print self.ss_flux_radial_solution
		error = (self.ss_flux_radial_solution - diff_prob_flux.conc) \
					/ self.ss_flux_radial_solution
		print "concentration radial flux error"
		print error
		for error_value in error:
			self.failIf(abs(error_value) > 0.10)
		#This error is set a little high - need to correct.
		
	def solve_ss_flux_radial(self):
		a = self.INNER_RADIUS
		b = self.OUTER_RADIUS
		f = self.SURF_FLUX
		d = self.diff_coeff
		k = self.SURF_CONC
		r = self.substrate_struct.total_xloc_list
		self.ss_flux_radial_solution = (a**2 *f*(b-r) + b*d*k*r)/(b*d*r)
	
	def test_zero_flux_boundaries(self):
		self.left_bound = FluxBoundary(0,0)
		self.right_bound = FluxBoundary(0,0)
		self.init_conc = self.INIT_CONC*(numpy.ones(shape = (self.points), \
			dtype=numpy.float32))
		self.DELTA_T = 0.1
		self.TIME_STEPS = 50
		diff_prob_flux = DiffusionProblem1D(molecule = self.molecule, 
									linear_or_radial = "radial",
									left_boundary = self.left_bound, 
									right_boundary = self.right_bound,
									init_conc = self.init_conc, 
									substrates = self.substrate_struct, 
									delta_t = self.DELTA_T)
		diff_prob_flux.diffuse_multiple_time_steps(self.TIME_STEPS)
		print "radial no flux simulation"
		print diff_prob_flux.conc
		print "radial no flux analytical"
		no_flux_radial = self.init_conc
		print no_flux_radial
		error = (no_flux_radial - diff_prob_flux.conc) \
					/ no_flux_radial
		print "radial noflux error"
		print error
		for error_value in error:
			self.failIf(abs(error_value) > 0.02)


if __name__ == '__main__':
	unittest.main()
