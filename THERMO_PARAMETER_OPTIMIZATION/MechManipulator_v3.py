import os
import numpy as np
from copy import deepcopy
from MechanismParser import Parser

class Manipulator:
	def __init__(self, copy_of_mech, unsrt_object, perturbation, perturbation_type="opt", parameter_dict=None,flag="reaction"):
		self.mechanism = deepcopy(copy_of_mech)
		#print(unsrt_object)
		#print(len(perturbation))
		self.unsrt = unsrt_object
		self.perturbation_type = perturbation_type
		self.parameter_dict = parameter_dict
		#print(parameter_dict)
		self.flag = flag
		if flag == "reaction":
			self.rxn_type = parameter_dict["type"]
			self.rxn_data = parameter_dict["data"]
			self.rxn_dict = parameter_dict["reaction"]
			self.rxn_list = [parameter_dict["reaction"][index] for index in parameter_dict["reaction"]]
			self.perturbation = perturbation
		else:
			self.perturbation = []
			self.parameter_dict = parameter_dict
			count = 0 
			for species in self.unsrt:
				temp = []
				s = self.unsrt[species].selection
				#s = self.unsrt[species] ## causing issues
				for i in s:
					temp.append(perturbation[count])
					count+=1
				self.perturbation.append(temp)
		#print(self.perturbation)
	

	def calculate_enthalpy_and_entropy(self,T,coeff, R=8.314):
		"""
		Calculate enthalpy and entropy at a given temperature T using NASA polynomial coefficients.
		R is in cal/(mol K).
		"""
		a1,a2,a3,a4,a5,a6,a7 = coeff
		# Calculate enthalpy (H) at temperature T
		H_RT = (
		a1 + a2 * T / 2 + a3 * T**2 / 3 +
		a4 * T**3 / 4 + a5 * T**4 / 5 + a6 / T
		)
		H = H_RT*T  # Returns enthalpy H in cal/mol

		# Calculate entropy (S) at temperature T
		S_R = (
		a1 * np.log(T) + a2 * T + a3 * T**2 / 2 +
		a4 * T**3 / 3 + a5 * T**4 / 4 + a7
		)
		S = S_R  # Returns entropy S in 

		return H, S

	
	def cp_component_of_NASA7(self,T,coeff):
		R = 8.314
		
		a1, a2, a3, a4, a5 = coeff
		
		H_RT = (
		a1 + a2 * T / 2 + a3 * T**2 / 3 +
		a4 * T**3 / 4 + a5 * T**4 / 5 
		)
		H_RT_1000 = H_RT *T  # Returns enthalpy H in cal/mol
		
		S_R = (
		a1 * np.log(T) + a2 * T + a3 * T**2 / 2 +
		a4 * T**3 / 3 + a5 * T**4 / 4
		)
		S = S_R  # Returns entropy S in 
		return H_RT_1000,S_R	

	def get_NASA7_Curves(self,T,coeff):
		R = 8.314
		
		a1, a2, a3, a4, a5, a6, a7 = coeff
		
		H_RT = (
		a1 + a2 * T / 2 + a3 * T**2 / 3 +
		a4 * T**3 / 4 + a5 * T**4 / 5 + a6 / T
		)
		H_RT_1000 = H_RT*T   # Returns enthalpy H in cal/mol
		S_R = (
		a1 * np.log(T) + a2 * T + a3 * T**2 / 2 +
		a4 * T**3 / 3 + a5 * T**4 / 4 + a7
		)
		return H_RT_1000,S_R
	'''
	def get_High_a5_a6(self,T,nominal_coeff_low,perturbed_high):
		
		a5_,a6_ = self.get_NASA7_Curves(T,nominal_coeff_low)
		
		cp_a5,cp_a6 = self.cp_component_of_NASA7(T,perturbed_high)
		
		return a5_ - cp_a5, a6_ - cp_a6
	'''
	def solve_a6_a7_from_target_HS(self, T_ref, cp_coeff, H_target, S_target):

		a1, a2, a3, a4, a5 = cp_coeff

		H_cp = T_ref * (
			a1
			+ a2*T_ref/2.0
			+ a3*T_ref**2/3.0
			+ a4*T_ref**3/4.0
			+ a5*T_ref**4/5.0
		)

		S_cp = (
			a1*np.log(T_ref)
			+ a2*T_ref
			+ a3*T_ref**2/2.0
			+ a4*T_ref**3/3.0
			+ a5*T_ref**4/4.0
		)

		a6 = H_target - H_cp
		a7 = S_target - S_cp

		return float(a6), float(a7)
	def get_nominal_HS_at_T(self, T, coeff):

		a1, a2, a3, a4, a5, a6, a7 = coeff

		H = T * (
			a1
			+ a2*T/2.0
			+ a3*T**2/3.0
			+ a4*T**3/4.0
			+ a5*T**4/5.0
			+ a6/T
		)

		S = (
			a1*np.log(T)
			+ a2*T
			+ a3*T**2/2.0
			+ a4*T**3/3.0
			+ a5*T**4/4.0
			+ a7
		)

		return float(H), float(S)
	def adjust_a6_a7_for_perturbation(self,T,coefficients):
		"""
		Adjust a6 and a7 to ensure H_mod(298 K) and S_mod(298 K) match original values
		after perturbing a1-a5.

		Parameters:
		coefficients: array-like of length 7 [a1, a2, a3, a4, a5, a6, a7]

		Returns:
		Modified a6 and a7 to maintain enthalpy and entropy values at 298 K.
		"""
		T_ref = T  # Reference temperature in Kelvin
		R = 8.314  # Gas constant in J/(mol K)
		a1, a2, a3, a4, a5, a6, a7 = coefficients

		# Adjust a6 and a7 based on the delta values
		a6_mod = a6 - T_ref*(a1 + a2 * T_ref/ 2 + a3 * T_ref**2 / 3 + a4 * T_ref**3 / 4 + a5 * T_ref**4 / 5)
		a7_mod = a7 - (a1 * np.log(T_ref) + a2 * T_ref + a3 * T_ref**2 / 2 + a4 * T_ref**3 / 3 + a5 * T_ref**4 / 4)

		return a6_mod, a7_mod

	def perturb_cp(self, index, beta, mechanism):
		thermo_data = mechanism["species"][index]['thermo']['data']
		T = [298,1000]
		for i in range(2):
			a6_mod, a7_mod = self.adjust_a6_a7_for_perturbation(T[i],thermo_data[i])
			thermo_data[i][5] = float(a6_mod)
			thermo_data[i][6] = float(a7_mod)
			thermo_data[i][:5] = [float(value) * float(beta) for value in thermo_data[i][:5]]
		
		mechanism["species"][index]['thermo']['data']	 = deepcopy(thermo_data)
		return mechanism

	def perturb_enthalpy(self, index, beta, mechanism):
		thermo_data = mechanism["species"][index]['thermo']['data']	
		for i in range(2):
			thermo_data[i][5] += float(beta)  # Perturb the 7th entry
		mechanism["species"][index]['thermo']['data']	 = deepcopy(thermo_data)
		return mechanism
		
	def perturb_entropy(self, index, beta, mechanism):
		thermo_data = mechanism["species"][index]['thermo']['data']
		for i in range(2):
			thermo_data[i][6] += float(beta)  # Perturb the 7th entry
		mechanism["species"][index]['thermo']['data'] = deepcopy(thermo_data)
		return mechanism
		
	def _extract_species_data(self, parameter_dict):
		self.species_data = {}
		for species in parameter_dict:
			for index,dict_ in enumerate(self.mechanism["species"]):
				if dict_["name"] == species:
					self.species_data[species] = index
		#print(self.species_data)

	def getRxnDetails(self):
		rxn_dict = {}
		rxn_data = self.mechanism["reactions"]
		for rxn in self.rxn_list:
			new_rxn_data = {}
			temp = []
			index_ = []
			for index, data in enumerate(rxn_data):
				if rxn == data["equation"]:
					temp.append(data)
					index_.append(index)
			new_rxn_data["temp"] = temp
			new_rxn_data["index"] = index_
			rxn_dict[rxn] = new_rxn_datamani
		return rxn_dict

	def getRxnType(self):
		rxn_type = {}
		rxn_data = self.mechanism["reactions"]
		for rxn in self.rxn_list:
			for data in rxn_data:
				if rxn in data["equation"]:
					if "type" in data:
						if data["type"] == "three-body":
							rxn_type[data["equation"]] = "ThirdBody"
						elif data["type"] == "falloff":
							rxn_type[data["equation"]] = "Falloff"
						elif data["type"] == "pressure-dependent-Arrhenius" and "duplicate" not in data:
							rxn_type[data["equation"]] = "PLOG"
						elif data["type"] == "pressure-dependent-Arrhenius" and "duplicate" in data:
							rxn_type[data["equation"]] = "PLOG-Duplicate"
							break
					elif "duplicate" in data:
						rxn_type[data["equation"]] = "Duplicate"
						break
					else:
						rxn_type[data["equation"]] = "Elementary"
		return rxn_type
	
	def HeatCapacityPerturbation(self,index,species,zeta,mechanism):

		convertor = np.asarray(self.unsrt[species].selection)

		species_object = self.unsrt[species]
		species_Data = self.unsrt[species]
		cov = species_object.cov
		temprature_range = species_object.temp_limit
		thermo_details = mechanism["species"][index]["thermo"]

		if (temprature_range) == "Low":

			c0 = self.unsrt[species].nominal
			p0 = self.unsrt[species].nominal[0:5]

			unsrt_perturbation = np.asarray(cov.dot(zeta)).flatten()
			p = p0 + convertor*unsrt_perturbation

			T_ref = 298.15
			T_mid = self.unsrt[species].common_temp

			# Nominal enthalpy and entropy at the reference temperature
			H_nom, S_nom = self.get_nominal_HS_at_T(T_ref, c0)

			# Perturbation (replace later with actual uncertainty limits)
			delta_H = np.random.uniform(-0.020,0.020)
			delta_S = np.random.uniform(-0.010,0.010)

			H_target = H_nom*(1.0 + delta_H)
			S_target = S_nom*(1.0 + delta_S)

			# Recover a6 and a7 using the reference temperature
			a6, a7 = self.solve_a6_a7_from_target_HS(
				T_ref,
				p,
				H_target,
				S_target
			)

			thermo_details["data"][0] = [
				p[0], p[1], p[2], p[3], p[4],
				a6, a7
			]

			mechanism["species"][index]["thermo"] = deepcopy(thermo_details)

			# Propagate to T_mid for continuity
			H_mid, S_mid = self.get_nominal_HS_at_T(
				T_mid,
				[p[0], p[1], p[2], p[3], p[4], a6, a7]
			)

			if not hasattr(self, "HS_targets"):
				self.HS_targets = {}

			species_name = species.split(":")[0]

			# Store propagated values at T_mid for the high-temperature fit
			self.HS_targets[species_name] = (
				H_mid,
				S_mid
			)

		else:

			T_mid = self.unsrt[species].common_temp

			p0_high = self.unsrt[species].nominal[0:5]

			unsrt_perturbation = np.asarray(cov.dot(zeta)).flatten()
			p = p0_high + convertor*unsrt_perturbation

			species_name = species.split(":")[0]

			# Target values propagated from the low-temperature polynomial
			H_target, S_target = self.HS_targets[species_name]

			b6, b7 = self.solve_a6_a7_from_target_HS(
				T_mid,
				p,
				H_target,
				S_target
			)

			thermo_details["data"][1] = [
				p[0], p[1], p[2], p[3], p[4],
				b6, b7
			]

			mechanism["species"][index]["thermo"] = deepcopy(thermo_details)
		
	def ElementaryPerturbation(self, index, beta, mechanism):
		perturbation_factor = beta
		reaction_details = mechanism["reactions"][index]["rate-constant"]
		pre_exponential_factor = np.log(float(reaction_details["A"]))
		reaction_details["A"] = float(np.exp(pre_exponential_factor + perturbation_factor))
		mechanism["reactions"][index]["rate-constant"] = deepcopy(reaction_details)
		return mechanism
	
	def PlogPerturbation(self, index, beta, mechanism):
		if beta == 0:
			perturbation_factor = 1.0
		else:
			perturbation_factor = np.exp(beta)
		reaction_details = mechanism["reactions"][index]["rate-constants"]
		new_rxn_details = []
		for rxn in reaction_details:
			temp = {
				"P": rxn["P"],
				"A": float(float(rxn["A"]) * perturbation_factor),
				"b": rxn["b"],
				"Ea": rxn["Ea"]
			}
			new_rxn_details.append(temp)
		mechanism["reactions"][index]["rate-constants"] = deepcopy(new_rxn_details)
		return mechanism

	def DupPlogPerturbation(self, rxn_object, beta, mechanism):
		rxn_object_a = {"temp": [], "index": []}
		rxn_object_a["temp"].append(rxn_object["temp"][0])
		rxn_object_a["index"].append(int(rxn_object["index"][0]) - 1)
		rxn_object_b = {"temp": [], "index": []}
		rxn_object_b["temp"].append(rxn_object["temp"][1])
		rxn_object_b["index"].append(int(rxn_object["index"][1]) - 1)
		new_mechanism = self.PlogPerturbation(rxn_object_a, beta, mechanism)
		new_mechanism = self.PlogPerturbation(rxn_object_b, beta, new_mechanism)
		return new_mechanism

	def DupElementaryPerturbation(self, rxn_object, beta, mechanism):
		rxn_object_a = {"temp": [], "index": []}
		rxn_object_a["temp"].append(rxn_object["temp"][0])
		rxn_object_a["index"].append(int(rxn_object["index"][0]) - 1)
		rxn_object_b = {"temp": [], "index": []}
		rxn_object_b["temp"].append(rxn_object["temp"][1])
		rxn_object_b["index"].append(int(rxn_object["index"][1]) - 1)
		new_mechanism = self.ElementaryPerturbation(rxn_object_a, beta, mechanism)
		new_mechanism = self.ElementaryPerturbation(rxn_object_a, beta, new_mechanism)
		return new_mechanism

	def TroePerturbation(self, index, beta, mechanism):
		perturbation_factor = beta
		reaction_details_low = mechanism["reactions"][index]["low-P-rate-constant"]
		pre_exponential_factor_low = np.log(float(reaction_details_low["A"]))
		reaction_details_low["A"] = float(np.exp(pre_exponential_factor_low + perturbation_factor))
		mechanism["reactions"][index]["low-P-rate-constant"] = deepcopy(reaction_details_low)
		return mechanism

	def doPerturbation(self):  # purtubing data.... flag: 'reaction' for reaction perturbation, 'thermo' for thermo perturbation.
		#print(self.flag,)
		if self.flag == "thermo":
			mechanism = self.mechanism
			self._extract_species_data(self.parameter_dict)
			perturb = ""
			count = 0
			#print(self.species_data)
			for index,species in enumerate(self.unsrt):
				beta = np.asarray(self.perturbation[index])  # using the purturbation array to modify cp, h and s
				index_ = self.species_data[species.split(":")[0]]
				#if float(abs(beta)) > 0:
				perturb += f"{species}\t{beta}"
				#type_of_rxn = rxn_type[index]
				#data = rxn_data[rxn]

				#if type_of_rxn == "Elementary":
				#print("Entering the heat capacity perturbation")
				new_mechanism = self.HeatCapacityPerturbation(index_,species, beta, mechanism)
				#if float(abs(beta[0])) > 0:
					#perturb += f"{species}_cp\n"
					#perturb += f"{beta[0]}"
					#mechanism = self.perturb_cp(index, beta[0],mechanism)
				
				#if float(abs(beta[1])) > 0:
					#perturb += f"{species}_H\n"
					#perturb += f"{beta[1]}"
					#mechanism = self.perturb_enthalpy(index, beta[1],mechanism)
				#if float(abs(beta[2])) > 0:
					#perturb += f"{species}_S\n"
					#perturb += f"{beta[2]}"
					#mechanism = self.perturb_entropy(index, beta[2],mechanism)
				#count+=1
			#"""
			return mechanism,perturb

		elif self.flag == "reaction" :
			rxn_type = self.rxn_type
			rxn_data = self.rxn_data
			mechanism = self.mechanism
			rxn_dict = self.rxn_dict
			perturb = ""
			for i, index in enumerate(rxn_dict):
				rxn = rxn_dict[index]
				index_ = index - 1
				beta = np.asarray(self.perturbation[i])
				if float(abs(beta)) > 0:
					perturb += f"{rxn}\t{beta}"
					type_of_rxn = rxn_type[index]
					data = rxn_data[rxn]

					if type_of_rxn == "Elementary":
						new_mechanism = self.ElementaryPerturbation(index_, beta, mechanism)

					elif type_of_rxn == "PLOG":
						new_mechanism = self.PlogPerturbation(index_, beta, mechanism)

					elif type_of_rxn == "PLOG-Duplicate":
						new_mechanism = self.DupPlogPerturbation(data, beta, mechanism)

					elif type_of_rxn == "Duplicate":
						new_mechanism = self.DupElementaryPerturbation(data, beta, mechanism)

					elif type_of_rxn == "ThirdBody":
						new_mechanism = self.ElementaryPerturbation(index_, beta, mechanism)

					elif type_of_rxn == "Falloff":
						new_mechanism = self.TroePerturbation(index_, beta, mechanism)
			return new_mechanism,perturb
		else:
			raise AssertionError(f"Invalid flag: {self.flag}!!\n\t-valid flag types ['thermo','reaction']\n")
