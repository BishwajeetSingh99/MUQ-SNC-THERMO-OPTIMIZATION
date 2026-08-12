try:
    import ruamel_yaml as yaml
except ImportError:
    from ruamel import yaml
import yaml

def custom_representer(dumper, value):
    return dumper.represent_scalar('tag:yaml.org,2002:float', str(value))

yaml.add_representer(float, custom_representer)

def convert_to_builtin(obj):
    if isinstance(obj, dict):
        return {convert_to_builtin(k): convert_to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_builtin(i) for i in obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj
import gc  
import os
import glob
import yaml
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import sys,os,glob,csv
from copy import deepcopy
import pickle
from sklearn.model_selection import train_test_split
# custom modlues .................................................
import ResponseSurface as PRS
import reaction_selection as rs 
import combustion_target_class  
import DesignMatrix as DM           
import simulation_manager as simulator 
import data_management
import create_parameter_dictionary as create_dict
import Uncertainty as uncertainty
from sample_generator  import sample_plot_all_species as sample_plot
from MechanismParser import Parser
from MechManipulator_v3 import Manipulator
import ga_optimizer_module_V as ga_mod
### KEY WORDS #######
optType = "optimization_type"
targets = "targets"
mech = "mechanism"
pre_file = "Initial_pre_file"
count = "Counts"
countTar = "targets_count"
home_dir = os.getcwd()
fuel = "fuel"
fuelClass = "fuelClass"
bin_solve = "solver_bin"
bin_opt = "bin"
globRxn = "global_reaction"
countThreads = "parallel_threads"
unsrt = "uncertainty_data"
thermoF = "thermo_file"
transF = "trans_file"
order = "Order_of_PRS"
startProfile = "StartProfilesData"
design = "Design_of_PRS"
countRxn = "total_reactions"
fT = "fileType"
add = "addendum"

#########################################
###    Reading the input file        ####
#########################################
if len(sys.argv) > 2:
    input_file = open(sys.argv[1],'r')
    optInputs = yaml.safe_load(input_file)
    rxn_list =[i.strip("\n") for i in open(sys.argv[2],"r").readlines()]
    species_list = [i.strip("\n") for i in open(sys.argv[3],"r").readlines()]
    #print(rxn_list)
    print("\n\t########################\n\tInput file , List of reactions, List of Species are found\n\t########################\n")
    #raise AssertionError("Stop")
elif len(sys.argv) > 1:
    input_file = open(sys.argv[1],'r')
    optInputs = yaml.safe_load(input_file)
    rxn_list = []
    species_list = []
    print("\n\t########################\n\tInput file found\n\t########################\n")
else:
    print("Please enter a valid input file name as arguement. \n Two arguments can be passed:\n\t1. Traget opt file\n\t2. List of reactions\nThe code will still work by passing only the first argument\n\nProgram exiting")
    exit()
#print("list of reactions ::::",rxn_list)
iFile = str(os.getcwd())+"/"+str(sys.argv[1])
dataCounts = optInputs[count]
binLoc = optInputs["Bin"]
inputs = optInputs["Inputs"]
locations = optInputs["Locations"]
startProfile_location = optInputs[startProfile]
stats_ = optInputs["Stats"]
global A_fact_samples

A_fact_samples = stats_["Sampling_of_PRS"]
if "sensitive_parameters" not in stats_:
    stats_["sensitive_parameters"] = "zeta"
    optInputs["Stats"]["sensitive_parameters"] = "zeta"
if "Arrhenius_Selection_Type" not in stats_:
    stats_["Arrhenius_Selection_Type"] = "all"
    optInputs["Stats"]["Arrhenius_Selection_Type"] = "all"
unsrt_location = locations[unsrt]
mech_file_location = locations[mech]
thermo_file_location = locations[thermoF]
trans_file_location = locations[transF]
fileType = inputs[fT]
samap_executable = optInputs["Bin"]["samap_executable"]
jpdap_executable = optInputs["Bin"]["jpdap_executable"]

if fileType == "chemkin":
    file_specific_input = "-f chemkin"
else:
    file_specific_input = ""
fuel = inputs[fuel]
global_reaction = inputs[globRxn]
design_type = stats_[design]
parallel_threads = dataCounts[countThreads]
targets_count = int(dataCounts["targets_count"])
rps_order = stats_[order]
PRS_type = stats_["PRS_type"]

#######################READ TARGET FILE ###################

print("\nParallel threads are {}".format(parallel_threads))
targetLines = open(locations[targets],'r').readlines()
addendum = yaml.safe_load(open(locations[add],'r').read())

#########################################
###    Reading the mechanism file    ####
#########################################

MECH = locations["mechanism"]
carbon_number = optInputs["SA"]["carbon_number"]

with open(MECH,'r') as file_:
    yaml_mech = file_.read()
mechanism = yaml.safe_load(yaml_mech)
analysis_type = optInputs['Inputs'].get('AnalysisType', None)

"""
#mechanism = yaml.safe_load(yaml_mech)
species = mechanism['phases'][0]["species"] # .................takes all the species availabel in mechanism file....................
species_data = mechanism["species"] # ................ takes data corresponding to each species like..........composition, thermo,transport.................
#print(species_data)
#raise AssertionError
reactions = mechanism["reactions"] #....... takes all the reactions present in mechanism file ....#3984 in case of butonate- MB2D.yaml file.....................
######...........we can choose thermo data here instead of reactions for sensitivity analysis of thermo data .........................#########


#print(reaction_dict)
#print(selected_species)
#raise AssertionError("stop")


analysis_type = optInputs['Inputs'].get('AnalysisType', None)
print("Analysis Type is :" ,analysis_type)
parameter_dict,selected_species,selected_reactions = create_dict.dictionary_creator(analysis_type, mechanism,carbon_number,rxn_list,species_list)
print(f"Total species selected:  {len(selected_species)}\n")
#print(parameter_dict)
#raise AssertionError
#raise AssertionError("STOP")
"""
####################################################
##  Unloading the target data                 ##
## TARGET CLASS CONTAINING EACH TARGET AS A CASE  ##
####################################################

targetLines = open(locations[targets],'r').readlines()
addendum = yaml.safe_load(open(locations[add],'r').read())

target_list = []
c_index = 0
string_target = ""

# --- ADD THESE THREE LINES BEFORE THE LOOP ---
groups = []
current_id = None
group_start_idx = 0

for target in targetLines[:targets_count]:
    if "#" in target:
        target = target[:target.index('#')]    
    add = deepcopy(addendum)
    t = combustion_target_class.combustion_target(target,add,c_index)
    
    # --- ADD THIS GROUP TRACKING BLOCK ---
    if current_id is None:
        current_id = t.dataSet_id
    elif t.dataSet_id != current_id:
        # ID changed, save boundaries of the completed group (start, end)
        groups.append((group_start_idx, c_index))
        # Reset markers for the next group block
        group_start_idx = c_index
        current_id = t.dataSet_id
    # -------------------------------------

    string_target+=f"{t.dataSet_id}|{t.target}|{t.species_dict}|{t.temperature}|{t.pressure}|{t.phi}|{t.observed}|{t.std_dvtn}\n"
    c_index +=1
    target_list.append(t)
case_dir = range(0,len(target_list))

# --- ADD THIS LINE AFTER THE LOOP TO CLOSE THE FINAL GROUP ---
if current_id is not None:
    groups.append((group_start_idx, c_index))

print("\n\toptimization targets identified\n")
target_file = open("target_data.txt","w")
target_file.write(string_target)
target_file.close()
print(groups)
############################################
##  Uncertainty Quantification            ##
##                         		  ##
############################################

if "unsrt.pkl" not in os.listdir():
    UncertDataSet = uncertainty.uncertaintyData(locations, binLoc)
    ############################################
    ##   Get unsrt data from UncertDataSet    ##
    ############################################

    unsrt_data = UncertDataSet.extract_uncertainty()
    
    # Save the object to a binary pickle file
    with open('unsrt.pkl', 'wb') as file_:
        pickle.dump(unsrt_data, file_)
    
    # ==============================================================================
    # DYNAMIC INSPECTION: DUMP EVERYTHING IN UNSRT DATA TO A TEXT FILE
    # ==============================================================================
    with open('thermo_species_data.txt', 'w') as txt_file:
        txt_file.write("# =========================================================\n")
        txt_file.write("# COMPLETE DIAGNOSTIC DUMP OF UNSRT_DATA OBJECTS\n")
        txt_file.write("# =========================================================\n\n")
        
        # If unsrt_data is a dictionary mapping species names to objects
        if isinstance(unsrt_data, dict):
            for key, obj in unsrt_data.items():
                txt_file.write(f"\n=========================================================\n")
                txt_file.write(f"KEY / SPECIES: {key}\n")
                txt_file.write(f"OBJECT TYPE: {type(obj)}\n")
                txt_file.write(f"=========================================================\n")
                
                # Extract all attributes present on this object dynamically
                try:
                    obj_attrs = vars(obj)
                except TypeError:
                    try:
                        obj_attrs = obj.__dict__
                    except AttributeError:
                        txt_file.write(f"Could not read __dict__ for this object. Raw string representation: {str(obj)}\n")
                        continue
                
                for attr_name, attr_val in obj_attrs.items():
                    txt_file.write(f"\n--> Attribute: {attr_name}\n")
                    txt_file.write(f"    Value: {repr(attr_val)}\n")
        else:
            # Fallback if unsrt_data itself is a single big master object instead of a dict
            txt_file.write(f"unsrt_data root type: {type(unsrt_data)}\n")
            try:
                root_attrs = vars(unsrt_data)
                for attr_name, attr_val in root_attrs.items():
                    txt_file.write(f"\n--> Attribute: {attr_name}\n")
                    txt_file.write(f"    Value: {repr(attr_val)}\n")
            except Exception as e:
                txt_file.write(f"Failed to parse root object variables: {str(e)}\n")
                
    print("Uncertainty analysis finished and text database written.")

else:
    # Load the object from the file
    with open('unsrt.pkl', 'rb') as file_:
        unsrt_data = pickle.load(file_)
    print("Uncertainty analysis already finished")
    
    
    
    
selected_species = []
parameter_dictionary,sellected_species,selected_reactions = create_dict.dictionary_creator(unsrt_data, analysis_type, mechanism,carbon_number,rxn_list,species_list)
#print(parameter_dictionary)
#print("sellected_species for parameter_dict \t ", sellected_species)
for species in unsrt_data:
    selected_species.append(species)
    #print(species ,dir(unsrt_data[species]) )

print(f"Total species selected:  {len(selected_species)}\n")
print(selected_species)
#raise AssertionError
analysis_type = optInputs['Inputs'].get('AnalysisType', None)
if selected_species != sellected_species:
	raise AssertionError('Species mismatch in unsrt_data and parameter_dictionary')


#For PLotting purpose
'''
design_matrix = DM.DesignMatrix(unsrt_data,"Tcube",5,10).get_thermo_samples()
#print(design_matrix)

sample_plot(unsrt_data , design_matrix) # generating sample plots in the folder "sample_plots"

raise AsssertionError("check DM")   #### for plotting the samples
'''
def getTotalUnknowns(N):
    n_ = 1 + 5*N + (5*N*(5*N-1))/2
    return int(n_)
    
def getSim(n,design):
    n_ = getTotalUnknowns(n)
    if design == "A-facto":
        sim = int(A_fact_samples)*n_
    elif design == "Tcube":
        #sim = 7 *n_
        sim = 5*n_
        
    else:
        sim = 9*n_    
    return sim
  

#design_matrix = DM.DesignMatrix(unsrt_data,"Tcube",5,len_active_params).get_thermo_samples()
#sample_plot(unsrt_data , design_matrix) # generating sample plots in the folder "sample_plots"
#raise AssertionError("UQ analysis is Done!!")

len_active_params = 7*len(selected_species)
thermo_design = "Tcube"
#thermo_design = "sample"

#no_sim = getSim(len_active_params,thermo_design)
#########################################
###    Creating Design Matrix for    ####
###    sensitivity analysis          ####
#########################################
## no error upto here.. problem is in creating the design matrix 
if analysis_type == "reaction":
    """
    For sensitivity analysis of reactions we create two design matrices:
        - For one, we multiply all reactions by a factor of 2
        - For the second, we divide all reactions by a factor of 0.5
    """
    if "DesignMatrix_x0_a_fact.csv" not in os.listdir():
        design_matrix_x0 = DM.DesignMatrix(selected_reactions, design_type, len(parameter_dictionary)).getNominal_samples(flag=analysis_type)
        s = ""
        for row in design_matrix_x0:
            for element in row:
                s += f"{element},"
            s += "\n"
        with open('DesignMatrix_x0_a_fact.csv', 'w') as ff:
            ff.write(s)
    else:
        design_matrix_file = open("DesignMatrix_x0_a_fact.csv").readlines()
        design_matrix_x0 = []
        for row in design_matrix_file:
            design_matrix_x0.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])

    if "DesignMatrix_x2_a_fact.csv" not in os.listdir():
        design_matrix_x2 = DM.DesignMatrix(selected_reactions, design_type, len(parameter_dictionary)).getSA_samples(flag=analysis_type)
        s = ""
        for row in design_matrix_x2:
            for element in row:
                s += f"{element},"
            s += "\n"
        with open('DesignMatrix_x2_a_fact.csv', 'w') as ff:
            ff.write(s)
    else:
        design_matrix_file = open("DesignMatrix_x2_a_fact.csv").readlines()
        design_matrix_x2 = []
        for row in design_matrix_file:
            design_matrix_x2.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])

elif analysis_type == "thermo":

    '''
    if "DesignMatrix_x0_a_fact.csv" not in os.listdir():
        design_matrix_x0 = design_matrix
        s = ""
        for row in design_matrix_x0:
            for element in row:
                s += f"{element},"
            s += "\n"
        with open('DesignMatrix_x0_a_fact.csv', 'w') as ff:
            ff.write(s)
    else:
        design_matrix_file = open("DesignMatrix_x0_a_fact.csv").readlines()
        design_matrix_x0 = []
        for row in design_matrix_file:
            design_matrix_x0.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])
    '''
    def getTotalUnknowns(N):
        n_ = 1 + N + (N*(N+1))/2
        return int(n_)
        
    def getSim(n,design):
        n_ = getTotalUnknowns(n)
        if design == "A-facto":
            sim = int(A_fact_samples)*n_
        elif design == "Tcube":
            sim = 3*n	# test case
            #sim = 5*n_
        else:
            sim = 5*n_    
            #sim = int(0.1*n_)
        return sim
    no_of_sim = {}
    if "DesignMatrix.csv" not in os.listdir(): ## 
        no_of_sim_ = getSim(len_active_params,thermo_design)
        print(len_active_params)
        print("no of simulations required",no_of_sim_)
        design_matrix = DM.DesignMatrix(unsrt_data,thermo_design,no_of_sim_,len_active_params).get_thermo_samples()  # this is the line from where we are getting erros in samples
        no_of_sim_ = len(design_matrix)
    else:
        no_of_sim_ = getSim(len_active_params,thermo_design)
        print("\n\n\n no of simulations ", no_of_sim_)
        #raise AssertionError
        design_matrix_file = open("DesignMatrix.csv").readlines()
        design_matrix = []
        no_of_sim_ = len(design_matrix_file)
        for row in design_matrix_file:
            design_matrix.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])
        design_matrix = np.array(design_matrix)
    #raise AssertionError("Design Matrix created!!")
        #sample_plot(unsrt_data , design_matrix)
        #raise AssertionError("Stop!")
        design_matrix_dict = {}
        for case in case_dir:
            design_matrix_dict[case] = design_matrix
            no_of_sim[case] = no_of_sim_



################################################################################
# PLACE ANCHOR RIGHT BEFORE THE SAMPLE PLOT STATEMENT AS REQUESTED
################################################################################
original_directory = os.getcwd()

#print(design_matrix)
#print("1st sample is plotted")
print("plotting cp samples")
#sample_plot(unsrt_data , design_matrix) # generating sample plots in the folder "sample_plots"
###################################################
#raise AssertionError("stop")
SSM = simulator.SM(target_list,optInputs,unsrt_data,design_matrix, ParameterDictionary = parameter_dictionary,flag = analysis_type)

if "Perturbed_Mech" not in os.listdir():
    os.mkdir("Perturbed_Mech")
    print("\nPerturbing the Mechanism files\n")

    chunk_size = 500
    params_yaml = [design_matrix[i:i+chunk_size] for i in range(0, len(design_matrix), chunk_size)]
    count = 0
    yaml_loc = []
    for params in params_yaml:
        yaml_list = SSM.getYAML_List(params)
        #yaml_loc = []
        location_mech = []
        index_list = []
        for i,dict_ in enumerate(yaml_list):
            index_list.append(str(count+i))
            location_mech.append(os.getcwd()+"/Perturbed_Mech")
            yaml_loc.append(os.getcwd()+"/Perturbed_Mech/mechanism_"+str(count+i)+".yaml")
        count+=len(yaml_list)
        #gen_flag = False
        #SSM.getPerturbedMechLocation(yaml_list,location_mech,index_list)
        SSM.getPerturbedMechLocation(yaml_list,location_mech,index_list)
        print(f"\nGenerated {count} files!!\n")
    print("\nGenerated the YAML files required for simulations!!\n")
else:
    print("\nYAML files already generated!!")
    yaml_loc = []
    location_mech = []
    index_list = []
    for i,sample in enumerate(design_matrix):
        index_list.append(i)
        location_mech.append(os.getcwd()+"/Perturbed_Mech")
        yaml_loc.append(os.getcwd()+"/Perturbed_Mech/mechanism_"+str(i)+".yaml")

selected_params = []
activeParameters = []
#activeParameters.extend(unsrt_data[species].activeParameters)
for species in selected_species:
    activeParameters += [species+'_a1', species+'_a2', species+'_a3', species+'_a4', species+'_a5'] 
for params in activeParameters:
    selected_params.append(1)
#print("active parameters line 386 run.py\n\n\n",activeParameters)
#print("\n\n selected_params", selected_params)
selected_params_dict = {}
design_matrix_dict = {}
yaml_loc_dict = {}
for case in case_dir:
    yaml_loc_dict[case] = yaml_loc
    design_matrix_dict[case] = design_matrix
    selected_params_dict[case] = selected_params   

print("Pertubed mechnism files creted")


# ==============================================================================
# GENERATE MATRIX IF IT DOES NOT EXIST
# ==============================================================================
# Force reset navigation context back to script root directory
os.chdir(original_directory)
# ==============================================================================
# GENERATE TARGET SPECIES LIST FROM UNCERTAINTY DATA
# ==============================================================================
# Extracts unique species names while preserving the original order
target_species_list = list(dict.fromkeys(
    species.split(':')[0] for species in unsrt_data
))

print(f"Extracted target species list for thermo lookup: {target_species_list}")

# FIXED: Explicitly use the initial directory anchor
perturbed_folder = os.path.join(original_directory, "Perturbed_Mech")
output_csv = os.path.join(original_directory, "design_matrix_with_actual_param.csv")

if not os.path.exists(output_csv):
    print(f"File '{output_csv}' not found. Generating actual design matrix...")
    
    # Safely scan for YAML files now that we are reliably in the original directory context
    mech_files = sorted(
        glob.glob(os.path.join(perturbed_folder, "mechanism_*.yaml")),
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )
    
    print(f"Found {len(mech_files)} perturbed mechanisms. Extracting coefficients...")
    
    if len(mech_files) == 0:
        raise FileNotFoundError(f"No mechanism files found in absolute path: {perturbed_folder}")
        
    new_design_matrix = []

    for mech_path in mech_files:
        with open(mech_path, 'r') as f:
            mech_data = yaml.safe_load(f)
        
        species_thermo_map = {
            spec['name']: spec.get('thermo', {}) 
            for spec in mech_data.get('species', [])
        }
        
        mechanism_row = []
        
        for species_name in target_species_list:
            thermo_data = species_thermo_map.get(species_name, {})
            
            low_coeffs = [0.0] * 7
            high_coeffs = [0.0] * 7
            
            if thermo_data.get('model') == 'NASA7' and 'data' in thermo_data:
                try:
                    low_coeffs = [float(c) for c in thermo_data['data'][0]]
                    high_coeffs = [float(c) for c in thermo_data['data'][1]]
                except Exception:
                    pass 
            
            mechanism_row.extend(low_coeffs)
            mechanism_row.extend(high_coeffs)
            
        new_design_matrix.append(mechanism_row)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(new_design_matrix)
    print(f"Successfully created: {output_csv}")
else:
    print(f"File '{output_csv}' detected. Skipping generation and loading directly.")



##############################################################
##      Doing simulations using the design matrix           ## 
##      The goal is to test the design matrix                ##
##############################################################
if "Opt" not in os.listdir():
    os.mkdir("Opt")
    os.chdir("Opt")
    optDir = os.getcwd()
    os.mkdir("Data")
    os.chdir("Data")
    os.mkdir("Simulations")
    os.mkdir("ResponseSurface")
    os.chdir("..")
else:
    os.chdir("Opt")
    optDir = os.getcwd()

if os.path.isfile("progress") == False:
    FlameMaster_Execution_location = SSM.make_dir_in_parallel(yaml_loc_dict)
else:
    print("Progress file detected")
    progress = open(optDir+"/progress",'r').readlines()
    FlameMaster_Execution_location = []
    #with open(optDir+"/locations") as infile:
    #   for line in infile:
    #        FlameMaster_Execution_location.append(line)

#############################################
##      Extracting simulation results       ##
#############################################
temp_sim_opt = {}
for case in case_dir:    
    os.chdir("Data/Simulations")
    if "sim_data_case-"+str(case)+".lst" in os.listdir():
        file_name = "sim_data_case-"+str(case)+".lst"
        with open(file_name, "r") as file_:
            unique_lines = file_.readlines()        
        ETA = [float(line.split("\t")[1]) for line in unique_lines]
        temp_sim_opt[str(case)] = ETA
        os.chdir(optDir)   
    else:
        os.chdir(optDir)
        os.chdir("case-"+str(case))    
        data_sheet, failed_sim, ETA = data_management.generate_target_value_tables(FlameMaster_Execution_location, target_list, case, fuel, input_=optInputs)
        temp_sim_opt[str(case)] = ETA
        f = open('../Data/Simulations/sim_data_case-'+str(case)+'.lst','w').write(data_sheet)
        g = open('../Data/Simulations/failed_sim_data_case-'+str(case)+'.lst','w').write(failed_sim)
        os.chdir(optDir)

###############################################
##      Generating the response surface      ##
###############################################
print("SIMULATIONS ARE DONE. GENERATING RESPONSE SURFACE")





# ==============================================================================
# LOAD THE ACTUAL DESIGN MATRIX, VALIDATE SHAPE, AND REUSE PER CASE
# ==============================================================================


full_actual_matrix = np.loadtxt(output_csv, delimiter=",")
mech_sim_count = full_actual_matrix.shape[0]

for case in temp_sim_opt:
    if len(temp_sim_opt[case]) != mech_sim_count:
        raise ValueError(
            f"Row mismatch error for case {case}! The design matrix file has {mech_sim_count} rows, "
            f"but this case simulation yData has {len(temp_sim_opt[case])} rows."
        )

print("SIMULATIONS ARE DONE. GENERATING RESPONSE SURFACE")

# Move into the 'Opt' directory so ResponseSurface.py finds the 'Data/ResponseSurface/' folder
os.chdir(optDir)

ResponseSurfaces = {}
selected_PRS = {}

for case_index, case in enumerate(temp_sim_opt):
    xData = full_actual_matrix
    yData = np.asarray(temp_sim_opt[case]).flatten()
    print(f"\n📊 [Y-VALUES INSPECTION] Case Index {case_index} ({case}):")
    print(f"   Shape: {yData.shape}")
    print(f"   Min Value: {np.min(yData):.6e} | Max Value: {np.max(yData):.6e}")
    print(f"   Raw Values:\n{yData}\n")
    # ==========================================================================
    
    # ==========================================================================
    # TRAIN / TEST SPLIT AND MODEL FITTING
    # ==========================================================================
    xTrain, xTest, yTrain, yTest = train_test_split(
        xData, yData,
        random_state=104, 
        test_size=0.2,   
        shuffle=True
    )
    print(f"--> [Data Split Matrix] Case {case_index} ({case}): Training on {xTrain.shape[0]} samples | Testing on {xTest.shape[0]} samples (Total: {xTrain.shape[0] + xTest.shape[0]} rows)")
    
    Response = PRS.ResponseSurface(
        xTrain, yTrain, case, case_index, 
        prs_type=PRS_type, 
        selected_params=selected_params_dict[case_index]
    )
    Response.create_response_surface()
    
    if Response.generated == False:
        Response.test(xTest, yTest)
        Response.plot(case_index)
    else:
        Response.test(xTest, yTest)
        Response.plot(case_index)
        
    ResponseSurfaces[case_index] = Response
    print(f"--> [RAM Safe] Response Surface Model for Case Index {case_index} trained successfully.")

    # ==========================================================================
    # PROACTIVE OOM MEMORY RECOVERY LAYER
    # ==========================================================================
    # 1. Break strong tracking reference markers to large arrays inside this local iteration scope
    del xTrain, xTest, yTrain, yTest, xData, yData
    
    # 2. Force the interpreter engine to drop internal caches and clear out unreferenced memory immediately
    gc.collect()

# Safely return back out to your main execution folder root at the very end
os.chdir(original_directory)

print("RESPONSES PLOTS CREATED")



print("Entering Optimization Phase.")
# ==============================================================================
#                 AUTOMATED OPTIMIZATION PROCEDURE PIPELINE
# ==============================================================================

# 1. Absolute Path Environments Setup
response_surface_dir = os.path.join(original_directory, "Opt", "Data", "ResponseSurface")
nominal_thermo_txt = os.path.join(original_directory, "thermo_species_data.txt")
active_species = target_species_list  

# 2. Automatically detect failed PRS cases via selection files
rejected_prs_cases = []
total_original_targets = len(target_list)

for case_idx in range(total_original_targets):
    selection_file = os.path.join(response_surface_dir, f"selection_case-{case_idx}.csv")
    if os.path.exists(selection_file):
        with open(selection_file, 'r') as sf:
            selection_status = int(sf.read().strip())
        if selection_status == 0:
            rejected_prs_cases.append(case_idx)
    else:
        rejected_prs_cases.append(case_idx)

print(f"📋 Total Input Cases: {total_original_targets}")
print(f"❌ Identified {len(rejected_prs_cases)} failed PRS cases to exclude: {rejected_prs_cases}")

exp_vals = np.array([np.log(float(t.observed) * 10) for t in target_list])
design_matrix_csv_file = os.path.join(original_directory, "DesignMatrix.csv")

# ==============================================================================
# LAUNCH GENETIC ALGORITHM THROUGH OPTIMIZATION MODULE
# ==============================================================================
print("\nInitiating Genetic Algorithm Optimization via Module Engine...")



best_parameters, minimum_cost = ga_mod.run_optimization_process(
    prs_coefficients_folder=response_surface_dir,
    thermo_data_file_path=nominal_thermo_txt,
    target_species_list=active_species,
    full_experimental_values=exp_vals,          
    rejected_cases=rejected_prs_cases,          
    groups=groups,                        
    xml_uncertainty_file= unsrt_location,  
    baseline_group_rms=None, 
    total_case_count=len(exp_vals),          
    ngen=600,          
    pop_size=500,      
    n_workers=22,      
    seed=42,          
    resume_from_dir=None,
    design_matrix_path=design_matrix_csv_file  
)

print(f"\n🚀 GA Optimization Completed.")
print(f"Minimized Balanced Log-Space Metric Score: {minimum_cost:.6e}")

np.savetxt("optimized_zeta_parameters.csv", best_parameters, delimiter=",")
print("Saved optimized parameter vector to: optimized_zeta_parameters.csv")
