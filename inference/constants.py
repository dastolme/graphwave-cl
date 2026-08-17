TOTAL_NUM_PIXELS = 2304

# RECO_URL = "https://s3.cr.cnaf.infn.it:7480/cygno:cygno-analysis/RECO/Run5_Saladin/"
# RECO_URL = "https://s3.cr.cnaf.infn.it:7480/cygno:cygno-analysis/users/dastolme/ML_test/"
# RECO_URL = "/raid/home/dastolfo/GraphWave-CL/AmBe_LG/"
RECO_URL = "/raid/home/dastolfo/GraphWave-CL/Run5_LG/"
# RECO_URL = "https://s3.cr.cnaf.infn.it:7480/cygno:cygno-analysis/RECO/Run4_lowgain_standardPar/"
# RECO_URL = "https://s3.cr.cnaf.infn.it:7480/cygno:cygno-analysis/RECO/Run4_Saladin/"

CMOS_RECO_VARIABLES = ['run', 'event', 'nSc', 'sc_integral', 'sc_xmean', 'sc_ymean', 'sc_rms']
PMT_RECO_VARIABLES = ['pmt_wf_event', 'pmt_wf_trigger', 'pmt_wf_sampling', 'pmt_wf_insideGE']

CMOS_REDUCED_RECO_VARIABLES = ['event', 'nSc']

ENERGY_THRESHOLD = 850
RADIUS_CUT = 800
RMS_CUT = 6

FAST_DAQ = 1024
FAST_DAQ_IDXS = [0,1,2,3]
SLOW_DAQ = 4000
SLOW_DAQ_CUT = 1500
SLOW_DAQ_IDXS = [4,5,6,7]

INSIDE_GE = 0