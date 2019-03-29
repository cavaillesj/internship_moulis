# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:08:06 2019

@author: jerome
"""



# =============================================================================
# param 1
# =============================================================================


"""
A = [0.2, 0.4, 0.6]
M = [0.5, 1., 1.5]
Alpha = [2, 4, 7, 10]
Beta = [2, 4, 7, 10]
Strength = [0.01, 0.05, 0.1]
"""



# =============================================================================
#   Second try
# =============================================================================


"""
DOSSIER = "plot/measures/coupled/param2/"

A = [0.2, 0.4]
M = [5., 7., 9.]
Strength = [0.001, 0.005, 0.01]
Alpha = [2, 7, 15, 28]
Beta = [5, 15, 50, 200]
"""


# =============================================================================
#   Third try
# =============================================================================

"""
DOSSIER = "plot/measures/coupled/param3/"

A = [0.1, 0.3, 0.5]
M = [4., 7., 10.]
Strength = [0.005, 0.01, 0.05, 0.1]
Alpha = [4, 15, 28]
Beta = [4, 15, 50, 200]


Variability = ["Variability_always", "Variability_until", "Variability_only", "Variability_10", "Variability_half", "Speed_collapse", "Viability"]
"""

# =============================================================================
#   Object save
# =============================================================================

"""
DOSSIER = "plot/measures/coupled/param4/"

A = [0.1, 0.3, 0.5]
M = [4., 7., 10.]
Strength = [0.005, 0.01, 0.05, 0.1]
Alpha = [4, 15, 28]
Beta = [4, 15, 50, 200]

Freq = [0.005, 0.1, 0.3, 0.5, 1., 3., 5., 9.]

#Variability = ["Variability_always", "Variability_until", "Variability_only", "Variability_10", "Variability_half", "Speed_collapse", "Viability"]
"""

# =============================================================================
# =============================================================================

"""
DOSSIER = "plot/measures/coupled/param5/"

A = [0.1, 0.3, 0.5]################## surrpimer 0.1
M = [4., 7., 10.]
Strength = [0.005, 0.01, 0.05, 0.1]
Alpha = [4, 15, 28]
Beta = [4, 15, 50, 200]

Freq = [0.005, 0.1, 0.3, 0.5, 1., 3., 5., 9.]

Variability = ["Variability_always", "Variability_until", "Variability_only", "Variability_10", "Variability_half", "Speed_collapse", "Viability", "Ratio"]
"""

# =============================================================================
# =============================================================================

"""
DOSSIER = "plot/measures/coupled/param6/"

A = [0.3, 0.5]################## surrpimer 0.1
M = [4., 7., 10.]
Strength = [0.001, 0.005, 0.01, 0.05]
Alpha = [4, 15, 28]
Beta = [10, 50, 200]

Freq = [0.005, 0.1, 0.3, 0.5, 1., 3., 5., 9.]

Variability = ["Variability_always", "Variability_until", "Variability_only", "Variability_10", "Variability_half", "Speed_collapse", "Viability", "Ratio"]
"""


# =============================================================================
# correction measures andnew measure : point
# =============================================================================

"""
DOSSIER = "plot/measures/coupled/param7/"

A = [0.3, 0.5]################## surrpimer 0.1
M = [4., 7., 10.]
Strength = [0.001, 0.005, 0.01, 0.05]
Alpha = [4, 15, 28]
Beta = [10, 50, 200]

Freq = [0.005, 0.1, 0.3, 0.5, 1., 3., 5., 9.]

Variability = ["Variability_always", "Variability_until", "Variability_only", "Variability_10", "Variability_half", "Speed_collapse", "Viability", "Ratio", "Point"]
"""

# =============================================================================
# TEST
# =============================================================================

"""
DOSSIER = "plot/measures/coupled/test/"

A = [0.5]################## surrpimer 0.1
M = [10.]
Strength = [0.005]
Alpha = [15]
Beta = [50]

Freq = [0.005, 0.1, 0.3, 0.5, 1., 3., 5., 9.]

Variability = ["Variability_always", "Variability_until", "Variability_only", "Variability_10", "Variability_half", "Speed_collapse", "Viability", "Ratio", "Point"]
"""



# =============================================================================
# param 8 : long time study (100 instead to 10)
# =============================================================================

"""
DOSSIER = "plot/measures/coupled/param8/"

A = [0.3, 0.5] 
M = [4., 7., 10.]
Strength = [0.001, 0.005, 0.01, 0.05]
Alpha = [4, 15, 28]
Beta = [10, 50, 200]

Freq = [0.005, 0.1, 0.3, 0.5, 1., 3., 5., 9.]

Variability = ["Variability_always", "Variability_until", "Variability_only", "Variability_10", "Variability_half", "Speed_collapse", "Viability", "Ratio", "Point"]
"""



# =============================================================================
# param 9 : long time study (100 instead to 10) with range parameter adapted
# =============================================================================

"""
DOSSIER = "plot/measures/coupled/param9/"

A = [0.3, 0.5] 
M = [3., 6., 9.]
Strength = [0.001, 0.002, 0.005, 0.008, 0.01]
Alpha = [8, 15, 25]
Beta = [30, 70, 200]
Freq = [0.005, 0.02, 0.1, 0.3, 0.5, 1., 4., 9.]

Variability = ["Variability_always", "Variability_until", "Variability_only", "Variability_10", "Variability_half", "Speed_collapse", "Viability", "Ratio", "Point"]
"""



# =============================================================================
# param 10 : final time as a parameter
# =============================================================================

DOSSIER = "plot/measures/coupled/param10/"

A = [0.3, 0.5] 
M = [2., 4., 7., 10.] # [2., 5., 10.]
Strength = [0.001, 0.002, 0.005, 0.008, 0.01]
Alpha = [8, 15, 25]
Beta = [5, 20, 40, 100]
Freq = [0.005, 0.02, 0.1, 0.3, 0.6, 1., 2., 4., 6., 9.]
FinalTime = [10, 40, 100]

Variability = ["Variability_always", "Variability_until", "Variability_only", "Variability_10", "Variability_half", "Speed_collapse", "Viability", "Ratio", "Point"]


