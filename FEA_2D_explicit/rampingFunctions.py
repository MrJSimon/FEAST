##############################################################################
##
## Author:      
##
## Description: The script contains available ramping function to ease
##              explicit analysis
##
##############################################################################


# Load in modulus
#import numpy as np



def logistic_ramp(yval = float, 
                  nstep = float, nsteps = float,
                  xcenter = 0.1, slope = 2.5):

    """
    Employs a logistic ramp to the targeted yval

    Parameters
    ----------
    yval  : float
    
    nstep : float
    
    nsteps : float

    xcenter : float 
    inflection point on the logistic ramp
       
    slope : float 
    
    Returns
    -------
    yval * logistic_ramp : float
    modified yval
    
    """
    
    # Normalize step
    norm_step = nstep/nsteps
    
    # Compute logistic-ramp up
    nom = (1.0 - norm_step) * xcenter
    deo = (1.0 - xcenter) * norm_step
    
    # # Compute logistic-ramp up
    if nstep != 0:
        
        # Compute nominator and denominator to be used
        # in logistic function
        nom = (1.0 - norm_step) * xcenter
        deo = (1.0 - xcenter) * norm_step
        
        # Compute amplitude
        Amp =  1.0 / (1.0 + (nom/deo)**slope)
            
    else:
        Amp = 10e-19
        
    # Apply aplitude/ramp to yval
    return yval * Amp
    
## Piecewize linear ramping function to be implemented
# ramp_end = 0.01 * total_time
# if current_time < ramp_end:
    # v_scale = current_time / ramp_end
# else:
    # v_scale = 1.0
# current_v_target = np.copy(velocities)
# current_v_target[:,-1] = current_v_target[:,-1]*v_scale

#print(velocities)
#print(current_v_target)
