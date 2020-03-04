#! /usr/bin/env python
"""Phase contraint overlap tool. This tool calculates the minimum and maximum phase of
the primary or secondary transit (by default, primary) based on parameters provided by the user.

Authors:
    Catherine Martlin, 2018
    Mees Fix, 2018
    Nestor Espinoza, 2020

Usage:

  calculate_constraint <target_name> [--t_start=<t0>] [--period=<p>] [--obs_duration=<obs_dur>] [--transit_duration=<trans_dur>] [--window_size=<win_size>] [--secondary] [--eccentricity=<ecc>] [--omega=<omega>] [--inclination=<inc>] [--winn_approx]
  
Arguments:
  <target_name>                     Name of target
Options:
  -h --help                         Show this screen.
  --version                         Show version.
  --t_start=<t0>                    The starting time of the transit in BJD or HJD.
  --period=<p>                      The period of the transit in days.
  --obs_duration=<obs_dur>          The duration of the observation in hours.
  --transit_duration=<trans_dur>    The duration of the transit in hours.
  --window_size=<win_size>          The window size of the transit in hours [default: 1.0]
  --secondary                       If active, calculate phases for secondary eclipses (user needs to supply eccentricity, omega and inclination).
  --eccentricity=<ecc>              The eccentricity of the orbit (needed for secondary eclipse constraints).
  --omega=<omega>                   The argument of periastron passage (needed for secondary eclipse constraints).
  --inclination=<inc>               The inclination of the orbit (needed for secondary eclipse constraints).
  --winn_approx                     If active, instead of running the whole Kepler equation calculation, time of secondary eclipse is calculated using eq. (6) in Winn (2010; https://arxiv.org/abs/1001.2010v5)
"""

import math
import os

import argparse
from docopt import docopt
import numpy as np
import requests
import urllib

from scipy import optimize
#from exoctk.utils import get_target_data

def calculate_phase(period, obsDur, winSize, t0 = None, ecc = None, omega = None, inc = None, secondary = False, winn_approx = False):
    ''' Function to calculate the min and max phase. 

        Parameters
        ----------
        period : float
            The period of the transit in days.
        obsdur : float
            The duration of the observation in hours.
        winSize : float
            The window size of transit in hours. Default is 1 hour.
        t0 : float
            The time of (primary) transit center (only needed for secondary eclipses).
        ecc : float
            The eccentricity of the orbit (only needed for secondary eclipses).
        omega : float
            The argument of periastron passage, in degrees (only needed for secondary eclipses).
        inc : float
            The inclination of the orbit, in degrees (only needed for secondary eclipses).
        secondary : boolean
            If True, calculation will be done for secondary eclipses.
        winn_approx : boolean
            If True, secondary eclipse calculation will use the Winn (2010) approximation to estimate time of secondary eclipse --- (only valid for not very eccentric and inclined orbits).

        Returns
        -------
        minphase : float
            The minimum phase constraint.
        maxphase : float
            The maximum phase constraint. '''

    if not secondary:
        minphase = 1.0 - ((obsDur + winSize)/2.0/24/period)
        maxphase = 1.0 - ((obsDur - winSize)/2.0/24/period)
    else:
        deg_to_rad = (np.pi/180.)
        # Calculate time of secondary eclipse:
        tsec = calculate_tsec(period, ecc, omega*deg_to_rad, inc*deg_to_rad, t0 = t0, winn_approximation = winn_approx)
        # Calculate difference in phase-space between primary and secondary eclipse (note calculate_tsec ensures tsec is 
        # *the next* secondary eclipse after t0):
        phase_diff = (tsec - t0)/period
        # Estimate minphase and maxphase centered around this phase (thinking here is that, e.g., if phase_diff is 0.3 
        # then eclipse happens at 0.3 after 1 (being the latter by definition the time of primary eclipse --- i.e., transit). 
        # Because phase runs from 0 to 1, this implies eclipse happens at phase 0.3):
        minphase = phase_diff - ((obsDur + winSize)/2.0/24/period)
        maxphase = phase_diff - ((obsDur - winSize)/2.0/24/period)
        # Wrap the phases around 0 and 1 in case limits blow in the previous calculation (unlikely, but user might be doing 
        # something crazy or orbit could be extremely weird such that this can reasonably happen in the future). Note this 
        # assumes -1 < minphase,maxphase < 2:
        if minphase < 0:
            minphase = 1. + minphase
        if maxphase > 1:
            maxphase = maxphase - 1.
    return minphase, maxphase

def calculate_obsDur(transitDur):
    ''' Function to calculate the min and max phase.

        Parameters
        ----------
        transitDur : float
            The duration of the transit in hours.

        Returns
        -------
        obsdur : float
            The duration of the observation in hours. '''

    obsDur = np.max((6, 3*transitDur+1))

    return obsDur

def drsky_2prime(x, ecc, omega, inc):
    ''' Second derivative of function drsky

    This is the second derivative with respect to f of the drsky function. 

    Parameters
    ----------

    x : float
      True anomaly
    ecc : float
      Eccentricity of the orbit
    omega : float
      Argument of periastron passage (in radians)
    inc : float
      Inclination of the orbit (in radians) 
      
    Returns
    -------
    drsky_2prime : float
      Function evaluated at x, ecc, omega, inc'''

    sq_sini = np.sin(inc)**2
    sin_o_p_f = np.sin(x+omega)
    cos_o_p_f = np.cos(x+omega)
    ecosf = ecc*np.cos(x)
    esinf = ecc*np.sin(x)

    f1 = esinf - esinf*sq_sini*(sin_o_p_f**2)
    f2 = -sq_sini*(ecosf + 4.)*(sin_o_p_f*cos_o_p_f)

    return f1+f2

def drsky_prime(x, ecc, omega, inc):
    ''' Derivative of function drsky

    This is the first derivative with respect to f of the drsky function. 

    Parameters
    ----------

    x : float
      True anomaly
    ecc : float
      Eccentricity of the orbit
    omega : float
      Argument of periastron passage (in radians)
    inc : float
      Inclination of the orbit (in radians) 
      
    Returns
    -------
    drsky_prime : float
      Function evaluated at x, ecc, omega, inc'''

    sq_sini = np.sin(inc)**2
    sin_o_p_f = np.sin(x+omega)
    cos_o_p_f = np.cos(x+omega)
    ecosf = ecc*np.cos(x)
    esinf = ecc*np.sin(x)

    f1 = (cos_o_p_f**2 - sin_o_p_f**2)*(sq_sini)*(1. + ecosf)
    f2 = -ecosf*(1 - (sin_o_p_f**2)*(sq_sini))
    f3 = esinf*sin_o_p_f*cos_o_p_f*sq_sini
    
    return f1+f2+f3

def drsky(x, ecc, omega, inc):
    ''' Function whose roots we wish to find to obtain time of secondary (and primary) eclipse(s)

    When one takes the derivative of equation (5) in Winn (2010; https://arxiv.org/abs/1001.2010v5), and equates that to zero (to find the 
    minimum/maximum of said function), one gets to an equation of the form g(x) = 0. This function (drsky) is g(x), where x is the true 
    anomaly.

    Parameters
    ----------

    x : float
      True anomaly
    ecc : float
      Eccentricity of the orbit
    omega : float
      Argument of periastron passage (in radians)
    inc : float
      Inclination of the orbit (in radians) 
      
    Returns
    -------
    drsky : float
      Function evaluated at x, ecc, omega, inc '''

    sq_sini = np.sin(inc)**2
    sin_o_p_f = np.sin(x+omega)
    cos_o_p_f = np.cos(x+omega)

    f1 = sin_o_p_f*cos_o_p_f*sq_sini*(1. + ecc*np.cos(x))
    f2 = ecc*np.sin(x)*(1. - sin_o_p_f**2 * sq_sini)
    return f1 - f2

def getE(f,ecc):
    """ Function that returns the eccentric anomaly

    Note normally this is defined in terms of cosines (see, e.g., Section 2.4 in Murray and Dermott), but numerically 
    this is troublesome because the arccosine doesn't handle negative numbers by definition (equation 2.43). That's why 
    the arctan version is better as signs are preserved (derivation is also in the same section, equation 2.46).

    Parameters
    ----------

    f : float
      True anomaly

    ecc : float
      Eccentricity

    Returns
    -------
    E : float
      Eccentric anomaly """

    return 2. * np.arctan(np.sqrt((1.-ecc)/(1.+ecc))*np.tan(f/2.))  

def getM(E, ecc):
    """ Function that returns the mean anomaly using Kepler's equation

    Parameters
    ----------

    E : float
      Eccentric anomaly

    ecc: float
      Eccentricity

    Returns
    -------
    M : float
      Mean anomaly """

    return E - ecc*np.sin(E)

def getLTT(a, c, ecc, omega, inc, f):
    ''' Function that calculates the Light Travel Time (LTT) for eclipses and transit

        This function returns the light travel time of an eclipse or transit given the orbital parameters, 
        the semi-major axis of the orbit and the speed of light. Consistent units must be used for the latter 
        two parameters. The returned time is in the units of time given by the speed of light input parameter.

        Parameters
        ----------
        a : float
            Semi-major axis of the orbit. Can be in any unit, as long as it is consistent with c.
        c : float
            Speed of light. Can be in any unit, as long as it is consistent with a. The time-unit for the speed of light 
            will define the returned time-unit for the light-travel time.
        ecc : float
            Eccentricity of the orbit.
        omega : float
            Argument of periastron passage (in radians).
        inc : string
            Inclination of the orbit (in radians)
        f : float
            True anomaly at the time of transit and/or eclipse (in radians).
        
        Returns
        -------
        ltt : float
            The light travel time, defined here as the time it takes for a photon to go from the planet at transit/eclipse to the plane in the sky where the star is located.

    '''
    # Distance from star to the planet in stellar reference system:
    r = a*( 1. - ecc**2)/(1. + ecc*np.cos(f))
    # Projected distance from planet to the plane the star is on the sky ('light travel distance'):
    ltd = r * np.sin(omega + f)*np.sin(inc)
    # Return distance divided by c to get LTT:
    return ltd/c

def calculate_tsec(period, ecc, omega, inc, t0 = None, tperi = None, winn_approximation = False):
    ''' Function to calculate the time of secondary eclipse. 
      
        This uses Halley's method (Newton-Raphson, but using second derivatives) to first find the true anomaly (f) at which secondary eclipse occurs, 
        then uses this to get the eccentric anomaly (E) at secondary eclipse, which gives the mean anomaly (M) at secondary 
        eclipse using Kepler's equation. This finally leads to the time of secondary eclipse using the definition of the mean 
        anomaly (M = n*(t - tau) --- here tau is the time of pericenter passage, n = 2*pi/period the mean motion).

        Time inputs can be either the time of periastron passage directly or the time of transit center. If the latter, the 
        true anomaly for primary transit will be calculated using Halley's method as well, and this will be used to get the 
        time of periastron passage.
        
        Parameters
        ----------
        period : float
            The period of the transit in days. 
        ecc : float
            Eccentricity of the orbit
        omega : float
            Argument of periastron passage (in radians)
        inc : string
            Inclination of the orbit (in radians)

        t0 : float
            The transit time in BJD or HJD (will be used to get time of periastron passage).

        tperi : float
            The time of periastron passage in BJD or HJD (needed if t0 is not supplied).

        winn_approximation : boolean
            If True, the approximation in Winn (2010) is used --- (only valid for not very eccentric and inclined orbits).

        Returns
        -------
        tsec : float
            The time of secondary eclipse '''
    # Check user is not playing trick on us:
    if period<0:
        raise Exception('Period cannot be a negative number.')
    if ecc<0 or ecc>1:
        raise Exception('Eccentricity (e) is out of bounds (0 < e < 1).')

    # Use true anomaly approximation given in Winn (2010) as starting point:
    f_occ_0 = (-0.5*np.pi) - omega
    if not winn_approximation:
        f_occ = optimize.newton(drsky, f_occ_0, fprime = drsky_prime, fprime2 = drsky_2prime, args = (ecc, omega, inc,))
    else:
        f_occ = f_occ_0
    # Define the mean motion, n:
    n = 2.*np.pi/period

    # If time of transit center is given, use it to calculate the time of periastron passage. If no time of periastron 
    # or time-of-transit center given, raise error:
    if tperi is None:
        # For this, find true anomaly during transit. Use Winn (2010) as starting point:
        f_tra_0 = (np.pi/2.) - omega
        if not winn_approximation:
            f_tra = optimize.newton(drsky, f_tra_0, fprime = drsky_prime, fprime2 = drsky_2prime, args = (ecc, omega, inc,))
        else:
            f_tra = f_tra_0
        # Get eccentric anomaly during transit:
        E = getE(f_tra, ecc)

        # Get mean anomaly during transit:
        M = getM(E, ecc)

        # Get time of periastron passage from mean anomaly definition:
        tperi = t0 - (M/n)

    elif (tperi is None) and (t0 is None):
        raise ValueError('The time of periastron passage or time-of-transit center has to be supplied for the calculation to work.')

    # Get eccentric anomaly:
    E = getE(f_occ, ecc)

    # Get mean anomaly during secondary eclipse:
    M = getM(E, ecc)

    # Get the time of secondary eclipse using the definition of the mean anomaly:
    tsec = (M/n) + tperi

    # Note returned time-of-secondary eclipse is the closest to the time of periastron passage and/or time-of-transit center. Check that 
    # the returned tsec is the *next* tsec to the time of periastron or t0 (i.e., the closest *future* tsec):
    if t0 is not None:
        tref = t0
    else:
        tref = tperi
    if tref > tsec:
        while True:
            tsec += period
            if tsec > tref:
                break
    return tsec

def phase_overlap_constraint(target_name, period=None, t0=None, obs_duration=None, transit_dur=None, window_size=None, secondary = False, ecc = 0., omega = 90., inc = 90., winn_approx = False):
    ''' The main function to calculate the phase overlap constraints.
        We will update to allow a user to just plug in the target_name
        and get the other variables.

        Parameters
        ----------
        period : float
            The period of the transit in days.
        t0 : float
            The start time in BJD or HJD.
        obs_duration : float
            The duration of the observation in hours.
        transit_dur : float
            The duration of the transit/eclipse in hours.
        winSize : float
            The window size of transit in hours. Default is 1 hour.
        target_name : string
            The name of the target transit. 
        secondary : boolean
            If True, phase constraint will be the one for the secondary eclipse. Default is primary (i.e., transits).
        ecc : float
            Eccentricity of the orbit. Needed only if secondary is True.
        omega : float
            Argument of periastron of the orbit in degrees. Needed only if secondary is True.
        inc : float
            Inclination of the orbit in degrees. Needed only if secondary is true.
        winn_approx : boolean
            If True, instead of running the whole Kepler equation calculation, time of secondary eclipse is calculated using eq. (6) in Winn (2010; https://arxiv.org/abs/1001.2010v5)

        Returns
        -------
        minphase : float
            The minimum phase constraint.
        maxphase : float
            The maximum phase constraint. '''

    if obs_duration == None:
        if period == None:
            data = get_target_data(target_name)
            
            period = data['orbital_period']
            t0 = data['transit_time']
            # If transit/eclipse duration not supplied by the user, extract it from the get_target_data function:
            if transit_dur is None:
                transit_dur = data['transit_duration']
                if secondary:
                # Factor from equation (16) in Winn (2010). This is, of course, an approximation.
                # TODO: Implement equation (13) in the same paper. Needs optimization plus numerical integration.
                    factor = np.sqrt(1. - ecc**2)/(1. - ecc*np.sin(omega*(np.pi/180.)))
                    transit_dur = transit_dur*factor

    if obs_duration == None:
        data, _  = get_target_data(target_name)
        transit_dur = data['transit_duration'] *24.0
        obs_duration = calculate_obsDur(transit_dur)

    minphase, maxphase = calculate_phase(period, obs_duration, window_size, t0 = t0, ecc = ecc, omega = omega, inc = inc, secondary = secondary, winn_approx = winn_approx)
    
    # Is this the return that we want? Do we need to use t0 for something? 
    # NE: Not needed, because by defaut APT assumes phase = 1 is where the transit happens.
    print('MINIMUM PHASE: {}, MAXIMUM PHASE: {}'.format(minphase, maxphase))
    return minphase, maxphase

# Need to make entry point for this!
if __name__ == '__main__':
    args = docopt(__doc__, version='0.1')

    # First, save the secondary flag (which is a boolean):
    secondary = args['--secondary']
    # Save the winn_approx flag as well:
    winn_approx = args['--winn_approx']
    # Convert all entries from strs to floats (this will convert the --secondary arg above, but that's OK):
    for k,v in args.items():
        try:
            args[k] = float(v)
        except (ValueError, TypeError):
            # Handles None and char strings.
            continue

    # Check that if the user wants to get secondary eclipse constraints, eccentricity, omega and inclination has also been 
    # supplied:
    if secondary and ((args['--eccentricity'] is None) or (args['--inclination'] is None) or (args['--omega'] is None)):
        raise Exception('If --secondary, eccentricity, omega and inclination have to be supplied (e.g., --eccentricity = 0.1 --omega = 30.1 --inclination=89.2)')
    
    phase_overlap_constraint(args['<target_name>'], period = args['--period'], 
                             t0 = args['--t_start'], obs_duration = args['--obs_duration'],
                             transit_dur = args['--transit_duration'],
                             window_size = args['--window_size'], secondary = secondary,
                             ecc = args['--eccentricity'], omega = args['--omega'],
                             inc = args['--inclination'], winn_approx = winn_approx)
