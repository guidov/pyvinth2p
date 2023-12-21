import numpy as np
cimport cython
cimport numpy as np
from libc.math cimport log

DTYPE32 = np.float32
DTYPE64 = np.float64

ctypedef np.float32_t DTYPE32_t
ctypedef np.float64_t DTYPE64_t

cdef double ll(double x):
    return log(log(x + 2.72))

@cython.boundscheck(False)
@cython.wraparound(False)
def vinth2p_ecmwf_fast_4d(np.ndarray[DTYPE32_t, ndim=4] datai,
                                                np.ndarray[DTYPE64_t, ndim=1] hbcoefa,
                                                np.ndarray[DTYPE64_t, ndim=1] hbcoefb,
                                                double p0,
                                                np.ndarray[DTYPE32_t, ndim=3] psfc,
                                                np.ndarray[DTYPE64_t, ndim=1] plevo,
                                                np.ndarray[DTYPE32_t, ndim=3] phis,
                                                np.ndarray[DTYPE32_t, ndim=3] tbot,
                                                int var,
                                                int intyp,
                                                int kxtrp,
                                                double spval,
                                                np.ndarray[DTYPE32_t, ndim=4] datao):
    """
    Interpolates from hybrid grid to pressure grid using the ECMWF formulation
    as defined by the vinth2p_ecmwf function in NCAR NCL language.
    Args:
            datai : input data
            hbcoefa : A coeffs for transformation from hybrid to pressure grid
            hbcoefb : B coeffs for transformation from hybrid to pressure grid
            p0 : reference pressure (units: hPa)
            psfc : surface pressure (units: hPa)
            plevo : output pressure levels (units: hPa)
            phis : surface geopotential height (units: m2/s2)
            tbot : temperature at level closest to ground (units: K). Only used
                   if interpolating geopotential height
            var : one of: 1, 2 or 3 to indicate whether interpolating temperature, geopotential height
                  or some other variable
            intyp : specify the interpolation type. Either of: "1" for linear, "2"
                    for log or "3" for log-log interpolation
            kxtrp : integer, 0 (false) or 1 (true) for whether to extrapolate when pressure level is
                    outside the range of psfc

    Returns:
            3D array of shape (nplevo, nlat, nlon) where nplevo is the length of
            the input variable plevo
    """
    cdef int i, j, k, kp, it
    cdef double psfcmb, tstar, hgt, alnp, t0, tplat, alph, tprime0
    cdef float pk

    # Some pre-checks:::

    if var not in [1, 2, 3]:
        raise ValueError("var has to be 1, 2 or 3")


    if intyp not in [1, 2, 3]:
        raise ValueError("intyp has to be 1, 2 or 3")

    cdef int ntime = datai.shape[0]
    cdef int nlevi = datai.shape[1]
    cdef int nlat = datai.shape[2]
    cdef int nlon = datai.shape[3]

    cdef int nlevo = len(plevo)
    assert(nlevo >= 1)

    assert(len(hbcoefa) == len(hbcoefb) == nlevi)
    assert(psfc.shape[0] == ntime)
    assert(psfc.shape[1] == nlat)
    assert(psfc.shape[2] == nlon)

    # Defining some constants
    cdef double rd = 287.04
    cdef double ginv = 1. / 9.80616   # g inverse
    cdef double alpha = 0.0065 * rd * ginv

    # for numba we have output on the end of the function argument
    #cdef np.ndarray[DTYPE32_t, ndim=4] datao = np.zeros((ntime, nlevo, nlat, nlon), dtype=DTYPE32)  # Output array

    cdef np.ndarray[DTYPE64_t, ndim=1] plevi = np.zeros(nlevi, dtype=DTYPE64)

    for it in range(ntime):
        for i in range(nlat):
            for j in range(nlon):

                # vertical pressure levels for this column (hPa)
                plevi = hbcoefa * p0 + hbcoefb * psfc[it, i, j]

                for k in range(nlevo):  # Iterate over all user requested output levels
                    pk = plevo[k]
                    if pk <= plevi[0]:
                        # If branch for model top. If the pressure is lower than the
                        # top of the model
                        kp = 0
                    elif pk > plevi[nlevi - 1]:
                        # If branch for level below lowest hybrid level. If the pressure is
                        # higher than the lowest level
                        if kxtrp == 0:
                            datao[k, i, j] = spval
                            continue
                        elif var == 1:   # Interpolating the temperature
                            psfcmb = psfc[it, i, j]
                            tstar = datai[it, nlevi - 1, i, j] * (1. + alpha * (psfcmb / plevi[nlevi - 1] - 1.))
                            hgt = phis[it, i, j] * ginv

                            if hgt < 2000:
                                alnp = alpha * np.log(pk / psfcmb)
                            else:
                                t0 = tstar + 0.0065 * hgt
                                tplat = min(t0, 298.0)

                                if hgt <= 2500:
                                    tprime0 = 0.002 * ((2500 - hgt) * t0 + (hgt - 2000) * tplat)
                                else:
                                    tprime0 = tplat

                                if tprime0 < tstar:
                                    alnp = 0.0
                                else:
                                    alnp = rd * (tprime0 - tstar) / phis[it, i, j] * np.log(pk / psfcmb)

                            datao[it, k, i, j] = tstar * (1. + alnp + (0.5 * alnp**2) + (1.0 / 6.0 * (alnp**3)))
                            continue
                        elif var == 2:
                            psfcmb = psfc[it, i, j]
                            hgt = phis[it, i, j] * ginv
                            tstar = tbot[it, i, j] * (1. + alpha * (psfcmb / plevi[nlevi - 1] - 1.))
                            t0 = tstar + 0.0065 * hgt

                            if (tstar <= 290.5) and (t0 > 290.5):
                                alph = rd / phis[it, i, j] * (290.5 - tstar)
                            elif (tstar > 290.5) and (t0 > 290.5):
                                alph = 0
                                tstar = 0.5 * (290.5 + tstar)
                            else:
                                alph = alpha

                            if tstar < 255.:
                                tstar = 0.5 * (tstar + 255.)

                            alnp = alph * np.log(pk / psfcmb)
                            datao[it, k, i, j] = hgt - rd * tstar * ginv * np.log(pk / psfcmb) * (1. + 0.5 * alnp + (1. / 6. * (alnp**2)))
                            continue
                        else:
                            # Use the lowest sigma layer
                            datao[it, k, i, j] = datai[it, nlevi - 1, i, j]
                            continue

                    elif pk >= plevi[nlevi - 2]:
                        # If branch to check whether the output level in between 2
                        # lowest hybrid levels
                        kp = nlevi - 2
                    else:
                        # if branch for model interior. Find bracketing levels
                        kp = 0
                        while kp < nlevi:
                            if pk <= plevi[kp + 1]:
                                break
                            kp += 1

                    if intyp == 1:
                        datao[it, k, i, j] = datai[it, kp, i, j] + (datai[it, kp + 1, i, j] - datai[it, kp, i, j]) \
                                       * (pk - plevi[kp]) / (plevi[kp + 1] - plevi[kp])
                    elif intyp == 2:
                        datao[it, k, i, j] = datai[it, kp, i, j] + (datai[it, kp + 1, i, j] - datai[it, kp, i, j]) \
                                       * np.log(pk / plevi[kp]) / np.log(plevi[kp + 1] / plevi[kp])
                    elif intyp == 3:
                        datao[it, k, i, j] = datai[it, kp, i, j] + (datai[it, kp + 1, i, j] - datai[it, kp, i, j]) \
                                       * (ll(pk) - ll(plevi[kp])) / (ll(plevi[kp + 1]) - ll(plevi[kp]))


