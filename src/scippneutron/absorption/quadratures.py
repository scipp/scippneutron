"""Quadrature rules for computing integrals on the unit disk.

For each dictionary ``{x: ..., y: ..., weights: ...}`` in this module:
:math:`\\int_{||(x, y)|| < 1} f(x, y) dxdy \\approx \\sum_i weights[i] f(x[i], y[i])`

In practice the functions we integrate in the absorption module often have large values
and steep gradients close to the edge of the disk. Therefore it was found beneficial to
use a quadrature rule with more points close to the edge, such as disk256_cheb.

In most cases the moderately large disk55 quadrature seems to be accurate enough.
The cheapest quadrature disk12 should probably only be used for testing and in
simple cases where accuracy is less important and speed is essential.
"""

# https://mathsfromnothing.au/circle-quadrature-rules/?i=1
disk12 = {
    'weights': [
        0.390106733337761,
        0.384170402095294,
        0.384170402095296,
        0.39010673333776,
        0.163453148423726,
        0.232710566932579,
        0.232710566932578,
        0.163453148423725,
        0.232710566932577,
        0.232710566932577,
        0.16764490907296,
        0.167644909072959,
    ],
    'x': [
        -0.409866456399588,
        -0.193667113928664,
        0.193667113928663,
        0.40986645639959,
        0.382490518597346,
        -0.299340742790836,
        0.299340742790836,
        -0.382490518597346,
        -0.812646983446956,
        0.812646983446956,
        -0.825746255906219,
        0.825746255906219,
    ],
    'y': [
        0.189199039648859,
        -0.41954575379659,
        0.419545753796586,
        -0.189199039648863,
        0.82859846305183,
        0.812646983446954,
        -0.812646983446955,
        -0.828598463051831,
        -0.299340742790836,
        0.299340742790836,
        0.381173907187935,
        -0.381173907187935,
    ],
}
# https://www.sciencedirect.com/science/article/pii/S0377042721006324
# File T17_6segment.txt
disk55 = {
    'x': [
        0.0,
        0.36377542,
        0.52279286,
        0.67804703,
        0.77618651,
        0.63052358,
        0.89264378,
        0.87776477,
        0.69180144,
        0.97882448,
        0.18188771,
        0.00334963,
        0.3454346,
        0.1386123,
        -0.16849766,
        0.41653865,
        0.09034867,
        -0.24553628,
        0.41320962,
        -0.18188771,
        -0.51944322,
        -0.33261243,
        -0.6375742,
        -0.79902124,
        -0.47610513,
        -0.78741609,
        -0.93733772,
        -0.56561485,
        -0.36377542,
        -0.52279286,
        -0.67804703,
        -0.77618651,
        -0.63052358,
        -0.89264378,
        -0.87776477,
        -0.69180144,
        -0.97882448,
        -0.18188771,
        -0.00334963,
        -0.3454346,
        -0.1386123,
        0.16849766,
        -0.41653865,
        -0.09034867,
        0.24553628,
        -0.41320962,
        0.18188771,
        0.51944322,
        0.33261243,
        0.6375742,
        0.79902124,
        0.47610513,
        0.78741609,
        0.93733772,
        0.56561485,
    ],
    'y': [
        0.00000000e00,
        0.00000000e00,
        2.97966774e-01,
        -7.40288249e-03,
        2.88075787e-01,
        5.58597297e-01,
        3.43907273e-02,
        4.02452063e-01,
        6.82932625e-01,
        8.79912004e-02,
        3.15038755e-01,
        6.01735282e-01,
        5.83504510e-01,
        8.16235127e-01,
        8.25348085e-01,
        7.90247555e-01,
        9.61392619e-01,
        9.40583936e-01,
        8.91682464e-01,
        3.15038755e-01,
        3.03768508e-01,
        5.90907393e-01,
        5.28159339e-01,
        2.66750788e-01,
        7.55856828e-01,
        5.58940556e-01,
        2.57651311e-01,
        8.03691264e-01,
        4.45496404e-17,
        -2.97966774e-01,
        7.40288249e-03,
        -2.88075787e-01,
        -5.58597297e-01,
        -3.43907273e-02,
        -4.02452063e-01,
        -6.82932625e-01,
        -8.79912004e-02,
        -3.15038755e-01,
        -6.01735282e-01,
        -5.83504510e-01,
        -8.16235127e-01,
        -8.25348085e-01,
        -7.90247555e-01,
        -9.61392619e-01,
        -9.40583936e-01,
        -8.91682464e-01,
        -3.15038755e-01,
        -3.03768508e-01,
        -5.90907393e-01,
        -5.28159339e-01,
        -2.66750788e-01,
        -7.55856828e-01,
        -5.58940556e-01,
        -2.57651311e-01,
        -8.03691264e-01,
    ],
    'weights': [
        0.11991389,
        0.10939037,
        0.08960587,
        0.08090154,
        0.05466292,
        0.05634123,
        0.04252012,
        0.02931195,
        0.024671,
        0.01620811,
        0.10939037,
        0.08960587,
        0.08090154,
        0.05466292,
        0.05634123,
        0.04252012,
        0.02931195,
        0.024671,
        0.01620811,
        0.10939037,
        0.08960587,
        0.08090154,
        0.05466292,
        0.05634123,
        0.04252012,
        0.02931195,
        0.024671,
        0.01620811,
        0.10939037,
        0.08960587,
        0.08090154,
        0.05466292,
        0.05634123,
        0.04252012,
        0.02931195,
        0.024671,
        0.01620811,
        0.10939037,
        0.08960587,
        0.08090154,
        0.05466292,
        0.05634123,
        0.04252012,
        0.02931195,
        0.024671,
        0.01620811,
        0.10939037,
        0.08960587,
        0.08090154,
        0.05466292,
        0.05634123,
        0.04252012,
        0.02931195,
        0.024671,
        0.01620811,
    ],
}
# https://www.sciencedirect.com/science/article/pii/S0377042721006324
# File T37_FullSym_Cheby.txt
disk256_cheb = {
    'x': [
        -9.89354995e-01,
        -9.89354995e-01,
        -9.63603853e-01,
        -9.60614314e-01,
        -9.60614314e-01,
        -9.48778797e-01,
        -9.48778797e-01,
        -9.15685626e-01,
        -9.07471645e-01,
        -9.07471645e-01,
        -9.02811793e-01,
        -9.02811793e-01,
        -9.02419597e-01,
        -9.02419597e-01,
        -8.78289223e-01,
        -8.78289223e-01,
        -8.40041966e-01,
        -8.40041966e-01,
        -8.38750627e-01,
        -8.23327406e-01,
        -8.23327406e-01,
        -8.21841102e-01,
        -8.21841102e-01,
        -8.10373461e-01,
        -8.10373461e-01,
        -7.88519484e-01,
        -7.88519484e-01,
        -7.45399150e-01,
        -7.45399150e-01,
        -7.42515786e-01,
        -7.42515786e-01,
        -7.27807207e-01,
        -7.27807207e-01,
        -7.16489300e-01,
        -7.16489300e-01,
        -7.07093783e-01,
        -7.07093783e-01,
        -6.89481452e-01,
        -6.89481452e-01,
        -6.46317592e-01,
        -6.37347457e-01,
        -6.37347457e-01,
        -6.32239589e-01,
        -6.32239589e-01,
        -6.18454877e-01,
        -6.18454877e-01,
        -6.18406587e-01,
        -6.18406587e-01,
        -5.87765586e-01,
        -5.87765586e-01,
        -5.86854016e-01,
        -5.86854016e-01,
        -5.53963344e-01,
        -5.53963344e-01,
        -5.32035716e-01,
        -5.32035716e-01,
        -5.15554766e-01,
        -5.15554766e-01,
        -5.05943611e-01,
        -5.05943611e-01,
        -5.05773807e-01,
        -5.05773807e-01,
        -4.94621055e-01,
        -4.94621055e-01,
        -4.58672962e-01,
        -4.58672962e-01,
        -4.46644301e-01,
        -4.46644301e-01,
        -4.35339932e-01,
        -4.35339932e-01,
        -4.27144020e-01,
        -4.27144020e-01,
        -4.24463829e-01,
        -4.06830496e-01,
        -4.06830496e-01,
        -3.67539722e-01,
        -3.67539722e-01,
        -3.54806351e-01,
        -3.54806351e-01,
        -3.54128735e-01,
        -3.54128735e-01,
        -3.38749848e-01,
        -3.38749848e-01,
        -3.20108006e-01,
        -3.20108006e-01,
        -2.95433758e-01,
        -2.95433758e-01,
        -2.75915381e-01,
        -2.75915381e-01,
        -2.72109524e-01,
        -2.72109524e-01,
        -2.59420458e-01,
        -2.59420458e-01,
        -2.48973447e-01,
        -2.23822307e-01,
        -2.23822307e-01,
        -2.02025178e-01,
        -2.02025178e-01,
        -1.85577585e-01,
        -1.85577585e-01,
        -1.80379643e-01,
        -1.80379643e-01,
        -1.73329587e-01,
        -1.73329587e-01,
        -1.26525282e-01,
        -1.26525282e-01,
        -1.25996184e-01,
        -1.25996184e-01,
        -1.14525985e-01,
        -1.14525985e-01,
        -9.29891232e-02,
        -9.29891232e-02,
        -9.02515493e-02,
        -9.02515493e-02,
        -8.72073904e-02,
        -8.72073904e-02,
        -1.02703699e-13,
        -1.02703699e-13,
        -9.95127221e-14,
        -9.95127221e-14,
        -8.60758530e-14,
        -8.60758530e-14,
        -6.56417513e-14,
        -6.56417513e-14,
        -4.12008317e-14,
        -4.12008317e-14,
        -2.18633579e-14,
        -2.18633579e-14,
        0.00000000e00,
        2.17435766e-14,
        2.18892495e-14,
        4.09751075e-14,
        4.12496236e-14,
        6.52821242e-14,
        6.57194872e-14,
        8.56042751e-14,
        8.61777880e-14,
        9.89675284e-14,
        9.96305696e-14,
        1.02141023e-13,
        1.02825326e-13,
        8.72073904e-02,
        8.72073904e-02,
        9.02515493e-02,
        9.02515493e-02,
        9.29891232e-02,
        9.29891232e-02,
        1.14525985e-01,
        1.14525985e-01,
        1.25996184e-01,
        1.25996184e-01,
        1.26525282e-01,
        1.26525282e-01,
        1.73329587e-01,
        1.73329587e-01,
        1.80379643e-01,
        1.80379643e-01,
        1.85577585e-01,
        1.85577585e-01,
        2.02025178e-01,
        2.02025178e-01,
        2.23822307e-01,
        2.23822307e-01,
        2.48973447e-01,
        2.59420458e-01,
        2.59420458e-01,
        2.72109524e-01,
        2.72109524e-01,
        2.75915381e-01,
        2.75915381e-01,
        2.95433758e-01,
        2.95433758e-01,
        3.20108006e-01,
        3.20108006e-01,
        3.38749848e-01,
        3.38749848e-01,
        3.54128735e-01,
        3.54128735e-01,
        3.54806351e-01,
        3.54806351e-01,
        3.67539722e-01,
        3.67539722e-01,
        4.06830496e-01,
        4.06830496e-01,
        4.24463829e-01,
        4.27144020e-01,
        4.27144020e-01,
        4.35339932e-01,
        4.35339932e-01,
        4.46644301e-01,
        4.46644301e-01,
        4.58672962e-01,
        4.58672962e-01,
        4.94621055e-01,
        4.94621055e-01,
        5.05773807e-01,
        5.05773807e-01,
        5.05943611e-01,
        5.05943611e-01,
        5.15554766e-01,
        5.15554766e-01,
        5.32035716e-01,
        5.32035716e-01,
        5.53963344e-01,
        5.53963344e-01,
        5.86854016e-01,
        5.86854016e-01,
        5.87765586e-01,
        5.87765586e-01,
        6.18406587e-01,
        6.18406587e-01,
        6.18454877e-01,
        6.18454877e-01,
        6.32239589e-01,
        6.32239589e-01,
        6.37347457e-01,
        6.37347457e-01,
        6.46317592e-01,
        6.89481452e-01,
        6.89481452e-01,
        7.07093783e-01,
        7.07093783e-01,
        7.16489300e-01,
        7.16489300e-01,
        7.27807207e-01,
        7.27807207e-01,
        7.42515786e-01,
        7.42515786e-01,
        7.45399150e-01,
        7.45399150e-01,
        7.88519484e-01,
        7.88519484e-01,
        8.10373461e-01,
        8.10373461e-01,
        8.21841102e-01,
        8.21841102e-01,
        8.23327406e-01,
        8.23327406e-01,
        8.38750627e-01,
        8.40041966e-01,
        8.40041966e-01,
        8.78289223e-01,
        8.78289223e-01,
        9.02419597e-01,
        9.02419597e-01,
        9.02811793e-01,
        9.02811793e-01,
        9.07471645e-01,
        9.07471645e-01,
        9.15685626e-01,
        9.48778797e-01,
        9.48778797e-01,
        9.60614314e-01,
        9.60614314e-01,
        9.63603853e-01,
        9.89354995e-01,
        9.89354995e-01,
    ],
    'y': [
        -8.54274913e-02,
        8.54274913e-02,
        1.18007437e-16,
        -2.53749455e-01,
        2.53749455e-01,
        -1.72139874e-01,
        1.72139874e-01,
        1.12139147e-16,
        -3.44220770e-01,
        3.44220770e-01,
        -4.19534898e-01,
        4.19534898e-01,
        -1.16912133e-01,
        1.16912133e-01,
        -2.79094436e-01,
        2.79094436e-01,
        -4.97688745e-01,
        4.97688745e-01,
        1.02717327e-16,
        -1.84344845e-01,
        1.84344845e-01,
        -4.43649756e-01,
        4.43649756e-01,
        -5.77019514e-01,
        5.77019514e-01,
        -3.54498357e-01,
        3.54498357e-01,
        -9.34074640e-02,
        9.34074640e-02,
        -6.08576295e-01,
        6.08576295e-01,
        -5.29492212e-01,
        5.29492212e-01,
        -2.77407786e-01,
        2.77407786e-01,
        -6.94829673e-01,
        6.94829673e-01,
        -4.56720232e-01,
        4.56720232e-01,
        7.91510771e-17,
        -6.46707088e-01,
        6.46707088e-01,
        -1.84887698e-01,
        1.84887698e-01,
        -4.04477034e-01,
        4.04477034e-01,
        -7.34752585e-01,
        7.34752585e-01,
        -5.78782836e-01,
        5.78782836e-01,
        8.02071653e-01,
        -8.02071653e-01,
        -3.19749600e-01,
        3.19749600e-01,
        -9.33545035e-02,
        9.33545035e-02,
        -7.15643074e-01,
        7.15643074e-01,
        -7.75115046e-01,
        7.75115046e-01,
        -5.12090758e-01,
        5.12090758e-01,
        8.36859927e-01,
        -8.36859927e-01,
        -2.33719109e-01,
        2.33719109e-01,
        -6.66231321e-01,
        6.66231321e-01,
        -8.94398638e-01,
        8.94398638e-01,
        -4.15374184e-01,
        4.15374184e-01,
        5.19818270e-17,
        -8.36355536e-01,
        8.36355536e-01,
        -5.93992220e-01,
        5.93992220e-01,
        -7.82165518e-01,
        7.82165518e-01,
        -1.30925410e-01,
        1.30925410e-01,
        -9.08333485e-01,
        9.08333485e-01,
        -3.17902921e-01,
        3.17902921e-01,
        -5.03985373e-01,
        5.03985373e-01,
        -9.54772822e-01,
        9.54772822e-01,
        -7.12941188e-01,
        7.12941188e-01,
        -8.79053329e-01,
        8.79053329e-01,
        3.04904535e-17,
        -1.96347607e-01,
        1.96347607e-01,
        -4.23307182e-01,
        4.23307182e-01,
        -6.19168816e-01,
        6.19168816e-01,
        -8.20902263e-01,
        8.20902263e-01,
        -9.47507544e-01,
        9.47507544e-01,
        -9.84480807e-01,
        9.84480807e-01,
        -3.08425341e-01,
        3.08425341e-01,
        -9.79085461e-02,
        9.79085461e-02,
        -7.36700729e-01,
        7.36700729e-01,
        -5.16916932e-01,
        5.16916932e-01,
        -9.04751651e-01,
        9.04751651e-01,
        -9.93154978e-01,
        9.93154978e-01,
        -9.62297912e-01,
        9.62297912e-01,
        -8.32362053e-01,
        8.32362053e-01,
        -6.34762258e-01,
        6.34762258e-01,
        -3.98416137e-01,
        3.98416137e-01,
        -2.11420844e-01,
        2.11420844e-01,
        0.00000000e00,
        -2.11420844e-01,
        2.11420844e-01,
        -3.98416137e-01,
        3.98416137e-01,
        -6.34762258e-01,
        6.34762258e-01,
        -8.32362053e-01,
        8.32362053e-01,
        -9.62297912e-01,
        9.62297912e-01,
        -9.93154978e-01,
        9.93154978e-01,
        -9.04751651e-01,
        9.04751651e-01,
        -5.16916932e-01,
        5.16916932e-01,
        -7.36700729e-01,
        7.36700729e-01,
        -9.79085461e-02,
        9.79085461e-02,
        -3.08425341e-01,
        3.08425341e-01,
        -9.84480807e-01,
        9.84480807e-01,
        -9.47507544e-01,
        9.47507544e-01,
        -8.20902263e-01,
        8.20902263e-01,
        -6.19168816e-01,
        6.19168816e-01,
        -4.23307182e-01,
        4.23307182e-01,
        -1.96347607e-01,
        1.96347607e-01,
        -0.00000000e00,
        -8.79053329e-01,
        8.79053329e-01,
        -7.12941188e-01,
        7.12941188e-01,
        -9.54772822e-01,
        9.54772822e-01,
        -5.03985373e-01,
        5.03985373e-01,
        -3.17902921e-01,
        3.17902921e-01,
        -9.08333485e-01,
        9.08333485e-01,
        -1.30925410e-01,
        1.30925410e-01,
        -7.82165518e-01,
        7.82165518e-01,
        -5.93992220e-01,
        5.93992220e-01,
        -8.36355536e-01,
        8.36355536e-01,
        -0.00000000e00,
        -4.15374184e-01,
        4.15374184e-01,
        -8.94398638e-01,
        8.94398638e-01,
        -6.66231321e-01,
        6.66231321e-01,
        -2.33719109e-01,
        2.33719109e-01,
        -8.36859927e-01,
        8.36859927e-01,
        -5.12090758e-01,
        5.12090758e-01,
        -7.75115046e-01,
        7.75115046e-01,
        -7.15643074e-01,
        7.15643074e-01,
        -9.33545035e-02,
        9.33545035e-02,
        -3.19749600e-01,
        3.19749600e-01,
        -8.02071653e-01,
        8.02071653e-01,
        -5.78782836e-01,
        5.78782836e-01,
        -7.34752585e-01,
        7.34752585e-01,
        -4.04477034e-01,
        4.04477034e-01,
        -1.84887698e-01,
        1.84887698e-01,
        -6.46707088e-01,
        6.46707088e-01,
        0.00000000e00,
        -4.56720232e-01,
        4.56720232e-01,
        -6.94829673e-01,
        6.94829673e-01,
        -2.77407786e-01,
        2.77407786e-01,
        -5.29492212e-01,
        5.29492212e-01,
        -6.08576295e-01,
        6.08576295e-01,
        -9.34074640e-02,
        9.34074640e-02,
        -3.54498357e-01,
        3.54498357e-01,
        -5.77019514e-01,
        5.77019514e-01,
        -4.43649756e-01,
        4.43649756e-01,
        -1.84344845e-01,
        1.84344845e-01,
        -0.00000000e00,
        -4.97688745e-01,
        4.97688745e-01,
        -2.79094436e-01,
        2.79094436e-01,
        -1.16912133e-01,
        1.16912133e-01,
        -4.19534898e-01,
        4.19534898e-01,
        -3.44220770e-01,
        3.44220770e-01,
        -0.00000000e00,
        -1.72139874e-01,
        1.72139874e-01,
        -2.53749455e-01,
        2.53749455e-01,
        0.00000000e00,
        -8.54274913e-02,
        8.54274913e-02,
    ],
    'weights': [
        0.00304864,
        0.00304864,
        0.00697655,
        0.00283667,
        0.00283667,
        0.00706348,
        0.00706348,
        0.00568279,
        0.00605789,
        0.00605789,
        0.00216072,
        0.00216072,
        0.00953328,
        0.00953328,
        0.00997456,
        0.00997456,
        0.00467235,
        0.00467235,
        0.01522108,
        0.01446203,
        0.01446203,
        0.00893787,
        0.00893787,
        0.00227121,
        0.00227121,
        0.01296512,
        0.01296512,
        0.01802272,
        0.01802272,
        0.00768166,
        0.00768166,
        0.01012426,
        0.01012426,
        0.0175236,
        0.0175236,
        0.00346221,
        0.00346221,
        0.01257942,
        0.01257942,
        0.02009356,
        0.00990121,
        0.00990121,
        0.01952127,
        0.01952127,
        0.01510938,
        0.01510938,
        0.00743716,
        0.00743716,
        0.01487829,
        0.01487829,
        0.00270526,
        0.00270526,
        0.01798968,
        0.01798968,
        0.02092744,
        0.02092744,
        0.00961021,
        0.00961021,
        0.00572029,
        0.00572029,
        0.01811231,
        0.01811231,
        0.00534969,
        0.00534969,
        0.02165024,
        0.02165024,
        0.01513193,
        0.01513193,
        0.0024117,
        0.0024117,
        0.0212803,
        0.0212803,
        0.0202805,
        0.00729284,
        0.00729284,
        0.01746787,
        0.01746787,
        0.01368878,
        0.01368878,
        0.02249702,
        0.02249702,
        0.00607783,
        0.00607783,
        0.02375824,
        0.02375824,
        0.0188308,
        0.0188308,
        0.00259008,
        0.00259008,
        0.01751727,
        0.01751727,
        0.01024844,
        0.01024844,
        0.02674863,
        0.02196638,
        0.02196638,
        0.01935688,
        0.01935688,
        0.0207142,
        0.0207142,
        0.01486112,
        0.01486112,
        0.00709434,
        0.00709434,
        0.00265585,
        0.00265585,
        0.02254266,
        0.02254266,
        0.02420384,
        0.02420384,
        0.01849843,
        0.01849843,
        0.02104849,
        0.02104849,
        0.01129557,
        0.01129557,
        0.00104754,
        0.00104754,
        0.00372079,
        0.00372079,
        0.00770471,
        0.00770471,
        0.01029415,
        0.01029415,
        0.01140562,
        0.01140562,
        0.01279941,
        0.01279941,
        0.0224139,
        0.01279941,
        0.01279941,
        0.01140562,
        0.01140562,
        0.01029415,
        0.01029415,
        0.00770471,
        0.00770471,
        0.00372079,
        0.00372079,
        0.00104754,
        0.00104754,
        0.01129557,
        0.01129557,
        0.02104849,
        0.02104849,
        0.01849843,
        0.01849843,
        0.02420384,
        0.02420384,
        0.02254266,
        0.02254266,
        0.00265585,
        0.00265585,
        0.00709434,
        0.00709434,
        0.01486112,
        0.01486112,
        0.0207142,
        0.0207142,
        0.01935688,
        0.01935688,
        0.02196638,
        0.02196638,
        0.02674863,
        0.01024844,
        0.01024844,
        0.01751727,
        0.01751727,
        0.00259008,
        0.00259008,
        0.0188308,
        0.0188308,
        0.02375824,
        0.02375824,
        0.00607783,
        0.00607783,
        0.02249702,
        0.02249702,
        0.01368878,
        0.01368878,
        0.01746787,
        0.01746787,
        0.00729284,
        0.00729284,
        0.0202805,
        0.0212803,
        0.0212803,
        0.0024117,
        0.0024117,
        0.01513193,
        0.01513193,
        0.02165024,
        0.02165024,
        0.00534969,
        0.00534969,
        0.01811231,
        0.01811231,
        0.00572029,
        0.00572029,
        0.00961021,
        0.00961021,
        0.02092744,
        0.02092744,
        0.01798968,
        0.01798968,
        0.00270526,
        0.00270526,
        0.01487829,
        0.01487829,
        0.00743716,
        0.00743716,
        0.01510938,
        0.01510938,
        0.01952127,
        0.01952127,
        0.00990121,
        0.00990121,
        0.02009356,
        0.01257942,
        0.01257942,
        0.00346221,
        0.00346221,
        0.0175236,
        0.0175236,
        0.01012426,
        0.01012426,
        0.00768166,
        0.00768166,
        0.01802272,
        0.01802272,
        0.01296512,
        0.01296512,
        0.00227121,
        0.00227121,
        0.00893787,
        0.00893787,
        0.01446203,
        0.01446203,
        0.01522108,
        0.00467235,
        0.00467235,
        0.00997456,
        0.00997456,
        0.00953328,
        0.00953328,
        0.00216072,
        0.00216072,
        0.00605789,
        0.00605789,
        0.00568279,
        0.00706348,
        0.00706348,
        0.00283667,
        0.00283667,
        0.00697655,
        0.00304864,
        0.00304864,
    ],
}
