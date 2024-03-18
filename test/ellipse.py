import numpy as np
import pocket.advis
import matplotlib.pyplot as plt

mu = np.asarray([
    [1., 1.,],
    [0., 0.,]
])
sig = [
    np.asarray([[.2, .3], [.3, 1.2]]),
    np.asarray([[.5, 0.], [0., .5]]),
]

ax = pocket.advis.visualise_distributions(mu, sig, labels=['Cars', 'DTD'])
plt.show()