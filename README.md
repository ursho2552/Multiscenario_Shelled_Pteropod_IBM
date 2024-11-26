## Multi-scenario Shelled Pteropod Individual-Based Model (mspIBM)

**spIBM** is a set of python modules and methods to simulate the mortality, growth, development and behaviour (diel vertical migration, spawning, active avoidance of low oxygen) of shelled pteropods across different life-stages. The model uses the size as a key trait to determine the behaviour and movement of individuals. The movement is simulated using the python package [parcels v2.1.3](https://doi.org/10.5281/zenodo.3630568).

### spIBM manuscript and code

The manuscript detailing the changes to the model for the Multiscenario study, has been published in [Global Change Biology](https://doi.org/10.1111/gcb.17345), and can be cited as:
*Hofmann Elizondo, U., Vogt, M., Bednaršek, N., Münnich, M., & Gruber, N. (2024). The impact of aragonite saturation variability on shelled pteropods: An attribution study in the California Current System. Global Change Biology, 30, e17345. https://doi.org/10.1111/gcb.17345*

### Main changes

The main changes in this version is the addition of multiple scenarios to which the individuals react. By doing this, we can simulate pteropod populations under different scenarios, creating "what if"-scenarios relative to a NULL-scenario where pteropods are not affected by low $\Omega$ conditions. In addition to the NULL-scenario, we have a scenario that contains the effects of the natural variability (N), the long-term trend (T), and extremes (E) on $\Omega$ conditions, the second scenario contains the natural variability and the long-term trend, and the third scenario contains only the natural variability. The differences in responses can then be used to attribute responses to each component.

In addition, we improved how the growth, mortality, and diel vertical migration are represented, to fit better with recent studies in the California Current System.


