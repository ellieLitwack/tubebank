# tubebank

tubebank is a Python library for dealing with tube banks in crossflow. You can learn more about tube banks at [Thermopedia](https://www.thermopedia.com/content/1211/).

tubebank is based on empirical data from A. Zukauskas. It is not highly accurate, especially in the low-intermediate Re range.

tubebank was developed by Ellie Litwack for Baltimore Aircoil Company.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tubebank.

```bash
pip install tubebank
```

## Usage

```python
import tubebank

#  If n <= 7, a correction factor for short tube banks is applied.
n = 6 #  The number of rows.
diameter = 0.02 #  tube diameters (length)
st = 0.03 #  transverse pitch (length)
sl = 0.03 #  longitudinal pitch (length)
mean_superficial_velocity = 2 #  (length/time)
density = 1.2 #  mass/(length*time)
viscosity = 1.8e-5 #  dynamic viscosity (mass/(length*time))
staggered = True #  False for inline banks, true for stagggered ones
#  alpha, beta, viscosity_ratio, and bounds_warnings are optional.
#  get the pressure drop across a tube bank
tubebank.get_pressure_drop(n, staggered, diameter, st, sl,
     mean_superficial_velocity, density, viscosity)

prandlt_number = 0.7
#  get the Nusselt number of a tube bank
#  valid for n >= 20, 0.7 ~<= Prandlt number ~<= 500,
#  1000 ~<= Re ~<= 2e6
tubebank.get_nusselt_number(staggered, diameter, mean_superficial_velocity, 
     density, viscosity, st, sl, prandlt_number, n)

```

## Contributing
Pull requests are welcome! I would be particularly interested in contributions that could incorporate more recent/accurate data.

## License
[MIT](https://choosealicense.com/licenses/mit/)