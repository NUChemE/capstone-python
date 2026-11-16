# Python for capstone scale-up and techno-economics

Materials from a guest session in the Chemical Engineering Capstone at Northeastern University,
16 November 2026 (Richard West).

## Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NUChemE/capstone-python/blob/main/scaleup-in-python.ipynb)

Click the badge to open the worked example, [`scaleup-in-python.ipynb`](scaleup-in-python.ipynb),
in Google Colab. You need a Google account and nothing else. Run the cells in order with
**Shift + Enter**. To keep your changes, use *File → Save a copy in Drive*.

The notebook takes a toy process, A → B in a CSTR, from a 50 L pilot reactor to a 300,000 t/yr
plant:

1. All the parameters in one dictionary.
2. The mole balance written as a Python function.
3. Solving it with `fsolve`: *rating* mode (given the reactor, what conversion?) and *design* mode
   (given the conversion, what size of reactor?).
4. Sweeping the production rate.
5. Checking what changes with scale: the jacket can no longer remove the heat.
6. Capital cost by the six-tenths rule, operating costs, and the minimum selling price.
7. A tornado chart of what matters most, and the optimum conversion.
8. An appendix with the same reaction in a PFR, using `solve_ivp`.

All the numbers in it are placeholders. Swap in your own.

## Python is not Aspen

Python has no property package and no library of unit operations until you add one. What it does
well is the work around them. You can model the step that isn't standard, sweep the scale, add
costs, and run sensitivity and uncertainty analyses, all in one notebook that you can rerun and
someone else can check. Many projects use both: Aspen for the rigorous flowsheet and properties,
and Python for everything wrapped around its numbers.

## Which tool for which problem

| You want to… | Use | Where to learn it |
|---|---|---|
| Solve steady-state mass and energy balances (algebraic equations) | `scipy.optimize.fsolve` or `root` | [pycse: nonlinear algebra](https://kitchingroup.cheme.cmu.edu/pycse/book/07-nla-1.html) |
| Model a batch reactor, a PFR, or a start-up (ODEs) | `scipy.integrate.solve_ivp` | [pycse: systems of ODEs](https://kitchingroup.cheme.cmu.edu/pycse/book/04-fode-2.html) |
| Fit a rate law or a correlation to data | `scipy.optimize.curve_fit` | [pycse: nonlinear regression](https://kitchingroup.cheme.cmu.edu/pycse/book/11-regression.html) |
| Find the best design (minimize cost, maximize profit) | `scipy.optimize.minimize`, `minimize_scalar` | [pycse: optimization](https://kitchingroup.cheme.cmu.edu/pycse/book/10-min-max.html), [constrained](https://kitchingroup.cheme.cmu.edu/pycse/book/13-constrained-optimization.html) |
| Keep track of units | [`pint`](https://pint.readthedocs.io/) | its tutorial |
| Get accurate properties of a common pure fluid (steam, CO₂, air, N₂, H₂, NH₃, refrigerants, light hydrocarbons) | [`CoolProp`](http://www.coolprop.org/) | its docs; start with `PropsSI` |
| Get properties of almost any other chemical, or of mixtures (flashes, bubble and dew points, activity coefficients, ΔH<sub>f</sub>) | [`thermo`](https://thermo.readthedocs.io/) and [`chemicals`](https://chemicals.readthedocs.io/) | their docs; values are often estimated, so check a few against a handbook |
| Work with kinetics, equilibrium and thermochemistry of gas mixtures | [`cantera`](https://cantera.org/) | its examples |
| Handle tables of costs, data, or results | [`pandas`](https://pandas.pydata.org/) | its "10 minutes to pandas" page |
| Build rigorous, equation-oriented flowsheets | [IDAES](https://idaes.org/) (built on [Pyomo](https://www.pyomo.org/)) | powerful, but a steep climb |
| See where flowsheet modeling in Python is heading | [difflow](https://kitchingroup.cheme.cmu.edu/differentiable-flowsheets/README.html) | [TEA example](https://kitchingroup.cheme.cmu.edu/differentiable-flowsheets/examples/05_technoeconomic_analysis.html) |

Everything above can be installed in Colab with a cell such as `!pip install pint CoolProp`.

A note on **difflow**: John Kitchin's differentiable flowsheets library does unit operations,
recycle loops, costing, NPV and optimization in one place, and it can differentiate all of it.
It is alpha software. Its own README says much of the code was written with an AI assistant and
asks users to check the equations. That applies to any code an AI writes for you, including your
own.

## A good free book

John Kitchin's **[pycse: Python Computations in Science and Engineering](https://kitchingroup.cheme.cmu.edu/pycse/intro.html)**
was written for chemical engineers. Every chapter is a notebook with an *Open in Colab* button.
Start with the chapters linked in the table above.

## Installing Python on your own computer

Colab is fine for this course. If you want Python on your own machine for the rest of your project:

1. Install **[Miniforge](https://github.com/conda-forge/miniforge)** (free, small) or the
   **[Anaconda Distribution](https://www.anaconda.com/download)** (larger, with a graphical
   launcher; free for university teaching and research, but check its terms before using it at a
   company).
2. Install the libraries. In a terminal (on Windows, the *Miniforge Prompt*):
   `conda install numpy scipy matplotlib pandas jupyterlab`
3. Edit and run notebooks in **[VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)**,
   or start JupyterLab by typing `jupyter lab`.

If your laptop can't cope, Northeastern Research Computing runs JupyterLab in the browser on the
Explorer cluster through [Open OnDemand](https://rc-docs.northeastern.edu/).

## About

The notebook and this page were planned by Richard West and mostly written by Claude Code
(Opus 5.5), an AI assistant. Every cell was run to check that it works. All the numbers are
illustrative placeholders. MIT licence (see [LICENSE](LICENSE)).
