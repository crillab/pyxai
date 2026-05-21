# Example:
# @article{TODO}
# Check V1.0: Ok
# Check V2.0: Ok

from pyxai import Builder, Explaining

# Builder part
model = Builder.LinearModelSimple(3, coefficients=[(5, 2, -3)], intercepts=[-1])

explainer = Explaining.initialize(model, instance=(1,1,0))

explainer.set_upper_lower_bounds(((0, 0, 0), (1, 1, 1)))

print("reasons:", explainer.model_specific_reason())

