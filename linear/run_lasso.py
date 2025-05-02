from Lassoer import Lassoer

# import pickle
# import numpy as np

"""
Optimal alphas per target season:
3.327878787878788e-05
6.363636363636363e-06
3.6e-06
2.9e-05
"""

model = Lassoer(season=4, alpha=2.9e-05)
model.fit()

# to find model sparsity
# zero_weights = np.sum(model.model.coef_ == 0)
# print(f"{(1000 - zero_weights)}")

# to save model
# with open('lasso.pkl','wb') as f:
#     pickle.dump(model.model, f)
