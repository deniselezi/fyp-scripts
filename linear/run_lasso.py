from Lassoer import Lassoer


"""
3.327878787878788e-05
6.363636363636363e-06
3.6e-06
2.9e-05
"""


# for season in range(1, 5):
#     model = Lassoer(season=season)
#     model.fit()


model = Lassoer(season=4, alpha=2.9e-05)
model.fit()
