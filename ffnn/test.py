from Neuraler import Neuraler

model = Neuraler(2)


"""
Best learning rates:
16-17: 0.0035
undelayed 16-17: 0.0003
17-18: 0.0081
undelayed 17-18: 0.0094
"""

model.fit(tune=False, lr=0.0035)
