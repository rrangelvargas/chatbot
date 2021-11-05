from ml_model import Model

model = Model()

model.run('data/input/pairs.csv', 'data/output/cb_model/2-2_500/4000_checkpoint.tar')

model.evaluate_input()
# ml_model.train()
