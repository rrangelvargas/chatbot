from model import Model
import matplotlib.pyplot as plt

iteration_list = list(range(1, 4001))

m1 = Model(model_name="dot_model")
m1.run('data/input/pairs.csv')
loss1 = m1.train()

m2 = Model(model_name="dot_model", attn_model='general')
m2.run('data/input/pairs.csv')
loss2 = m2.train()

m3 = Model(model_name="dot_model", attn_model='concat')
m3.run('data/input/pairs.csv')
loss3 = m3.train()

plt.plot(
    iteration_list, loss1, 'b',
    iteration_list, loss2, 'r',
    iteration_list, loss3, 'g'
)
plt.xlabel('Iteration')
plt.ylabel('Average Loss')
plt.show()



