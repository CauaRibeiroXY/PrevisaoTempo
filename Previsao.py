from pomegranate import *
import random
# Modelo Discreto
sol = DiscreteDistribution({
    "chovendo": 0.2,
    "ensolarado": 0.8
})

chuva = DiscreteDistribution({
    "chovendo": 0.7,
    "ensolarado": 0.3
})

states = [chuva, sol]

# Modelo de Transição
transitions = numpy.array(
    [[0.4, 0.6], # Previsão de amanha se hoje for chuva   [sol,chuva]
     [0.1, 0.9]] # Previsão de amanha se hoje for sol
)


starts = numpy.array([0.3, 0.7])

# Criar Modelo
model = HiddenMarkovModel.from_matrix(
    transitions, states, starts,
    state_names=["chuva", "sol"]
)
model.bake()


# Dados observaveis
amostra = []
for i in range(0,100):
  a = random.randrange(0,2)
  #print(a)
  if a == 0:
    amostra.append('chovendo')
  else:
    amostra.append('ensolarado')


# Previsão baseada na amostra 
previsoes = model.predict(amostra)
for previsao in previsoes:
    print(model.states[previsao].name)