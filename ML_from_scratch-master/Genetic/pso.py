#generate population 
#maintaain global best fitness, and local best fitness 
#choose params 
#generate donor vector 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn 
pop_size = 10 
LB = -10
UB = 10
D = 3
alpha = 0.05
beta = 0.05


def fitness_function(x):
	k = x.position
	return sum(k*k)


class Chromosome:
	def __init__(self, LB, UB, D):
		self.position = np.random.uniform(LB, UB, D)
		self.velocity = np.zeros(D)
		self.best_position = self.position	
		self.best_fitness = np.inf			


chromosomes = []
glo_best = np.inf
global_best_chrom = np.zeros(D)
particle_fitnesses = []

global_best_curve = []
#generate random chromosomes 
for i in range(100):
	chromosomes.append(Chromosome(LB, UB, D))
	particle_fitnesses.append([])
	fit = fitness_function(chromosomes[i])
	if fit < chromosomes[i].best_fitness:
		chromosomes[i].best_fitness = fit
		chromosomes[i].best_position = chromosomes[i].position
	if fit < glo_best:
		glo_best = fit
		global_best_chrom = chromosomes[i].position

for i in range(2000):
	glo_fit = np.inf
	glo_chomo = np.zeros(D)
	for i, chromosome in enumerate(chromosomes):
		chromosome.velocity += alpha*(global_best_chrom - chromosome.position) + \
						beta*(chromosome.best_position - chromosome.position)
		if any(chromosome.velocity) < -1 or any(chromosome.velocity) > 1:
			chromosome.velocity = np.random.uniform(-1, 1, D)

		chromosome.position += chromosome.velocity
		if any(chromosome.position) < LB or any(chromosome.velocity) > UB:
			chromosome.velocity = np.random.uniform(LB, UB, D)

		fit = fitness_function(chromosome)	
		if fit < chromosome.best_fitness:
			chromosome.best_fitness = fit
			chromosome.best_position = chromosome.position
		particle_fitnesses[i].append(chromosome.best_fitness)	
		if fit < glo_fit:
			glo_fit = fit	
			glo_chomo = chromosome.position
		

	if glo_fit < glo_best:
		glo_best = glo_fit
		global_best_chrom = glo_chomo

	global_best_curve.append(glo_best)	
				
plt.plot(global_best_curve,  'o')	
for i in range(len(particle_fitnesses)):
	plt.plot(particle_fitnesses[i])
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.xlim(0, 15)
plt.legend(loc='best')
plt.show()			




