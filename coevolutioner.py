'''
Created on Apr 4, 2014

@author: kevin_000
'''

import numpy as np
import math as mt
import random as rd
#from sklearn.preprocessing import scale
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import copy

maxorf = 300 # Number of genes wanted.

numstart = 100 # Number of organisms to begin with.
num_species = 10 # Must be a divisor of numstart

map_row = 30
map_col = 30

vital_size = 10
apt_size = 10
sight_size = 10

resource_size = 30 # Upper bound on how many resources can appear on a square. Don't make it larger than 100
mutation = 0.01
max_migrate = 3

generations = 100
output = "coevolutioner_workspace"

class gene(object):
    def __init__(self,code,vital,fert,apt,speed,sight,diet,brain):
        self.code = code
        self.vital = vital
        self.fert = fert
        self.apt = apt # The strength of the organism to fight and compete.
        self.speed = speed # The ability of the organism to escape.
        self.sight = sight
        self.diet = diet
        self.brain = brain

class plant(object):
    def __init__(self,resource,i,j):
        self.amount = np.random.randint(resource)+1
        self.i = i
        self.j = j
        self.resource = resource
        self.bitten = False
    def growth(self):
        self.amount+=5 # Ensures that plant can't grow past given resource size.
    def die(self,thefield):
        thefield.resource_list.remove(self)
        thefield.resource_map[self.i][self.j] = [0]

class field(object):
    def __init__(self,row,col,resource):
        self.space = [[[] for _ in xrange(col)] for _ in xrange(row)]
        self.resource_list = []
        self.resource_map = [[[] for _ in xrange(col)] for _ in xrange(row)]
        for i in xrange(row):
            for j in xrange(col):
                if np.random.random() < resource/float(100):
                    self.resource_map[i][j] = plant(resource,i,j)
                    self.resource_list.append(self.resource_map[i][j])

class organism(object):
    def __init__(self,genes,genome,i,j,reporter,sorted):
        self.genes = genes # Iterate through the gene pool, which is created at the beginning.
        self.sorted = sorted
        self.genome = genome # Organisms with the same genes are in the same species.
        self.vital = 0
        self.startvital = 0
        self.speed = 0 # Ability to escape and move.
        self.sight = 0
        self.fert = 0
        self.apt = 0
        self.diet = 0
        self.gender = np.random.randint(2)
        self.children = []
        self.parent = []
        self.i = i
        self.j = j
        self.choice = 0
        self.brain = 0
        self.reporter = reporter
        for gene in genes:
            self.vital += gene.vital
            self.sight += gene.sight
            self.speed = gene.speed
            self.fert = gene.fert
            self.apt += gene.apt
            self.diet = gene.diet
            self.brain = gene.brain
        if self.speed > 1:
            self.speed = 1
        if self.fert > 1:
            self.fert = 1
        self.startvital = self.vital
        self.age = self.vital # An organism ages starting from its initial vitality.
    def eat(self,food):
        if isinstance(food,organism) and (self.diet == 1 or self.diet == 2):
            food.vital -= self.apt
            self.vital += mt.ceil(self.apt*0.5)
        if isinstance(food,plant) and (self.diet == 0 or self.diet == 2):
            food.amount -= self.apt
            food.bitten = True
            self.vital += self.apt # The amount of nutrient transfer is based on how strong the organism is.
    def die(self,thefield,org_list):
        if self.reporter:
            print "--------------Reporter has died.-----------------"
        thefield.space[self.i][self.j].remove(self)
        org_list.remove(self)
    def decide(self,up,down,left,right,thefield):
        randomwalk = [self.i+np.random.randint(-10,11),self.j+np.random.randint(-10,11)]
        if self.brain == 0: # Dumb organism
            if self.reporter:
                print "Reporter is a dumb organism."
            self.choice = randomwalk
        if self.brain in [1,2,3,4]:
            hunt = [[],[]]
            mate = [[],[]]
            danger = False
            for x in xrange(up,down):
                i = x
                while i < 0:
                    i = len(thefield.space)+i
                while i >= len(thefield.space):
                    i = i-len(thefield.space)
                for y in xrange(left,right):
                    j = y
                    while j < 0:
                        j = len(thefield.space[0])+j
                    while j >= len(thefield.space[0]):
                        j = j-len(thefield.space[0])
                    for spot in thefield.space[i][j]:
                        if isinstance(spot,organism) and spot is not self:
                            for spot in thefield.space[i][j]:
                                if spot.apt > self.apt:
                                    danger = True
                                    escape = [(self.i-(spot.i+1)+self.i),(self.j-(spot.j+1)+self.j)]
                                if(self.diet == 1 or self.diet == 2):
                                    if spot not in self.children and spot not in self.parent:
                                        hunt[0].append([x,y])
                                        hunt[1].append(mt.sqrt((x-self.i)**2+(y-self.j)**2))
                                if speciation(spot,self) and spot.gender != self.gender:
                                    mate[0].append([x,y])
                                    mate[1].append(mt.sqrt((x-self.i)**2+(y-self.j)**2))
                                if spot in self.children:
                                    self.children.remove(spot)
                                if spot in self.parent:
                                    self.parent.remove(spot)
                            if isinstance(thefield.resource_map[i][j],plant) and (self.diet == 0 or self.diet == 2): # If it is a herbivore, plants will motivate its movement as well.
                                hunt[0].append([x,y])
                                hunt[1].append(mt.sqrt((x-self.i)**2+(y-self.j)**2))
                if self.brain in [1,2]: # Prioritize hunting over mating
                    if len(hunt[0])>0:
                        closest = np.argmin(hunt[1])
                        x_move,y_move = hunt[0][closest]
                        self.choice = [x_move,y_move]
                    elif len(mate[0])>0:
                        closest = np.argmin(mate[1])
                        x_move,y_move = mate[0][closest]
                        self.choice = [x_move,y_move]
                    else:
                        self.choice = randomwalk
                if self.brain in [3,4]: # Prioritize mating over hunting
                    if len(mate[0])>0:
                        closest = np.argmin(mate[1])
                        x_move,y_move = mate[0][closest]
                        self.choice = [x_move,y_move]
                    elif len(hunt[0])>0:
                        closest = np.argmin(hunt[1])
                        x_move,y_move = hunt[0][closest]
                        self.choice = [x_move,y_move]
                    else:
                        self.choice = randomwalk
                if self.brain in [1,3]: # Coward, always escapes if danger present
                    if danger:
                        self.choice = escape


def binary(i):
    if i == 0:
        return "0"
    s = ''
    while i:
        if i & 1 == 1:
            s = "1" + s
        else:
            s = "0" + s
        i >>= 1
    return s

def genomemaker(maxorf,gene_pool):
    genome_size = np.random.randint(50,100)
    genome_size = genome_size**100
    genome = list(binary(genome_size)) # Turn genome binary string into list.
    orf = [0,np.random.randint(len(binary(maxorf)))]
    genelist = []
    cistrons = []
    while orf[1] < len(genome):
        genome[orf[0]] = '*' # Transcription start site
        cistron = ''.join(genome[(orf[0]+1):orf[1]]) # Get the code between the asterisks that act as transcription sites.
        for gene in gene_pool:
            if gene.code == cistron and cistron not in cistrons:
                cistrons.append(cistron)
                genelist.append(gene)
        orf[0] = orf[1]
        orf[1] = orf[1]+np.random.randint(len(binary(maxorf)))
    return genome,genelist

def genesorter(genelist):
    integers = []
    for gene in genelist:
        integers.append(int(gene.code,2))
    sorted = np.sort(integers)
    #genelist.sort(key = lambda sort_idx: sort_idx)
    #sorted = genelist
    return sorted

def genepool(maxorf_input,vital_input,apt_input,sight_input):
    genes = []
    bit = '0'
    bit_rep = 0
    for i in xrange(maxorf+1):
        vital = 1
        fert = np.random.random()
        apt = 0
        speed = np.random.random()
        sight = 0
        brain = np.random.randint(5)
        diet = np.random.randint(3)
        trait = np.random.randint(3)
        if trait == 0:
            vital = np.random.randint(vital_input)
        if trait == 1:
            apt = np.random.randint(apt_input)
        if trait == 2:
            sight = np.random.randint(sight_input)
        genes.append(gene(bit,vital,fert,apt,speed,sight,diet,brain))
        bit_rep += 1
        bit = binary(bit_rep)

    return genes

def creation(maxorf,numstart,gene_pool,row,col,resource):
    '''
    size = max gene size
    numstart = number of starting organisms
    genes = the gene pool
    '''
    org_list = []
    thefield = field(row,col,resource)
    reporter = True
    iteration = numstart
    while iteration > 0:
        genome,genes = genomemaker(maxorf,gene_pool)
        for x in xrange(numstart/num_species):
            genelist = copy.deepcopy(genes)
            sorted = genesorter(genelist)
            i = np.random.randint(len(thefield.space))
            j = np.random.randint(len(thefield.space[0]))
            org = organism(genelist,genome,i,j,reporter,sorted)
            print "vital for org"+str(iteration)+": "+str(org.vital)
            print "sight for org"+str(iteration)+": "+str(org.sight)
            print "fert for org"+str(iteration)+": "+str(org.fert)
            print "apt for org"+str(iteration)+": "+str(org.apt)
            print "speed for org"+str(iteration)+": "+str(org.speed)
            print "gender for org"+str(iteration)+": "+str(org.gender)
            print "diet for org"+str(iteration)+": "+str(org.diet)
            print str(sorted)
            thefield.space[i][j].append(org)
            org_list.append(org)
            reporter = False
            iteration -= 1
    print "CREATION DONE ################################"
    return thefield,org_list

def grow_plants(thefield,resource):
    '''
    kills off plants that have zero or less amount.
    Grows plants that still have an amount.
    '''
    for plnt in thefield.resource_list:
        if plnt.amount <= 0:
            plnt.die(thefield)
        else:
            plnt.growth()
    if np.random.randint(resource) <= resource*0.5:
        x = np.random.randint(len(thefield.resource_map))
        y = np.random.randint(len(thefield.resource_map[0]))
        if not isinstance(thefield.resource_map[x][y],plant):
            seed = plant(resource,x,y)
            thefield.resource_map[x][y] = seed
            thefield.resource_list.append(seed)

def decide(thefield,org_list):
    for org in org_list:
        up = org.i-org.sight
        down = org.i+org.sight
        left = org.j-org.sight
        right = org.j+org.sight
        org.decide(up,down,left,right,thefield)

def move(thefield,org_list):
    for org in org_list:
        if org.reporter:
            print "Reporter's vitals: "+str(org.vital)
        if org.choice != 0:
            #print "before moving: "+str(org.i)+","+str(org.j)
            #print "goal: "+str(org.choice)
            temp1 = org.i
            temp2 = org.j
            x,y = org.choice
            org.choice = 0
            if x-org.i < 0:
                if org.reporter:
                    print "Reporter is moving up."
                org.i = int(round((x-org.i)*org.speed+org.i))
            else:
                if org.reporter:
                    print "Reporter is moving down."
                org.i = int(round((x-org.i)*org.speed+org.i))
            if y-org.j < 0:
                if org.reporter:
                    print "Reporter is moving left."
                org.j = int(round((y-org.j)*org.speed+org.j))
            else:
                if org.reporter:
                    print "Reporter is moving right."
                org.j = int(round((y-org.j)*org.speed+org.j))
            while org.i < 0:
                org.i = org.i + len(thefield.space)
            while org.i >= len(thefield.space):
                org.i = org.i - len(thefield.space)
            while org.j < 0:
                org.j = org.j + len(thefield.space[0])
            while org.j >= len(thefield.space[0]):
                org.j = org.j - len(thefield.space[0])
            #print "after moving: "+str(org.i)+","+str(org.j)
            thefield.space[org.i][org.j].append(org)
            thefield.space[temp1][temp2].remove(org)

### Different interaction outcomes

def compete(interactors,org_list,thefield):
#   print "Competing..."
    apts = []
    for org in interactors:
        apts.append(org.apt)
    strongest = np.argmax(apts)
    winner = interactors[strongest]
    if winner.diet == 1 or winner.diet == 2:
        for food in interactors:
            if food is not winner: # Code ensures that if there is only one interactor, it does nothing.
                #if np.random.random() >= food.speed: # Escape function
                    #if interactors[strongest].reporter:
                    #   print "Reporter will eat "+food.i+","+food.j
                winner.eat(food)
            if food.vital <= 0:
                food.die(thefield,org_list)
                print "We have a predation!"
    plnt = thefield.resource_map[winner.i][winner.j]
    if (winner.diet == 0 or winner.diet == 2) and isinstance(plnt,plant): # Add sharing feature later
#       print "vital before: "+str(winner.vital)
        plnt = thefield.resource_map[winner.i][winner.j]
        winner.eat(plnt)
        if plnt.amount <= 0:
            plnt.die(thefield)
#       print "vital after: "+str(winner.vital)
        for loser in interactors:
            if loser is not winner:
                loser.vital -= mt.ceil(winner.apt*0.5)
            if loser.vital <= 0:
                loser.die(thefield,org_list)
                print "We have a kill!"

def speciation(org1,org2):
    if org1.genes == org2.genes:
        return True
    else:
        return False

def crossover(org1,org2,org_list,thefield):
    genelist = []
    for i in xrange(np.max([len(org1.genes),len(org2.genes)])):
        if np.random.random() <= 0.5:
            if i >= len(org1.genes):
                genelist.append(org2.genes[i])
            else:
                genelist.append(org1.genes[i])
        else:
            if i >= len(org2.genes):
                genelist.append(org1.genes[i])
            else:
                genelist.append(org2.genes[i])
    genome = ''
    for gene in genelist:
        genome += '*'+gene.code
    genome = list(genome)
    sorted = genesorter(genelist)
    org = organism(genelist,genome,org1.i,org1.j,False,sorted)
    org_list.append(org)
    org.parent.append(org1)
    org.parent.append(org2)
    org1.children.append(org)
    org2.children.append(org)
    thefield.space[org1.i][org1.j].append(org)

def reproduction(interactors,org_list,thefield):
    males = [[],[]] # Courtship
    females = []
    for org in interactors:
        if org.gender == 1:
            males[0].append(org)
            males[1].append(org.fert)
        else:
            females.append(org)
    if len(males[0]) > 0 and len(females) > 0:
        handsomest = males[0][np.argmax(males[1])]
    #   print str(len(males)),str(len(females))
        for female in females:
            if speciation(handsomest,female):
                if np.random.random() < female.fert and len(interactors) <= 10:# Crossover stage
                    crossover(handsomest,female,org_list,thefield)
                    print female.fert, handsomest.fert
#                   print "ORGANISM BORN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

#def trait_report(thefield):

def interact(org_list,thefield):
    #print "org_list size: "+str(len(org_list)) may wind up smaller than checklist size because organisms died off.
    for i in xrange(len(thefield.space)):
        for j in xrange(len(thefield.space[0])):
            interactors = thefield.space[i][j]
            if len(interactors) > 0:
        #       print "interactor size: "+str(len(interactors))
        #       if len(interactors) > 5: # Overpopulation function
        #           for org in interactors:
        #               org.vital -= 10
                compete(interactors,org_list,thefield)
                if len(interactors) <= 10:
                    reproduction(interactors,org_list,thefield)
    #print "checklist size: "+str(len(checklist))

def overpopulation(thefield,org_list):
    for i in xrange(len(thefield.space)):
        for j in xrange(len(thefield.space[0])):
            while len(thefield.space[i][j]) > 10:
                org = thefield.space[i][j][-1]
                org.die(thefield,org_list)

def molecular_clock(org_list,mutation,gene_pool):
    if np.random.random() < mutation:
        for org in org_list:
            if np.random.random() < mutation*10:
                snp_ind = np.random.randint(len(org.genome))
                snp = rd.choice(['*','0','1'])
                org.genome[snp_ind] = snp
                cistron = ''
                tss = False
                genelist = []
                cistrons = []
                for i,base in enumerate(org.genome):
                    if tss == True and base == '*':
                        if len(cistron) != 0:
                            for gene in gene_pool:
                                if gene.code == cistron[1::] and cistron not in cistrons:
                                    if len(cistron)!=0:
                                        genelist.append(gene)
                                        cistrons.append(cistron)
                            cistron = ''
                            tss = False
                        else:
                            org.genome.pop(i)
                    if base == '*':
                        tss = True
                    cistron += base
                org.genes = genelist
                print "MUTATION!!!"

def age(thefield,org_list):
    for org in org_list:
        org.vital -= 5
        org.age -= 1
        if org.age <= 0 or org.vital <= 0:
            print "Someone has passed away of old age..."
            org.die(thefield,org_list)

def migration(thefield,org_list,maxorf,gene_pool):
    if max_migrate != 0:
        num_migrating = np.random.randint(max_migrate) # Set number of species migrating in
    else:
        num_migrating = 0
    for x in xrange(num_migrating):
        i = np.random.randint(len(thefield.space))
        j = np.random.randint(len(thefield.space[0]))
        genome,genelist = genomemaker(maxorf,gene_pool)
        sorted = genesorter(genelist)
        print "animal migration!"
        for y in xrange(numstart/num_species): # Set number of members per species.
            org = organism(genelist,genome,i,j,False,sorted)
            thefield.space[i][j].append(org)
            org_list.append(org)

def plotter(org_list,thefield,counter):
    organisms = [[],[]]
    plants = [[],[]]
    organisms_vitals = []
    plants_amounts = []
    for org in org_list:
        organisms[0].append(org.j+1)
        organisms[1].append(org.i+1)
        organisms_vitals.append(int(org.vital))
    for plnt in thefield.resource_list:
        plants[0].append(plnt.j+1)
        plants[1].append(plnt.i+1)
        plants_amounts.append(int(plnt.amount))
    plt.plot(organisms[0],organisms[1],'b.',plants[0],plants[1],'g*',markersize=7)
    for label,x,y in zip(organisms_vitals,organisms[0],organisms[1]):
        plt.annotate(label,xy=(x,y),size=5)
    for label,x,y in zip(plants_amounts,plants[0],plants[1]):
        plt.annotate(label,xy=(x,y),size=5)
    plt.axis([0,len(thefield.space[0])*1.01,0,len(thefield.space)*1.01])
    plt.title("The Field")
    plt.savefig(output+'/frame_'+str(counter)+'.png',dpi=100)
    plt.clf()

def reporter(org_list,thefield,era):
    '''
    Takes the field and returns a plot of it. For testing purposes to see if the moving on the torus is correct.
    '''
    plant_amount_total = 0
    plant_count_bitten = 0
    for plnt in thefield.resource_list:
        plant_amount_total += plnt.amount
        if plnt.bitten:
            plant_count_bitten += 1
    print str(plant_count_bitten)+" plants were bitten out of "+str(len(thefield.resource_list))+"."
    print "mean plant amount: "+str(np.mean(plant_amount_total))
    distance = 0
#   for i in xrange(len(org_list)-1):
#      for j in xrange(i+1,len(org_list)):
#          distance += np.sqrt((org_list[i].i-org_list[j].i)**2+(org_list[i].j-org_list[j].j)**2)
    species = {}
    species_time = {}
    reports = {}
    diet = {}
    brain = {}
    reports["vital"] = 0
    reports["speed"] = 0
    reports["fert"] = 0
    reports["apt"] = 0
    reports["sight"] = 0
    diet["herb"] = 0
    diet["carn"] = 0
    diet["omni"] = 0
    brain["dumb"] = 0
    brain["coward hunter"] = 0
    brain["brave hunter"] = 0
    brain["coward mater"] = 0
    brain["brave mater"] = 0
    for org in org_list:
        if str(org.sorted) not in species:
            species[str(org.sorted)] = 1
            species_time[str(org.sorted)] = era
        else:
            species[str(org.sorted)] += 1
        reports["vital"] += org.startvital/float(len(org_list))
        reports["speed"] += org.speed/float(len(org_list))
        reports["fert"] += org.fert/float(len(org_list))
        reports["apt"] += org.apt/float(len(org_list))
        reports["sight"] += org.sight/float(len(org_list))
        if org.diet == 0:
            diet["herb"] += 1/float(len(org_list))
        if org.diet == 1:
            diet["carn"] += 1/float(len(org_list))
        if org.diet == 2:
            diet["omni"] += 1/float(len(org_list))
        if org.brain == 0:
            brain["dumb"] += 1/float(len(org_list))
        if org.brain == 1:
            brain["coward hunter"] += 1/float(len(org_list))
        if org.brain == 2:
            brain["brave hunter"] += 1/float(len(org_list))
        if org.brain == 3:
            brain["coward mater"] += 1/float(len(org_list))
        if org.brain == 4:
            brain["brave mater"] += 1/float(len(org_list))
#   print "mean pairwise distance of organisms: "+str(distance/float(len(org_list)+0.01))
    numspecies = len(species)
    memberspecies = mt.ceil(np.mean(species.values()))
    numplants = len(thefield.resource_list)
    numorgs = len(org_list)
    print "number of species: "+str(numspecies)
    print "mean members per species: "+str(memberspecies)
    return numspecies,memberspecies,reports,numplants,numorgs,diet,brain,species,species_time

def coevolutioner(maxorf,row,col,resource,vital,apt,sight,generations,mutation,numstart):
    gene_pool = genepool(maxorf,vital,apt,sight)
    thefield,org_list = creation(maxorf,numstart,gene_pool,row,col,resource)
    plot_info = [[] for _ in xrange(19)]
    plot_iterate = 0
    for i in xrange(generations):
        grow_plants(thefield,resource)
        decide(thefield,org_list)
        move(thefield,org_list)
        overpopulation(thefield,org_list)
        interact(org_list,thefield) # Takes care of reproduction and competition
        num_interacted = 0
        molecular_clock(org_list,mutation,gene_pool)
        age(thefield,org_list)
        migration(thefield,org_list,maxorf,gene_pool)
        print "length of org_list: "+str(len(org_list))
        numspecies,memberspecies,reports,numplants,numorgs,diet,brain,species,species_time = reporter(org_list,thefield,i)
        plot_iterate+=1
        if len(org_list) > 0:
            plot_info[0].append(numspecies)
            plot_info[1].append(memberspecies)
            plot_info[2].append(reports["vital"])
            plot_info[3].append(reports["apt"])
            plot_info[4].append(reports["sight"])
            plot_info[5].append(reports["fert"])
            plot_info[6].append(reports["speed"])
            plot_info[7].append(numplants)
            plot_info[8].append(numorgs)
            plot_info[9].append(diet["herb"])
            plot_info[10].append(diet["carn"])
            plot_info[11].append(diet["omni"])
            plot_info[12].append(brain["dumb"])
            plot_info[13].append(brain["coward hunter"])
            plot_info[14].append(brain["brave hunter"])
            plot_info[15].append(brain["coward mater"])
            plot_info[16].append(brain["brave mater"])
            plot_info[17].append(species)
            plot_info[18].append(species_time)
        plotter(org_list,thefield,i)
        print "########GENERATION "+str(i)+"############"
    if len(org_list) > 0:
        plt.plot(range(plot_iterate),plot_info[0],'r',label = "Number of species")
        plt.plot(range(plot_iterate),plot_info[1],'b',label="Average Members per Species")
        plt.legend(bbox_to_anchor=(1,1),loc=1,prop={'size':8})
        plt.title("Number of Species and Average Number of Members per Species over Time.")
        plt.savefig(output+"/species_stats_plot.png",dpi=100)
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(range(plot_iterate),plot_info[2],'r', label="Vital")
        plt.plot(range(plot_iterate),plot_info[3],'b', label="Apt")
        plt.plot(range(plot_iterate),plot_info[4],'g', label="Sight")
        plt.legend(bbox_to_anchor=(1,1),loc=1,prop={'size':8})
        plt.subplot(2,1,2)
        plt.plot(range(plot_iterate),plot_info[5],'y', label="Fert")
        plt.plot(range(plot_iterate),plot_info[6],'m', label="Speed")
        plt.legend(bbox_to_anchor=(1,1),loc=1,prop={'size':8})
        plt.suptitle("Traits of Organisms over Time.")
        plt.savefig(output+"/traits_plot.png",dpi=100)
        plt.clf()
        plt.plot(range(plot_iterate),plot_info[7],'g',label="Number of plants")
        plt.plot(range(plot_iterate),plot_info[8],'r',label="Number of organisms")
        plt.title("Organisms vs Plants.")
        plt.legend(bbox_to_anchor=(1,1),loc=1,prop={'size':8})
        plt.savefig(output+"/org_v_plt.png",dpi=100)
        plt.clf()
        plt.plot(range(plot_iterate),plot_info[9],'g',label="Herbivores")
        plt.plot(range(plot_iterate),plot_info[10],'r',label="Carnivores")
        plt.plot(range(plot_iterate),plot_info[11],'b',label="Omnivores")
        plt.title("Diets over Time.")
        plt.legend(bbox_to_anchor=(1,1),loc=1,prop={'size':8})
        plt.savefig(output+"/diets.png",dpi=100)
        plt.clf()
        plt.plot(range(plot_iterate),plot_info[12],'k',label="Dumb")
        plt.plot(range(plot_iterate),plot_info[13],'b',label="Coward Hunter")
        plt.plot(range(plot_iterate),plot_info[14],'r',label="Brave Hunter")
        plt.plot(range(plot_iterate),plot_info[15],'g',label="Coward Mating")
        plt.plot(range(plot_iterate),plot_info[16],'m',label="Brave Mating")
        plt.title("Prevalent Strategy over Time.")
        plt.legend(bbox_to_anchor=(1,1),loc=1,prop={'size':8})
        plt.savefig(output+"/brains.png",dpi=100)
        plt.clf()
        y_values = {}
        x_values = {}
        for i,input in enumerate(plot_info[17]):
            for key in input.keys():
                if key in y_values:
                    y_values[key].append(input[key])
                    x_values[key].append(plot_info[18][i][key])
                else:
                    y_values[key] = [input[key]]
                    x_values[key] = [plot_info[18][i][key]]
        cmap = mpl.cm.autumn
        for i,key in enumerate(x_values.keys()):
            plt.plot(x_values[key],y_values[key],c=cmap(i/float(plot_iterate)))
        plt.title("Number of Members in Each Species over Time.")
        plt.savefig(output+"/species_progression_plot.png")
        plt.clf()
    else:
        print "Worldwide Extinction Event."

coevolutioner(maxorf,map_row,map_col,resource_size,vital_size,apt_size,sight_size,generations,mutation,numstart)

