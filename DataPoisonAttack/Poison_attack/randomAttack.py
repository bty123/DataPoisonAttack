#coding:utf-8
#author:Yu Junliang
import random

import numpy as np
from attack import Attack

class RandomAttack(Attack):
    def __init__(self,conf):
        super(RandomAttack, self).__init__(conf)


    def insertSpam(self,startID=0):
        print('Modeling random attack...')
        itemList = self.itemProfile.keys()
        if startID == 0:
            self.startUserID = len(self.userProfile)
        else:
            self.startUserID = startID

        for i in range(int(len(self.userProfile)*self.attackSize)):
            #fill 装填项目
            fillerItems = self.getFillerItems()
            for item in fillerItems:
                # RT
                pareto = random.paretovariate(2.0) - 1
                if pareto > 20.0:
                    pareto = format(random.uniform(15.0, 20.0), '.3f')
                self.spamProfile[str(self.startUserID)][str(itemList[item])] = round(float(pareto), 3)

                # TP
                # pareto = random.paretovariate(0.5) - 1
                # if pareto > 1000.0:
                #     pareto = format(random.uniform(500.0, 1000.0), '.3f')
                # self.spamProfile[str(self.startUserID)][str(itemList[item])] = round(float(pareto), 3)

                # normal
                # self.spamProfile[str(self.startUserID)][str(itemList[item])] = format(random.uniform(self.minScore,self.maxScore), '.3f')

            #target 目标项目
            # np.random.seed(11)
            for j in range(self.targetCount):

                target = np.random.randint(len(self.targetItems))
                # print(target)
                # self.spamProfile[str(self.startUserID)][self.targetItems[target]] = self.targetScore
                target_mu = 5
                target_sigma = 7
                while True:
                    a = np.random.normal(target_mu, target_sigma)
                    if a < target_mu:
                        a = 2 * target_mu - a
                    if a < 20:
                        break
                self.spamProfile[str(self.startUserID)][self.targetItems[target]] = format(a, '.3f')
                # self.spamProfile[str(self.startUserID)][self.targetItems[target]] = format(random.uniform(self.targetScore - 5.0, self.targetScore + 5.0), '.3f')
                # self.spamProfile[str(self.startUserID)][self.targetItems[target]] = format(random.uniform(self.targetScore - 100.0, self.targetScore + 100.0), '.3f')
                self.spamItem[str(self.startUserID)].append(self.targetItems[target])
            self.startUserID += 1
