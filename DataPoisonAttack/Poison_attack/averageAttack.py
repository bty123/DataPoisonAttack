#coding:utf-8
#author:Yu Junliang
#Date: 2016-03-15
import random

import numpy as np
from attack import Attack


class AverageAttack(Attack):
    def __init__(self,conf):
        super(AverageAttack, self).__init__(conf)


    def insertSpam(self,startID=0):
        print('Modeling average attack...')
        itemList = self.itemProfile.keys()
        if startID == 0:
            self.startUserID = len(self.userProfile)
        else:
            self.startUserID = startID

        for i in range(int(len(self.userProfile)*self.attackSize)):
            #fill 装填项目
            fillerItems = self.getFillerItems()
            for item in fillerItems:
                self.spamProfile[str(self.startUserID)][str(itemList[item])] = round(self.itemAverage[str(itemList[item])],3)
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
                        a = 2*target_mu - a
                    if a < 20:
                        break
                self.spamProfile[str(self.startUserID)][self.targetItems[target]] = format(a, '.3f')
                # self.spamProfile[str(self.startUserID)][self.targetItems[target]] = format(random.uniform(self.targetScore - 5.0, self.targetScore + 5.0), '.3f')
                # self.spamProfile[str(self.startUserID)][self.targetItems[target]] = format(random.uniform(self.targetScore - 100.0, self.targetScore + 100.0), '.3f')
                self.spamItem[str(self.startUserID)].append(self.targetItems[target])
            self.startUserID += 1






