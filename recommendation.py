#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import codecs
import random
import math
import wx
import operator
import sys


# 读文件u.user,返回一个有关的统计信息给用户，包含943个用户，有4个属性

def createUserList():
    print("createUserList")
    userList = list()

    filename = 'ml-100k/u.user'
    with open(filename, 'r') as f:
        records = f.readlines()

    for i in xrange(len(records)):
        userList.append(dict())

    for r in records:
        record = (r.strip()).split('|')
        userList[int(record[0])-1]['age'] = int(record[1])
        userList[int(record[0])-1]['gender'] = record[2]
        userList[int(record[0])-1]['occupation'] = record[3]
        userList[int(record[0])-1]['zip'] = record[4]

    return userList


# Read from the file u.item and return a list containing all of the
# information pertaining to positions given in the file.
# Then positionList contains as many elements as there are positions and information
# pertaining to the position with ID i should appear in slot i-1 in positionList.
#读取职位文件‘u.item’，里面包换职位列表
def createPositionList():
    print("createPositionList")
    positionList = list()

    filename = 'ml-100k/guangzhou.item'
    with open(filename, 'r') as f:
        records = f.readlines()

    for i in xrange(len(records)):
        positionList.append(dict())

    for r in records:
        record = (r.strip()).split(' ')
        positionList[int(record[0])-1]['title'] = record[1]
        positionList[int(record[0])-1]['release date'] = record[2]
        positionList[int(record[0])-1]['"position release date'] = record[6]
        positionList[int(record[0])-1]['need'] = record[10]
        positionList[int(record[0])-1]['genre'] = list()
        for n in range(5, 24):
            positionList[int(record[0])-1]['genre'].append(int(record[n]))

    return positionList


# Read ratings from the file u.data and return a list of 100,000 length-3
# tuples of the form (user, position, rating).
#读取评分列表，分别微微用户，职位，频分
def readRatings():
    print("readRatings")
    ratingList = list()

    filename = path
    with open(filename, 'r') as f:
        records = f.readlines()

    for r in records:
        record = (r.strip()).split('\t')
        ratingList.append((int(record[0]), int(record[1]), int(record[2])))

    return ratingList


# The function takes the rating tuple list constructed by readRatings and
# organizes the tuples in this list into two data structures: rLu and rLm.
# The list rLu is a list, with one element per user, of all the ratings provided by each user.
# The list rLm is a list, with one element per position, of all the ratings received by each position.
# The ratings provided by user with ID i should appear in slot i-1 in rLu, as a dictionary
# whose keys are IDs of positions that this user has rated and whose values are corresponding ratings.
# The list rLm is quite similar.
#rLu里面存储了每个用户的评分，rLm里面存储了每个职位的评分
def createRatingsDataStructure(numUsers, numItems, ratingTuples):
    print("createRatingsDataStructure")
    rLu = list()
    rLm = list()

    #for i in xrange(numUsers):
    #    rLu.append(dict())
    rLu = [{} for i in xrange(numUsers)]

    #for i in xrange(numItems):
    #    rLm.append(dict())
    rLm = [{} for i in xrange(numItems)]

    for r in ratingTuples:
        rLu[int(r[0]-1)][int(r[1])] = int(r[2])
        rLm[int(r[1]-1)][int(r[0])] = int(r[2])

    for user in rLu:
        try:
            m = sum(user.values())/float(len(user))
            user['mean'] = m
            max_common_position = 0
            for v in rLu:
                t = len(set(user.keys()) & set(v.keys()))
                if  t > max_common_position:
                    if v is not user:
                        max_common_position = t
            user['max_common'] = max_common_position
        except ZeroDivisionError:
            user['mean'] = 0.
            user['max_common'] = 0
    for position in rLm:
        try:
            m = sum(position.values())/float(len(position))
            position['mean'] = m
            max_common_user = 0
            for n in rLm:
                t = len(set(position.keys()) & set(n.keys()))
                if t > max_common_user:
                    if n is not position:
                        max_common_user = t
            position['max_common'] = max_common_user
        except ZeroDivisionError:
            position['mean'] = 0.
            position['max_common'] = 0

    return [rLu, rLm]


# Read from the file u.genre and returns the list of position genres listed in the file.
# The genres appears in the order in which they are listed in the file.
#读题材列表
def createGenreList():
    print("createGenreList")
    genreList = list()

    filename = 'u.genre'
    with open(filename, 'r') as f:
        records = f.readlines()

    for r in records:
        record = (r.strip()).split('|')
        if record[0]:
            genreList.append(record[0])

    return genreList


# Return the mean rating provided by user with given ID u.
#返回用户提供的评分预测
def meanUserRating(u, rLu):
    print("meanUserRating")
    if len(rLu[u-1])!= 0:
        mean_rating = rLu[u-1]['mean']
    else:
        mean_rating = 0.0

    return mean_rating


# Return the mean rating for a position with given ID m.
#返回职位提供的评分预测
def meanPositionRating(m, rLm):
    print("meanPositionRating")
    if len(rLm[m-1])!= 0:
        mean_rating = rLm[m-1]['mean']
    else:
        mean_rating = 0.0

    return mean_rating


# Given a user u and a position m, simply return a random integer rating in the range [1, 5].
#给一个用户u和职位m随机的返回[1, 5]的评分
def randomPrediction(u, m):
    print("randomPrediction")
    rating = random.randint(1,5)
    return rating


# Given a user u and a position m, simply return the mean rating that user u has given to positions.
# Here userRatings is a list with one element per user, each element being a dictionary
# containing all position-rating pairs associated with that user.
#给一个用户u和一个电影m，简单的返回用户给职位的评分
def meanUserRatingPrediction(u, m, userRatings):
    print("meanUserRatingPrediction")
    rating = meanUserRating(u, userRatings)
    return rating


# Given a user u and a position m, simply return the mean rating that position m has received.
# Here positionRatings is a list with one element per position, each element being a dictionary
# containing all user-rating pairs associated with that user.
#返回电影m接受的平均等级
def meanPositionRatingPrediction(u, m, positionRatings):
    print("meanPositionRatingPrediction")
    rating = meanPositionRating(m, positionRatings)
    return rating


# Given a user u and a position m, simply return the average of the mean rating
# that u gives and mean rating that m receives.
#给定用户U和电影M，简单地返回u给出的平均评级的平均值和M接收的平均评级。
def meanRatingPrediction(u, m, userRatings, positionRatings):
    print("meanRatingPrediction")
    rating_1 = meanUserRating(u, userRatings)
    rating_2 = meanPositionRating(m, positionRatings)
    rating = (rating_1 + rating_2)/2
    return rating


# The function partitions ratings into a training set and a testing set.
# The testing set is obtained by randomly selecting the given percent of the raw ratings.
# The remaining unselected ratings are returned as the training set.
#函数将评分划分为训练集和测试集。测试集是通过随机选择给定百分比的原始评级获得的。剩余的未选择的评级作为训练集返回。
def partitionRatings(rawRatings, testPercent):
    print("partitionRatings")
    trainingSet = list(rawRatings)
    testSet = list()

    test_size = int(len(rawRatings) * testPercent)

    while len(testSet) < test_size:
        index = random.randrange(len(trainingSet))
       # index=1
        global redex
        redex= index
        testSet.append(trainingSet.pop(index))

    return [trainingSet, testSet]


# The function computes the RMSE given lists of actual and predicted ratings.
# RMSE is computed by first taking the mean of the squares of differences between actual
# and predicted ratings and then taking the square root of this quantity.
#函数计算RMSE给出的实际和预测的额定值列表。首先通过取实际值之间的平方的平均值来计算RMSE。和预测的评级，然后采取这个量的平方根。
def rmse(actualRatings, predictedRatings):
    print("rmse")
    summation = 0.0
    length = len(actualRatings)

    for i in xrange(length):
        summation += (actualRatings[i] - predictedRatings[i]) ** 2

    rmse_value = math.sqrt(summation/length)
    return rmse_value


# The function computes the similarity in ratings between the two users, using the
# positions that the two users have commonly rated. The similarity between two users
# will always be between -1 and +1.
#计算两个用户的相似性加一或者减一
def similarity(u, v, userRatings):
    print("similarity")
    sim = 0.0
    sum_1 = sum_2 =sum_3 = 0.0

    # Find the positions that the two users have commonly rated.
    #查找对于同一个职位两个用户的相同评分
    common_position = list(set(userRatings[u-1].keys()) & set(userRatings[v-1].keys()))

    if len(common_position) == 0:
        return sim

    # Compute the mean ratings that user u or user v has given to positions.
    #计算用户U或用户V给电影的平均收视率。
    mean_rating_1 = meanUserRating(u, userRatings)
    mean_rating_2 = meanUserRating(v, userRatings)

    # Compute the similarity in ratings between the two users.
    for i in common_position:
        sum_1 += (userRatings[u-1][i] - mean_rating_1) * (userRatings[v-1][i] - mean_rating_2)
        sum_2 += (userRatings[u-1][i] - mean_rating_1) ** 2
        sum_3 += (userRatings[v-1][i] - mean_rating_2) ** 2
    if sum_1 == 0 or sum_2 == 0 or sum_3 == 0:
        return sim
    else:
        sim = sum_1 / ( math.sqrt(sum_2) * math.sqrt(sum_3) )
        return sim


# The function returns the list of (user ID, similarity)-pairs for the k users
# who are most similar to user u. The user u herself will be excluded from
# candidates being considered by this function.
def kNearestNeighbors(u, userRatings, k):
    print("kNearestNeighbors")
    sim_user_list = list() # a list of (similarity, user ID)-pairs
    user_sim_list = list() # a list of (user ID, similarity)-pairs

    for i in xrange(len(userRatings)):
        sim = 0
        if i != u-1:
            sim = similarity(u, i+1, userRatings)
            sim_user_list.append((sim, i+1))

    sim_user_list.sort()
    sim_user_list.reverse()

    if k == -1:
        return sim_user_list

    for i in sim_user_list[:k]:
        user_sim_list.append((i[1], i[0]))

    return user_sim_list


# calculate the KNN using pre-calculated similarity
def kNearestNeighbors2(u, userRatings, k, similarity_m):
    print("kNearestNeighbors2")
    sim_user_list = list() # a list of (similarity, user ID)-pairs
    user_sim_list = list() # a list of (user ID, similarity)-pairs

    for i in xrange(len(userRatings)):
        sim = 0
        if i != u-1:
            sim = similarity_m[u][i+1]
            sim_user_list.append((sim, i+1))

    sim_user_list.sort()
    sim_user_list.reverse()

    if k == -1:
        return sim_user_list

    for i in sim_user_list[:k]:
        user_sim_list.append((i[1], i[0]))

    return user_sim_list

# The function predicts a rating by user u for position m.
# It uses the ratings of the list of friends to come up with a rating by u of m according to formula.
# The argument corresponding to friends is computed by a call to the kNearestNeighbors function.
def CFRatingPrediction(u, m, userRatings, friends):
    print("CFRatingPrediction")
    # Construct a dictionary, whose keys are userID and
    # whose values are (similarity, mean_rating)-pairs.
    user_dict = dict()

    for i in friends:
        user_dict.setdefault(i[0], list())
        user_dict[i[0]].append(i[1])
        mean_rating = meanUserRating(i[0], userRatings)
        user_dict[i[0]].append(mean_rating)

    rating = meanUserRating(u, userRatings)
    sum_1 = 0.0
    sum_2 = 0.0

    for i in friends:
        if m in userRatings[i[0]-1]:
            sum_1 += (userRatings[i[0]-1][m] - user_dict[i[0]][1]) * user_dict[i[0]][0]
            sum_2 += abs(user_dict[i[0]][0])

    if sum_2 == 0:
        return rating
    else:
        rating += sum_1 / sum_2
        return rating


# The function computes a number using the formula (which is same to the function CFRatingPrediction),
# and then returns the average of this and mean rating of position m.
def CFMMRatingPrediction(u, m, userRatings, positionRatings, friends):
    print("CFMMRatingPrediction")
    rating = CFRatingPrediction(u, m, userRatings, friends)
    mean_rating_m = meanPositionRating(m, positionRatings)
    average_rating = (rating + mean_rating_m) / 2
    return average_rating

# return both the return value of CFMMRatingPrediction and CFMMRatingPrediction
def CFMMRatingPrediction2(u, m, userRatings, positionRatings, friends):
    print("CFMMRatingPrediction2")
    rating = CFRatingPrediction(u, m, userRatings, friends)
    mean_rating_m = meanPositionRating(m, positionRatings)
    average_rating = (rating + mean_rating_m) / 2
    return rating, average_rating

# The function computes the max number of positions that user u and another user have commonly rated.
# First, compute the number of positions that user u and each user in all-users have commonly rated.
# Second, find the max of all the numbers.
def maxCommonPosition(u, userRatings):
    print("maxCommonPosition")
    common_position_list = list()

    #length = len(userRatings)
    for i, v in enumerate(userRatings):
        if i != u-1:
            #common_position = 0
            #for position in userRatings[u-1]:
            #    if position in userRatings[i]:
            #        common_position += 1
            common_position = len(set(userRatings[u-1]) & set(v))
            common_position_list.append(common_position)

    #common_position_list.sort()
    #max_common_position = common_position_list[-1]
    return max(common_position_list)


# The function computes the new similarity in ratings between the two users, using the
# positions that the two users have commonly rated.
# The new similarity between two users will always be between 0.36788 and 1.
def userSimilarity(u, v, userRatings):
    print("userSimilarity")
    #common_position = 0.0
    max_common = maxCommonPosition(u, userRatings)

    # Find the positions that the two users have commonly rated.
    #for i in userRatings[u-1]:
    #    if i in userRatings[v-1]:
    #       common_position += 1
    common_position = float(len(set(userRatings[u-1]) & set(userRatings[v-1])))

    sim = similarity(u, v, userRatings)
    user_sim = math.exp(common_position/max_common - 1) * sim
    return user_sim

# calculate the userSimilarity using pre-calculated similarity and maxCommonUser
def userSimilarity2(u, v, userRatings, similarity_m):
    print("userSimilarity2")
    max_common = userRatings[u-1]['max_common']

    common_position = float(len(set(userRatings[u-1].keys()) & set(userRatings[v-1].keys())))

    sim = similarity_m[u][v]
    user_sim = sim > 0 and math.exp(common_position/max_common - 1) * sim or 0
    return user_sim


# The function returns the list of (user ID, similarity)-pairs for the k users
# who are most similar to user u. The user u herself will be excluded from
# candidates being considered by this function.
# The new similarity between users is computed by a call to the userSimilarity function.
def userKNearestNeighbors(u, userRatings, k):
    print("userKNearestNeighbors")
    sim_user_list = list() # a list of (similarity, user ID)-pairs
    user_sim_list = list() # a list of (user ID, similarity)-pairs

    for i in xrange(len(userRatings)):
        sim = 0
        if i != u-1:
            sim = userSimilarity(u, i+1, userRatings)
            sim_user_list.append((sim, i+1))

    sim_user_list.sort()
    sim_user_list.reverse()

    for i in sim_user_list[:k]:
        user_sim_list.append((i[1], i[0]))

    return user_sim_list

# calculate the userKNN using pre-calculated similarity
def userKNearestNeighbors2(u, userRatings, similarity_m):
    print("userKNearestNeighbors2")
    sim_user_list = list() # a list of (similarity, user ID)-pairs
    user_sim_list = list() # a list of (user ID, similarity)-pairs

    for i in xrange(len(userRatings)):
        sim = 0
        if i != u-1:
            sim = userSimilarity2(u, i+1, userRatings, similarity_m)
            sim_user_list.append((sim, i+1))

    sim_user_list.sort()
    sim_user_list.reverse()

    return sim_user_list

# The function predicts a rating by user u for position m.
# It uses the ratings of the list of friends to come up with a rating by u of m according to formula.
# The argument corresponding to friends is computed by a call to the userKNearestNeighbors function.
def userCFRatingPrediction(u, m, userRatings, friends):
    print("userCFRatingPrediction")
    rating = CFRatingPrediction(u, m, userRatings, friends)
    return rating


# This function computes the maximum value of the numbers of users who have both
# rated position i and another position which is in position set.
# First, compute the number of users that position m and each position in all-positions have been commonly rated by.
# Second, find the max of all the numbers.
def maxCommonUser(m, positionRatings):
    print("maxCommonUser")
    common_user_list = list()

    for i, n in enumerate(positionRatings):
        if i != m-1:
            common_user = len(set(positionRatings[m-1]) & set(n))
            common_user_list.append(common_user)

    #common_user_list.sort()
    #max_common_user = common_user_list[-1]
    return max(common_user_list)


# The function computes the similarity in ratings between the two positions, using the
# users who have commonly rated the two positions. The similarity between two positions
# will always be between -1 and +1.
def similarity_2(m, n, positionRatings):
    print("similarity_2")
    sim = 0.0
    sum_1 = sum_2 =sum_3 = 0.0

    # Find the users that have commonly rated the same two positions.
    common_user = list(set(positionRatings[m-1].keys()) & set(positionRatings[n-1].keys()))

    if len(common_user) == 0:
        return sim

    # Compute the mean ratings that position m or position n has been given to.
    mean_rating_1 = meanPositionRating(m, positionRatings)
    mean_rating_2 = meanPositionRating(n, positionRatings)

    # Compute the similarity in ratings between the two positions.
    for i in common_user:
        sum_1 += (positionRatings[m-1][i] - mean_rating_1) * (positionRatings[n-1][i] - mean_rating_2)
        sum_2 += (positionRatings[m-1][i] - mean_rating_1) ** 2
        sum_3 += (positionRatings[n-1][i] - mean_rating_2) ** 2
    if sum_1 == 0 or sum_2 == 0 or sum_3 == 0:
        return sim
    else:
        sim = sum_1 / ( math.sqrt(sum_2) * math.sqrt(sum_3) )
        return sim


# The function computes the new similarity in ratings between the two positions, using the
# users who have commonly rated the two positions.
# The new similarity between two positions will always be between 0.36788 and 1.
def positionSimilarity(m, n, positionRatings):
    print("positionSimilarity")
    #common_user = 0.0
    max_common = maxCommonUser(m, positionRatings)

    # Find the users that have commonly rated the same two positions.
    #for i in positionRatings[m-1]:
    #    if i in positionRatings[n-1]:
    #       common_user += 1
    common_user = float(len(set(positionRatings[m-1]) & set(positionRatings[n-1])))

    sim = similarity_2(m, n, positionRatings)
    position_sim = math.exp(common_user/max_common - 1) * sim
    return position_sim

# calculate the positionSimilarity using pre-calculated similarity and maxCommonPosition
def positionSimilarity2(m, n, positionRatings, similarity_m_2):
    print("positionSimilarity2")
    max_common = positionRatings[m-1]['max_common']

    common_user = float(len(set(positionRatings[m-1]) & set(positionRatings[n-1])))

    sim = similarity_m_2[m][n]
    position_sim = sim > 0 and math.exp(common_user/max_common - 1) * sim or 0
    return position_sim

# The function returns the list of (position ID, similarity)-pairs for the k positions
# which are most similar to position m. The position m itself will be excluded from
# candidates being considered by this function.
# The new similarity between positions is computed by a call to the positionSimilarity function.
def positionKNearestNeighbors(m, positionRatings, k):
    print("positionKNearestNeighbors")
    sim_position_list = list() # a list of (similarity, position ID)-pairs
    position_sim_list = list() # a list of (position ID, similarity)-pairs

    for i in xrange(len(positionRatings)):
        sim = 0
        if i != m-1:
            sim = positionSimilarity(m, i+1, positionRatings)
            sim_position_list.append((sim, i+1))

    sim_position_list.sort()
    sim_position_list.reverse()

    for i in sim_position_list[:k]:
        position_sim_list.append((i[1], i[0]))

    return position_sim_list

# calculate the positionKNN using pre-calculated similarity
def positionKNearestNeighbors2(m, positionRatings, similarity_m):
    print("positionKNearestNeighbors2")
    sim_position_list = list() # a list of (similarity, position ID)-pairs
    position_sim_list = list() # a list of (position ID, similarity)-pairs

    for i in xrange(len(positionRatings)):
        sim = 0
        if i != m-1:
            sim = positionSimilarity2(m, i+1, positionRatings, similarity_m)
            sim_position_list.append((sim, i+1))

    sim_position_list.sort()
    sim_position_list.reverse()

    return sim_position_list


# The function predicts a rating by user u for position m.
# It uses the ratings of the list of friends to come up with a rating of m by u according to formula.
# The argument corresponding to friends is computed by a call to the positionKNearestNeighbors function.
def positionCFRatingPrediction(u, m, positionRatings, friends):
    print("Construct a dictionary, whose keys are positionID and")
    # Construct a dictionary, whose keys are positionID and
    # whose values are (similarity, mean_rating)-pairs.
    position_dict = dict()

    for i in friends:
        position_dict.setdefault(i[0], list())
        position_dict[i[0]].append(i[1])
        mean_rating = meanPositionRating(i[0], positionRatings)
        position_dict[i[0]].append(mean_rating)

    rating = meanPositionRating(m, positionRatings)
    sum_1 = sum_2 = 0.0

    for i in friends:
        if u in positionRatings[i[0]-1]:
            sum_1 += (positionRatings[i[0]-1][u] - position_dict[i[0]][1]) * position_dict[i[0]][0]
            sum_2 += abs(position_dict[i[0]][0])

    if sum_2 == 0:
        return rating
    else:
        rating += sum_1 / sum_2
        return rating

def appoint_line(num,file):
    f=codecs.open(file,'r','utf-8')
    out = f.readlines()[num-1]
    return out

# This function returns the average of predicted ratings computed by userCFRatingPrediction function
# and positionCFRatingPrediction function.
def averageCFRatingPrediction(u, m, userRatings, positionRatings, friends_user, friends_position):
    rating_user = userCFRatingPrediction(u, m, userRatings, friends_user)
    rating_position = positionCFRatingPrediction(u, m, positionRatings, friends_position)
    average_rating = (rating_user + rating_position) / 2
    return average_rating


# The function evaluates each prediction algorithm by using an 80-20 split
# of the ratings into training and testing sets.
# To make sure that the reported rmse values are reliable, the process will be
# performed 10 repetitions. The average rmse value of each prediction algorithm
# (averaged over the 10 repetitions) will be written into a file "output_ec.txt".
def averageTenRuns():
 dicresult={}
 for re in xrange(10):
    numUsers = 943
    numItems = 1682
    numRatings = 1000
    testPercent = 0.2

    ratingList = readRatings()

    times = 2
    value_rmse = [[0.] * times for i in range(19)]

    for time in xrange(times): # repeat two times
        print("repeat two times")
        [trainingSet, testSet] = partitionRatings(ratingList, testPercent)
        [rLu, rLm] = createRatingsDataStructure(numUsers, numItems, trainingSet)

        actualRatings = list()
        predictedRatings = list()
        for i in xrange(5):
            predictedRatings.append(list())

        users = sorted(set([e[0] for e in testSet]))
        similarity_m = [[0.]*(numUsers+1) for i in range(numUsers+1)]
        positions = sorted(set([e[1] for e in testSet]))
        similarity_m_2 = [[0.]*(numItems+1) for i in range(numItems+1)]

        for n, u in enumerate(users):
            for v in users[n+1:]:
                s = similarity(u, v, rLu)
                similarity_m[u][v] = s
                similarity_m[v][u] = s

        for i, m in enumerate(positions):
            for n in positions[i+1:]:
                s = similarity_2(m, n, rLm)
                similarity_m_2[m][n] = s
                similarity_m_2[n][m] = s
                #sys.stdout.write("\r{}_{}".format(m, n))
                #sys.stdout.flush()

        for n, i in enumerate(testSet):
            u, m, r = i
            actualRatings.append(r)

            k_values = [(0, 0), (25, 25), (300, 300), (500, 500), (numUsers, numItems)]
            friends_user = userKNearestNeighbors2(u, rLu, similarity_m)
            friends_position = positionKNearestNeighbors2(m, rLm, similarity_m_2)

            for nk, k in enumerate(k_values):

                friends_user_k = [(e[1], e[0]) for e in friends_user[:k[0]]]
                friends_position_k = [(e[1], e[0]) for e in friends_position[:k[1]]]

                predictedRatings[nk].append(averageCFRatingPrediction(u, m, rLu, rLm, friends_user_k, friends_position_k))
            #sys.stdout.write("\r{}_{}".format(time, n))
            #sys.stdout.flush()

        for i in xrange(5):
            value_rmse[i][time] = rmse(actualRatings, predictedRatings[i])

    # compute the average rmse value of each prediction algorithm
    # (averaged over the 10 repetitions)
    average_rmse = [sum(v)/times for v in value_rmse]
    dicresult[redex]=average_rmse[4]
    print("out")
    with open('ml-100k/output_ec.txt', 'w') as f:
        f.write("New Collaborative Filtering Rating prediction Average RMSE (friends = 0): " + str(average_rmse[0]) + "\n")
        f.write("New Collaborative Filtering Rating prediction Average RMSE (friends = 25): " + str(average_rmse[1]) + "\n")
        f.write("New Collaborative Filtering Rating prediction Average RMSE (friends = 300): " + str(average_rmse[2]) + "\n")
        f.write("New Collaborative Filtering Rating prediction Average RMSE (friends = 500): " + str(average_rmse[3]) + "\n")
        f.write("New Collaborative Filtering Rating prediction Average RMSE (friends = all): " + str(average_rmse[4]) + "\n")
 dicresult=sorted(dicresult.items(), key=lambda x: x[1], reverse=True)
 with open('ml-100k/result.txt','w') as f:
     f.write(str(dicresult)+"\n")
 with codecs.open('ml-100k/text.txt', 'w','utf-8') as f:
     for i in range(len(dicresult)):
          f.write(appoint_line(dicresult[i][0],'ml-100k/guangzhou.csv')+"\n")
# Main program
wildcard1 = "All files (*.*)|*.*|" \
            "Python source (*.py; *.pyc)|*.py;*.pyc"
wildcard2 = "Python source (*.py; *.pyc)|*.py;*.pyc|" \
            "All files (*.*)|*.*"
class ButtonFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, '职位推荐系统',
                          size=(1500, 600))

        panel = wx.Panel(self, -1)
        self.button = wx.Button(panel, -1, "推荐职位", pos=(500, 60))
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.button)
        self.button.SetDefault()
        self.button1 = wx.Button(panel, -1, "训练模型",  pos=(300, 60))
        self.Bind(wx.EVT_BUTTON, self.OnClick1, self.button1)
        self.button1.SetDefault()
        self.button2 = wx.Button(panel, -1, "加载数据", pos=(100, 60))
        self.Bind(wx.EVT_BUTTON, self.OnClick2, self.button2)
        self.button2.SetDefault()


        self.basicText = wx.TextCtrl(panel,-1,"此处显示职位信息",  pos=(20, 180),size = (1300,300),style = wx.TE_MULTILINE)
        self.basicText1 = wx.TextCtrl(panel, -1, "此处显示评分信息", pos=(700, 10), size=(500, 100), style=wx.TE_MULTILINE)


    def OnClick(self, event):
        #averageTenRuns()
        self.basicText.Clear()
        with codecs.open('ml-100k/text.txt', 'r', 'utf-8') as f:
           for line in f.readlines():
               try:
                   self.basicText.AppendText(line)
               except ValueError as e:
                   pass
        self.basicText1.Clear()
        with codecs.open('ml-100k/result.txt', 'r', 'utf-8') as f:
            for line in f.readlines():
                try:
                    self.basicText1.AppendText(line)
                except ValueError as e:
                    pass
    def OnClick1(self, event):
        averageTenRuns()

    def OnClick2(self, event):
        wildcard = "All files (*.*)|*.*"
        dlg = wx.FileDialog(self, "Choose a filename",
                            wildcard=wildcard,
                            style=wx.DD_DEFAULT_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            global path
            path = dlg.GetPath()
        print path
        dlg.Destroy()
if __name__ == '__main__':
        app = wx.PySimpleApp()
        frame = ButtonFrame()
        frame.Show()
        app.MainLoop()
#if __name__ == '__main__':

   # averageTenRuns()
#print("finish")
