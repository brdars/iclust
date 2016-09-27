"""
Copyright (C) 2014  Sanja Brdar <brdars@uns.ac.rs>

This file is part of iclust.

Iclust is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Iclust is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with iclust.  If not, see <http://www.gnu.org/licenses/>.
"""

import scipy
import math
import random
import numpy as np



class Nmf(object):
    """ Non-negative matrix factorization. Returns two factorized matrices """
    def __init__(self, M, init = 'svd', components=150, iterations=10000):
        self.M = M
        self.init = init
        self.components = components
        self.iterations = iterations
        self.W = np.matrix(np.zeros((M.shape[0], components)))
        self.H = np.matrix(np.zeros((components, M.shape[1])))


    def _svdInit(self):
        U, s, V = scipy.linalg.svd(self.M)
##        U, s, V = scipy.sparse.linalg.svds(self.M)
        U = U[:, :self.components]
        S = np.diag(s[:self.components])
        V = V[:self.components,:]
        for j in range(0,self.components):
            x = U[:,j]
            y = V[j,:]
            xp = np.where(x > 0, x, 0)
            xn =  np.where(x < 0, abs(x), 0)
            yp = np.where(y > 0, y, 0)
            yn =  np.where(y < 0, abs(y), 0)
            xpnrm = np.linalg.norm(xp)
            ypnrm = np.linalg.norm(yp)
            mp = xpnrm * ypnrm
            xnnrm = np.linalg.norm(xn)
            ynnrm = np.linalg.norm(yn)
            mn = xnnrm * ynnrm
            if mp > mn:
                u=xp/xpnrm
                v = yp/ypnrm
                sigma = mp
            else:
                u=xn/xnnrm
                v = yn/ynnrm
                sigma = mn
            self.W[:,j] = np.matrix(math.sqrt(S[j,j]*sigma)*u).T
            self.H[j,:] = math.sqrt(S[j,j]*sigma)*v


    def _randInit(self):
        self.W = np.matrix(np.random.rand(self.M.shape[0], self.components))
        self.H = np.matrix(np.random.rand(self.components, self.M.shape[1]))

    def __calc_rec_error(self):
        self.rerror = np.sqrt(np.square(self.M - self.W * self.H).sum())

    def run(self):
        if self.init == 'svd':
            self._svdInit()
        if self.init == 'rand':
            self._randInit()
        for n in range(0, self.iterations):
            self.H = np.multiply(self.H, ((self.W).T * self.M) / ((self.W).T * self.W * self.H + 0.001))
            self.W = np.multiply(self.W, (self.M * (self.H).T) / (self.W * (self.H * (self.H).T) + 0.001))
        self.__calc_rec_error()
        self.create_clusters(range(self.H.shape[1]))

    def _rescale(self, W, H):
        """Rescales factor matrices W and H"""
        for j in range(W.shape[1]):
            w_list = []
            for i in range(W.shape[0]):
                w_list.append(abs(W[i,j]))
            maxValueW = max(w_list)

            h_list = []
            for k in range(H.shape[1]):
                h_list.append(H[j,k])
            maxValueH = max(h_list)

            if maxValueW != 0 and maxValueH != 0:
                resc = math.sqrt(maxValueH)/ math.sqrt(maxValueW)
                for i in range(W.shape[0]):
                    W[i,j] = W[i,j] * resc
                for k in range(H.shape[1]):
                    H[j,k] = H[j,k] / resc
        return W, H

    def _create_cluster_labels(self, attr_list):
        """Transforms clustering into list of labels"""
        cluster_list= [0]*len(attr_list)
        label = np.array([cluster[0] for cluster in self.clusters]).argsort().argsort().tolist()
        print label
        for i in range(len(self.clusters)):
            for j in self.clusters[i]:
                cluster_list[j] = label[i]
        return cluster_list


    def create_clusters(self, attr_list, thr = 0.0, ctype = 'exclusive'):
        """Extracts clusters from matrix factors W and H"""
        W, H = self._rescale(self.W, self.H)
        self.clusters = []
        self.labels = []

        if ctype == 'overlapping':
            for i in range(H.shape[0]):
                cluster = []
                for j in range(H.shape[1]):
                    if H[i,j]> thr:
                        cluster.append(attr_list[j])
                if len(cluster)>0:
                    self.clusters.append(cluster)

        if ctype == 'exclusive':
            w_sum_list = []
            for j in range(W.shape[1]):
                w_sum_list.append(sum([W[i,j] for i in range(W.shape[0])]))
            for i in range(H.shape[0]):
                cluster = []
                for j in range(H.shape[1]):
                    if H[i,j]> thr and H[i,j] * w_sum_list[i] == max(H[k,j] * w_sum_list[k] for k in range(H.shape[0])):
                        cluster.append(attr_list[j])
                if len(cluster)>0:
                    self.clusters.append(cluster)
            self.labels = self._create_cluster_labels(attr_list)

