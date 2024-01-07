import numpy as np
import scipy


class GaussianProcess():

    def llmin(x,fx,noise,NT=20):
        """ log likelihood minimization. does not need an initialized process. """

        # full grid approach over several scales
        thetas = np.linspace(-5,5,NT)
        lls = np.zeros(thetas.shape)
        for k in range(NT):
            try:
                gpr = GaussianProcess(x,
                                      fx,
                                      10**thetas[k],
                                      output_noise_std=noise)
                lls[k] = gpr.get_ll()
            except Exception:
                lls[k] = -9999

        p = scipy.interpolate.interp1d(thetas, -lls, fill_value=(np.min(-lls),np.max(-lls)), bounds_error=False)
        res = scipy.optimize.minimize_scalar(p, method="bounded", bounds=[np.min(thetas),np.max(thetas)])

        if not(res.success):
            print("Theta optimization did not converge")
            raise("Theta optimization did not converge")

        return 10**res.x, 10**thetas, lls

    def matern_kernel_v12(x,y,theta):
        if(len(x.shape)==1):
            x = x.reshape(1,-1)
        if(len(y.shape)==1):
            y = np.reshape(1,-1)
        
        distMatrix = scipy.spatial.distance.cdist(x,y, 'euclidean')
        return np.exp(-np.array(distMatrix)/(theta))
    
    def gaussian_kernel(x,y,theta):
        if(len(x.shape)==1):
            x = x.reshape(1,-1)
        if(len(y.shape)==1):
            y = y.reshape(1,-1)
        
        if (isinstance(theta, np.ndarray) and (len(theta.shape) > 1)):
            for k in range(x.shape[1]):
                x[:,k] = x[:,k] / np.sqrt(theta[k])
                y[:,k] = y[:,k] / np.sqrt(theta[k])
        else:
            x = x / np.sqrt(theta)
            y = y / np.sqrt(theta)
        
        distMatrix = scipy.spatial.distance.cdist(x,y, 'euclidean')
        return np.exp(-np.array(distMatrix)**2)
    
    def gaussian_kernel_grad(xx,yy,theta):
        """ computes the gradient of the kernel w.r.t. the first argument """
        x = xx.copy()
        y = yy.copy()
        if(len(x.shape)==1):
            x = x.reshape(1,-1)
        if(len(y.shape)==1):
            y = y.reshape(1,-1)
        
        if (isinstance(theta, np.ndarray) and (len(theta) > 1)):
            for k in range(x.shape[1]):
                x[:,k] = x[:,k] / np.sqrt(theta[k])
            for k in range(y.shape[1]):
                y[:,k] = y[:,k] / np.sqrt(theta[k])
        else:
            x = x / np.sqrt(theta)
            y = y / np.sqrt(theta)

        distMatrix = scipy.spatial.distance.cdist(x,y, 'euclidean')
        kxy = np.exp(-np.array(distMatrix)**2)

        xmy = np.zeros((x.shape[1],x.shape[0],y.shape[0]))
        if x.shape[1] == 1:
            xm = np.dot(xx,np.ones(y.shape).T)
            ym = np.dot(yy,np.ones(x.shape).T).T
            xmy[0,:,:] = xm-ym
        else:
            for k in range(x.shape[1]):
                xm = np.dot(xx[:,k].reshape(-1,1),np.ones((1,yy.shape[0])))
                ym = np.dot(yy[:,k].reshape(-1,1),np.ones((1,xx.shape[0]))).T
                xmy[k,:,:] = xm-ym

        res = np.zeros((x.shape[1], distMatrix.shape[0], distMatrix.shape[1]))
        for k in range(res.shape[0]):
            distMatrixK = xmy
            if (isinstance(theta, np.ndarray) and (len(theta) > 1)):
                res[k,:,:] = -2/(theta[k]) * kxy * np.array(distMatrixK[k,:])
            else:
                res[k,:,:] = -2/(theta) * kxy * np.array(distMatrixK[k,:])
        return res

    def __init__(self,
                 data: np.array,
                 fdata: np.array,
                 theta: np.array,
                 kernel = gaussian_kernel,
                 kernel_grad = gaussian_kernel_grad,
                 output_noise_std = 1e-5,
                 rcond = 1e-10, # used in lstsq
                 verbose=False) -> None:
        """ sets up the gaussian process on a list of points """
        data = data.copy()
        if(len(data.shape)==1):
            data = data.reshape(-1,1)
            
        self.__data = data
        self.__fdata = fdata
        self.__theta = theta
        self.__kernel = kernel
        self.__kernel_grad = kernel_grad
        self.__initialized = True
        self.__output_noise_std = output_noise_std
        self.__rcond = rcond
        
        if verbose and output_noise_std < 1e-5:
            print('Output noise std set to',output_noise_std,'which might cause matrices to become non-positive definite.')
        
        if not(self.__fdata is None):
            self.__alpha, self.__L, self.__ll = self.data_cov(output_noise_std, verbose)
        
    def get_ll(self):
        return self.__ll
    
    def data_cov(self, 
                 output_noise_std,
                 verbose):
        if (self.__initialized is None):
            raise ValueError('Must initialize first')
        k_xx = self.__kernel(self.__data,self.__data, self.__theta) + \
               output_noise_std**2 * np.identity(self.__data.shape[0])
        
        L = np.linalg.cholesky(k_xx)
        alpha0,res0,_,_ = np.linalg.lstsq(L, self.__fdata, rcond=self.__rcond)
        alpha,res,_,_ = np.linalg.lstsq(L.T, alpha0, rcond=self.__rcond)

        logLikelihood = -1/2*np.dot(self.__fdata.T, alpha) \
                        - np.sum(np.log(np.diag(L))) \
                        - L.shape[0]/2*np.log(2*np.pi)
        
        return alpha, L, np.array(logLikelihood).flatten()[0]
    
    def sample(self, xnew: np.array):
        m,std = self.predict(xnew)
        u = np.random.randn(xnew.shape[0],xnew.shape[1])
        L = np.linalg.cholesky(std+1e-5*self.__theta*np.identity(std.shape[0]))
        fnew= m+np.dot(L,u)
        return fnew
        
    def predict(self, xnew: np.array):
        if (self.__initialized is None):
            raise ValueError('Must initialize first')
        xnew = xnew.copy()
        if(len(xnew.shape)==1):
            xnew = x.reshape(1,-1)
            
        k_xxs = self.__kernel(self.__data,xnew, self.__theta)
        
        mean = np.dot(k_xxs.T, self.__alpha)
        
        k_xsxs = self.__kernel(xnew,xnew, self.__theta)
        
        v,_,_,_ = np.linalg.lstsq(self.__L, k_xxs, rcond=self.__rcond)
        std = k_xsxs - np.dot(v.T, v)
        
        return mean, std
        
    def predict_grad(self, xnew: np.array, dx=1e-8) -> None:
        if (self.__initialized is None):
            raise ValueError('Must initialize first')
        if(len(xnew.shape)==1):
            xnew = xnew.reshape(1,-1)
            
        if 1==1:
            k_xxs = self.__kernel_grad(self.__data,xnew, self.__theta)
            mean_grad = np.zeros((xnew.shape[0],xnew.shape[1]))

            for k in range(k_xxs.shape[0]):
                kxxx = np.dot(k_xxs[k,:,:].T, self.__alpha)
                mean_grad[:,k] = kxxx
        else:
            mean_grad = np.zeros((xnew.shape[0],xnew.shape[1]))
            for k in range(xnew.shape[1]):
                v1 = np.array(xnew)
                v1[:,k] = v1[:,k] - dx
                v2 = np.array(xnew)
                v2[:,k] = v2[:,k] + dx
                m1,_ = self.predict(v1)
                m2,_ = self.predict(v2)
                mean_grad[:,k] = (m2-m1)/(2*dx)
        
        return mean_grad
    
    def solve(self, x0, f0, xnew, gnew, solver_tolerance=1e-5, verbose=False):
        """ solve d/dx[f](ynew)=g(ynew) for f """
        if(len(xnew.shape)==1):
            xnew = xnew.reshape(1,-1)
        if(len(gnew.shape)==1):
            gnew = gnew.reshape(1,-1)
        if(len(x0.shape)==1):
            x0 = x0.reshape(1,-1)
            
        # self.__data = np.row_stack([self.__data, x0])
        
        k_xx = self.__kernel(self.__data,self.__data, self.__theta) +\
               self.__output_noise_std**2 * np.identity(self.__data.shape[0])
        # k_xxinv = np.linalg.pinv(k_xx, rcond=self.__rcond)
        
        k_x0x = self.__kernel(x0,self.__data, self.__theta)
        
        # print(k_x0x)
        
        # L = np.linalg.cholesky(k_xx)
        # alpha0,res0,_,_ = np.linalg.lstsq(L, self.__fdata, rcond=self.__rcond)
        
        k_grad =self.__kernel_grad(xnew, self.__data, self.__theta)
        
        mean_grad = np.zeros((xnew.shape[0]*xnew.shape[1],self.__data.shape[0]))

        for k in range(k_grad.shape[0]):
            #kxxx = np.dot(k_grad[k,:,:], k_xxinv)
            kxxx = np.linalg.lstsq(k_xx.T, k_grad[k,:,:].T, rcond=solver_tolerance)[0].T
            if verbose:
                print('kxxx',kxxx.shape)
            mean_grad[(k*xnew.shape[0]):((k+1)*xnew.shape[0]),:] = kxxx
        
        gnew = gnew.reshape(-1,1)
        
        if verbose:
            print('gnew',gnew.shape)
            print('k_xx',k_xx.shape)
            print('k_x0x',k_x0x.shape)
            print('k_grad',k_grad.shape)
            print('mean_grad',mean_grad.shape)
            print('f0',f0.shape)
        
        kx0xkinv = np.linalg.lstsq(k_xx.T, k_x0x.T, rcond=solver_tolerance)[0].T
        mean_grad = np.row_stack([mean_grad, kx0xkinv])
        gnew = np.row_stack([gnew,f0])
        
        if verbose:
            print('mean_grad',mean_grad.shape)
            print('gnew',gnew.shape)
            
        fx,residuals,rank,singularValues = np.linalg.lstsq(mean_grad, gnew, rcond=self.__rcond)
        # fx,residuals,rank,singularValues = np.linalg.lstsq(L, a0, rcond=self.__rcond)
        
        if verbose:
            print('fx',fx.shape)
        #print('rank',rank)
        # print('residuals',residuals)
        #print('singularValues',singularValues)
        
        # fx = fx[0:-(x0.shape[0])]
        
        return (fx)