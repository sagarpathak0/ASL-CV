# Try CuPy first, fallback to NumPy
try:
    import numpy as np
    GPU_AVAILABLE = False
except ImportError:
    import numpy as np
    GPU_AVAILABLE = False

#Layers
class conv3x3:
    def __init__(self, num_filters, input_depth):
        self.num_filters = num_filters
        scale = 1.0/np.sqrt(3*3*input_depth)
        self.filters = np.random.randn(num_filters, 3,3,input_depth)*scale
    
    def interate_regions(self, image):
        h, w, d = image.shape
        #loop over all 3*3 regions of the image
        for i in range(h-2):
            for j in range(w-2):
                patch  = image[i:i+3,j:j+3, :]
                yield i, j, patch

    def forward(self, input):
        self.last_input = input #store the backProp
        h, w, d = input.shape # output shrinks by 2 because filter size = 3
        output = np.zeros((h-2, w-2, self.num_filters))
        for i,j,patch in self.interate_regions(input):
            output[i,j] = np.sum(patch*self.filters, axis=(1,2,3))
        return output
    
    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        for i,j,patch in self.interate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i,j,f] * patch
        #update weights
        self.filters -= learn_rate * d_L_d_filters
        return None
    
#MAX pooling layer
class MaxPool2:
    def interate_regions(self, image):
        h,w,d=image.shape
        for i in range(h//2):
            for j in range(w//2):
                patch = image[i*2:i*2+2,j*2:j*2+2,:]
                yield i,j,patch


    def forward(self, input):
        self.last_input = input
        h,w,d = input.shape

        output = np.zeros((h//2, w//2, d))
        for i,j,patch in self.interate_regions(input):
            output[i,j] = np.max(patch, axis=(0,1))
        return output
    
    def backprop(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)
        h, w, d = self.last_input.shape
        
        for i, j, patch in self.interate_regions(self.last_input):
            # Find the indices of the maximum values in the patch
            max_indices = np.unravel_index(np.argmax(patch.reshape(patch.shape[0]*patch.shape[1], patch.shape[2]), axis=0), (patch.shape[0], patch.shape[1]))
            for k in range(d):
                max_i, max_j = max_indices[0][k], max_indices[1][k]
                d_L_d_input[i*2 + max_i, j*2 + max_j, k] = d_L_d_out[i, j, k]
        
        return d_L_d_input
    
#Softmax
class Softmax:
    def __init__(self, input_len, nodes):
        scale = 1.0/np.sqrt(input_len)
        self.weights = np.random.randn(input_len, nodes) * scale
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.lest_input_shape = input.shape
        input_flat = input.flatten()
        self.last_input = input_flat
        #raw scores
        totals = np.dot(input_flat, self.weights) + self.biases

        #softmax
        exp = np.exp(totals - np.max(totals))
        self.out = exp / np.sum(exp, axis=0)
        return self.out
    
    def backprop(self, d_L_d_out, learn_rate):
        for i, grad in enumerate(d_L_d_out):
            if grad==0:continue
            #recompute softmax derivatives
            t_exp = np.exp(self.last_input @ self.weights + self.biases - np.max(self.last_input @ self.weights))
            S = t_exp / np.sum(t_exp)

            d_out_d_t = -S * S[i]
            d_out_d_t[i] = S[i] * (1 - S[i])

            #Chain rule pieces
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            d_L_d_t = grad * d_out_d_t

            #weight/bias update
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

        return d_L_d_inputs.reshape(self.lest_input_shape)
    

#CNN model
class CNN:
    def __init__(self, num_classes=29):
        self.num_classes = num_classes
        self.conv = conv3x3(8, 1)
        self.pool = MaxPool2()
        # After conv: 28-2=26, after pool: 26//2=13
        # So input to softmax should be 13*13*8 = 1352
        self.softmax = Softmax(13*13*8, num_classes)

    def forward(self, image, label):
        out = self.conv.forward(image)
        self.conv_out = out.copy()  # Store conv output before ReLU
        out = np.maximum(0, out) #ReLU

        out = self.pool.forward(out)
        out = self.softmax.forward(out)

        #Loss = -log(probability of true label)
        loss = -np.log(out[label])
        #accuracy = 1 if predicted correct
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc
    
    def train(self, image, label, lr=0.005):
        out, loss, acc = self.forward(image, label)

        #Gradient WRT softmax output
        gradient = np.zeros(self.num_classes)
        gradient[label] = -1 / out[label]

        #backpass
        grad_back = self.softmax.backprop(gradient, lr)
        grad_back = self.pool.backprop(grad_back)
        # Apply ReLU derivative (1 if > 0, 0 otherwise)
        grad_back[self.conv_out <= 0] = 0
        self.conv.backprop(grad_back, lr)
        return loss, acc