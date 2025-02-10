import torch
import torch.nn as nn

# Define WrappedGPT class
class WrappedGPT:
    """
    A wrapper for a GPT layer (or similar neural network layer) that performs additional operations,
    such as tracking running statistics (specifically, the squared L2 norm averages) of its input.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        # Store the provided layer (e.g., an instance of nn.Linear)
        self.layer = layer
        
        # Get the device (CPU, GPU, etc.) where the layer's weights are stored
        self.dev = self.layer.weight.device
        
        # Retrieve the number of rows and columns from the layer's weight matrix.
        # These could be used to understand the dimensions of the layer.
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        # Initialize a tensor of zeros to accumulate a running statistic for each feature/column.
        # This tensor is stored on the same device as the layer's weights.
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        
        # Initialize a counter for the total number of samples processed so far.
        self.nsamples = 0

        # Store additional metadata for identification or debugging purposes.
        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        """
        Process a new batch of inputs (and outputs, though 'out' is not used in this method)
        to update the running statistic (scaler_row) based on the new data.
        """

        # If the input tensor is 2-dimensional (i.e., [batch_size, features]),
        # add an extra dimension so that it becomes 3-dimensional.
        # This ensures that there is always a batch dimension.
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        
        # 'tmp' holds the number of new batches/samples from the input.
        tmp = inp.shape[0]
        
        # Special handling if the wrapped layer is an instance of nn.Linear.
        if isinstance(self.layer, nn.Linear):
            # If the input tensor has 3 dimensions (e.g., [batch, sequence_length, features]),
            # reshape it into a 2D tensor where the first dimension combines batch and sequence.
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            # Transpose the input so that each row corresponds to a feature.
            # After transposition, the shape becomes [features, number_of_samples],
            # making it convenient to compute per-feature statistics.
            inp = inp.t()

        # Update the running average stored in scaler_row:
        # First, scale the current scaler_row by the ratio of the old sample count to the new total.
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        
        # Increment the total number of samples processed by the number of new samples.
        self.nsamples += tmp

        # Ensure the input tensor is of type float32 for numerical operations.
        inp = inp.type(torch.float32)
        
        # Compute the L2 norm (Euclidean norm) for each row (i.e., for each feature across all samples).
        # Squaring the norm gives the sum of squares for each feature.
        # Dividing by the total number of samples provides an average contribution,
        # which is then added to the running statistic in scaler_row.
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
