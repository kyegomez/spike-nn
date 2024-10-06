# spike-nn
Spike neural network


```
import math
import torch
from torch import nn
from loguru import logger
from typing import List, Dict, Tuple

class MultiHeadAttention(nn.Module):
    """
    Implements multi-head self-attention.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Split the embedding dimension into num_heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, v)

        # Reshape back to (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out(attn_output)


class FeedForward(nn.Module):
    """
    Implements a feed-forward network (position-wise).
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class TransformerNeuron(nn.Module):
    """
    Represents a transformer neuron with spiking behavior. The neuron spikes when its
    accumulated potential exceeds a threshold. It implements a basic transformer architecture.
    
    Args:
        embed_dim (int): Dimension of the input embeddings.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of the feed-forward hidden layer.
        threshold (float): Firing threshold.
        reset_potential (float): Potential to reset to after spiking.
        refractory_period (float): Time during which the neuron cannot spike after firing.
        decay_rate (float): Rate at which potential decays over time if not spiking.
    """

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int,
                 threshold: float = 1.0, reset_potential: float = 0.0,
                 refractory_period: float = 1.0, decay_rate: float = 0.01):
        super(TransformerNeuron, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.threshold = threshold
        self.reset_potential = reset_potential
        self.potential = reset_potential
        self.spiking = False
        self.refractory_period = refractory_period
        self.last_spike_time = None  # Track last spike time for refractory period
        self.decay_rate = decay_rate
        self.current_time = time.time()

        logger.info(f"Initialized Transformer Neuron with threshold: {self.threshold}, refractory period: {self.refractory_period}, and decay rate: {self.decay_rate}")

    def accumulate(self, x: torch.Tensor) -> None:
        """
        Accumulates potential based on the input. If the threshold is crossed, the neuron spikes.
        The potential decays over time if not spiking.
        
        Args:
            x (torch.Tensor): Input tensor to process and accumulate potential.
        """
        # Decay potential over time before accumulating
        elapsed_time = time.time() - self.current_time
        self.potential -= self.decay_rate * elapsed_time
        self.potential = max(self.potential, 0)  # Ensure potential doesn't go below zero
        self.current_time = time.time()

        # If in refractory period, skip accumulation
        if self.last_spike_time and (time.time() - self.last_spike_time) < self.refractory_period:
            logger.debug("Neuron in refractory period, skipping accumulation.")
            return

        # Accumulate potential based on transformer block output
        attn_output = self.attention(x)
        attn_output = self.norm1(attn_output + x)
        ff_output = self.feed_forward(attn_output)
        ff_output = self.norm2(ff_output + attn_output)

        potential_change = ff_output.mean().item()  # Mean activation value as potential change
        self.potential += potential_change

        logger.debug(f"Accumulated potential: {self.potential} (threshold: {self.threshold})")

        # Check if potential exceeds the threshold
        if self.potential >= self.threshold:
            self.spiking = True
            logger.info("Neuron is spiking!")
        else:
            self.spiking = False

    def spike(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the output of the neuron when it spikes.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Neuron output when it spikes.
        """
        if not self.spiking:
            logger.warning("Neuron is not ready to spike.")
            return x  # No modification if not spiking

        # Reset neuron state after spiking and record spike time
        self.potential = self.reset_potential
        self.spiking = False
        self.last_spike_time = time.time()
        logger.info("Neuron spiked, returning transformed output.")

        return self.feed_forward(x)


class SpikingNeuralNetwork(nn.Module):
    """
    Represents a spiking neural network composed of transformer neurons.
    
    The SNN takes an input, propagates it through the neurons, and collects outputs from neurons that spike.

    Args:
        neuron_configs (List[Tuple[int, int, int, float, float, float, float]]): A list of tuples representing the 
            configuration for each neuron. Each tuple contains:
            - embed_dim (int): Dimension of the input embeddings.
            - num_heads (int): Number of attention heads.
            - hidden_dim (int): Dimension of the feed-forward hidden layer.
            - threshold (float): Firing threshold for the neuron.
            - reset_potential (float): Potential value after spiking.
            - refractory_period (float): Refractory period for the neuron.
            - decay_rate (float): Decay rate for potential if not spiking.
    """

    def __init__(self, neuron_configs: List[Tuple[int, int, int, float, float, float, float]]):
        super(SpikingNeuralNetwork, self).__init__()
        self.neurons = nn.ModuleList([
            TransformerNeuron(embed_dim, num_heads, hidden_dim, threshold, reset_potential, refractory_period, decay_rate)
            for embed_dim, num_heads, hidden_dim, threshold, reset_potential, refractory_period, decay_rate in neuron_configs
        ])

        logger.info(f"Initialized Spiking Neural Network with {len(self.neurons)} neurons.")

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Propagates input through the network and collects outputs from neurons that spike.
        
        Args:
            x (torch.Tensor): Input tensor to feed into the network.

        Returns:
            Dict[int, torch.Tensor]: A dictionary with neuron index as key and the output tensor as value.
        """
        outputs = {}

        for i, neuron in enumerate(self.neurons):
            neuron.accumulate(x)
            if neuron.spiking:
                neuron_output = neuron.spike(x)
                outputs[i] = neuron_output
                logger.debug(f"Neuron {i} spiked and returned transformed output.")

        if not outputs:
            logger.warning("No neurons spiked in this iteration.")

        return outputs


if __name__ == "__main__":
    # Configure logging
    logger.add("snn_log.log", rotation="10 MB")  # Save logs to file

    # Example configuration for
    # Example configuration for neurons: embed_dim, num_heads, hidden_dim, threshold, reset_potential, refractory_period, decay_rate
    neuron_configs = [
        (512, 8, 2048, 1.0, 0.0, 1.0, 0.02),
        (512, 8, 2048, 1.5, 0.5, 1.5, 0.03),
        (512, 8, 2048, 2.0, 1.0, 2.0, 0.05)
    ]

    # Initialize spiking neural network
    snn = SpikingNeuralNetwork(neuron_configs)

    # Sample input: Randomly generated tensor of shape (batch_size, seq_len, embed_dim)
    input_tensor = torch.randn(1, 10, 512)

    # Run the SNN with the input tensor
    output = snn(input_tensor)
    logger.info(f"Final output from spiking neurons: {output}")
```
