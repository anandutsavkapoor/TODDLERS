import numpy as np

class CloudDistribution:
    def __init__(self, alpha, bin_centers, n):
        self.alpha = alpha
        self.original_bin_centers = bin_centers
        self.extended_bin_centers = self._generate_extended_bin_centers(bin_centers)
        self.bins = self._generate_bin_edges(self.extended_bin_centers)
        self.n = n

    def _generate_extended_bin_centers(self, bin_centers):
        log_centers = np.log10(bin_centers)
        extended_log_centers = np.concatenate([[log_centers[0] - 0.25], log_centers, [log_centers[-1] + 0.25]])
        return 10**extended_log_centers

    def _generate_bin_edges(self, bin_centers):
        half_diffs = np.diff(bin_centers) / 2
        edges = [bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2] + \
                list(bin_centers[:-1] + half_diffs) + \
                [bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2]
        return np.array(edges)
    
    def power_law_cdf(self, x):
        min_val = self.bins[0]
        max_val = self.bins[-1]
        norm = max_val**(self.alpha + 1) - min_val**(self.alpha + 1)
        return (x**(self.alpha + 1) - min_val**(self.alpha + 1)) / norm
    
    def inverse_power_law_cdf(self, u):
        min_val = self.bins[0]
        max_val = self.bins[-1]
        norm = max_val**(self.alpha + 1) - min_val**(self.alpha + 1)
        return ((u * norm) + min_val**(self.alpha + 1))**(1 / (self.alpha + 1))
    
    def sample_from_cdf(self):
        cdf_values = self.power_law_cdf(self.bins)
        bin_probabilities = np.diff(cdf_values)
        samples = np.random.choice(self.extended_bin_centers, size=self.n, p=bin_probabilities)
        return samples
    
    def sample_directly(self):
        uniform_samples = np.random.uniform(0, 1, self.n)
        power_law_samples = self.inverse_power_law_cdf(uniform_samples)
        bin_indices = np.digitize(power_law_samples, self.bins) - 1
        return self.extended_bin_centers[bin_indices]

    def sample_and_filter(self):
        samples = self.sample_directly()
        filtered_samples = samples[np.isin(samples, self.original_bin_centers)]
        while len(filtered_samples) < self.n:
            additional_samples = self.sample_directly()
            additional_filtered_samples = additional_samples[np.isin(additional_samples, self.original_bin_centers)]
            filtered_samples = np.concatenate([filtered_samples, additional_filtered_samples])[:self.n]
        return filtered_samples


