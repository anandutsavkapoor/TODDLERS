import numpy as np
from names_and_constants import MASS_BIN_CENTERS, AGE_START, AGE_LIMIT

class Particle:
    def __init__(self, x, y, z, h, age, metallicity, sfe, density, mass, cloud_mass=None, decomposed_mass=None):
        self.x = x
        self.y = y
        self.z = z
        self.h = h
        self.age = age
        self.metallicity = metallicity
        self.sfe = sfe
        self.density = density
        self.mass = mass # particle's original mass
        self.cloud_mass = cloud_mass  # Corresponds to the mass of the bin
        self.decomposed_mass = decomposed_mass


    def _sample_new_position(self, r):
        """Sample a random point within a sphere of radius r"""
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        u = np.random.uniform(0, 1)
        theta = np.arccos(costheta)
        rad = r * np.cbrt(u)
        
        x = rad * np.sin(theta) * np.cos(phi)
        y = rad * np.sin(theta) * np.sin(phi)
        z = rad * np.cos(theta)
        
        return x, y, z

    def _smooth_age(self, scale):
        """Smooth age using Gaussian function"""
        self.age += np.random.normal(0, scale)
        return np.clip(self.age, AGE_START, AGE_LIMIT)

    def decompose(self, r, smooth_age_scale, mass_fractions):
        """Decompose the particle into components corresponding to specific mass bins"""
        decomposed_particles = []
        
        for i, cloud_mass in enumerate(MASS_BIN_CENTERS):
            new_h = self.h * np.cbrt(mass_fractions[i])
            dx, dy, dz = self._sample_new_position(r)
            new_age = self._smooth_age(smooth_age_scale)
            new_particle = Particle(self.x + dx, self.y + dy, self.z + dz, new_h, new_age, 
                                    self.metallicity, self.sfe, self.density, self.mass, 
                                    cloud_mass=cloud_mass, decomposed_mass=self.mass * mass_fractions[i])
            decomposed_particles.append(new_particle)
        
        return decomposed_particles

