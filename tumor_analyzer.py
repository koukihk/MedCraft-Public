import glob
import os
import pickle
import random
import string
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import plotly.offline as pyo
from deap import base, creator, tools, algorithms
from scipy import interpolate
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from tqdm import tqdm


class EllipsoidOptimizer:
    def __init__(self, data, n_gen=200, pop_size=300, coverage_weight=0.35, compactness_weight=0.65):
        self.data = data
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.coverage_weight = coverage_weight
        self.compactness_weight = compactness_weight

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", np.random.uniform, 0.5, 5.0)
        self.toolbox.register("attr_offset", np.random.uniform, -20.0, 20.0)
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                              (self.toolbox.attr_float, self.toolbox.attr_float, self.toolbox.attr_float,
                               self.toolbox.attr_offset, self.toolbox.attr_offset, self.toolbox.attr_offset), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.eval_ellipsoid)

    def eval_ellipsoid(self, individual):
        fitter = EllipsoidFitter(self.data)
        scale_factors = np.round(individual[:3]).astype(int)
        center_offset = np.round(individual[3:]).astype(int)
        coverage, compactness = fitter.evaluate(scale_factors, center_offset)
        fitness = (self.coverage_weight * coverage) + (self.compactness_weight * compactness)
        return fitness,

    def optimize(self):
        population = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1, similar=np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        algorithms.eaSimple(population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=self.n_gen,
                            stats=stats, halloffame=hof, verbose=True)

        best_individual = hof[0]
        best_individual = self.local_search(best_individual)
        return best_individual

    def local_search(self, individual):
        def objective(x):
            fitter = EllipsoidFitter(self.data)
            scale_factors = np.round(x[:3]).astype(int)
            center_offset = np.round(x[3:]).astype(int)
            coverage, compactness = fitter.evaluate(scale_factors, center_offset)
            fitness = (self.coverage_weight * coverage) + (self.compactness_weight * compactness)
            return -fitness  # Minimize negative fitness

        bounds = [(0.5, 5.0)] * 3 + [(-20.0, 20.0)] * 3
        result = minimize(objective, individual, bounds=bounds, method='L-BFGS-B')
        best_individual = np.round(result.x)
        return best_individual


class EllipsoidFitter:
    def __init__(self, data=None, scale_factors=[3, 3, 3], center_offset=[0, 0, 0]):
        self.data = np.array(data) if data is not None else np.array([])
        self.scale_factors = scale_factors
        self.center_offset = np.array(center_offset)
        self.center = None
        self.axes = None
        self.radii = None

        if self.data.size > 0:
            self.filtered_data = self._remove_outliers(self.data)
            self.center, self.axes, self.radii = self._fit_ellipsoid(self.filtered_data)

    def _remove_outliers(self, data):
        if data.size == 0:
            return np.array([])
        mean = np.mean(data, axis=0)
        cov_matrix = np.cov(data, rowvar=False)
        if np.linalg.det(cov_matrix) == 0:
            return data  # 如果协方差矩阵是奇异的，返回原始数据
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        mahalanobis_distances = [mahalanobis(sample, mean, inv_cov_matrix) for sample in data]
        threshold = np.percentile(mahalanobis_distances, 98)
        filtered_data = data[np.array(mahalanobis_distances) <= threshold]
        return filtered_data

    def _fit_ellipsoid(self, data):
        if data.size == 0:
            return None, None, None
        pca = PCA(n_components=3)
        pca.fit(data)
        center = np.round(np.mean(data, axis=0) + self.center_offset).astype(int)
        axes = np.round(pca.components_, 1)
        variances = pca.explained_variance_
        radii = np.round(np.sqrt(variances) * self.scale_factors).astype(int)
        radii = np.where(radii == 0, 1, radii)
        return center, axes, radii

    def evaluate(self, scale_factors, center_offset):
        if self.data.size == 0:
            return 0, 0  # 如果没有数据，返回默认值
        self.scale_factors = scale_factors
        self.center_offset = center_offset
        self.filtered_data = self._remove_outliers(self.data)
        self.center, self.axes, self.radii = self._fit_ellipsoid(self.filtered_data)

        distances = np.linalg.norm((self.filtered_data - self.center) @ np.linalg.inv(np.diag(self.radii)), axis=1)
        coverage = np.mean(distances <= 1)
        volume = (4 / 3) * np.pi * np.prod(self.radii)
        compactness = 1 / volume

        return coverage, compactness

    def set_precomputed_parameters(self, center, axes, radii):
        self.center = np.array(center)
        self.axes = np.array(axes)
        self.radii = np.array(radii)

    @classmethod
    def from_precomputed_parameters(cls, center, axes, radii):
        instance = cls()
        instance.set_precomputed_parameters(center, axes, radii)
        return instance

    def get_random_point(self):
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        u = np.random.uniform(0, 1)
        theta = np.arccos(costheta)
        r = u ** (1 / 3)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        random_point = np.dot([x, y, z], self.axes.T * self.radii) + self.center
        return random_point

    def plot_ellipsoid(self):
        """
        Plot the fitted ellipsoid along with the given data points.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.filtered_data[:, 0], self.filtered_data[:, 1], self.filtered_data[:, 2], color='b', s=1)

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.radii[0] * np.outer(np.cos(u), np.sin(v))
        y = self.radii[1] * np.outer(np.sin(u), np.sin(v))
        z = self.radii[2] * np.outer(np.ones_like(u), np.cos(v))

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], self.axes) + self.center

        ax.plot_wireframe(x, y, z, color='r', alpha=0.1)
        plt.show()

    def plot_interactive_ellipsoid(self, folder='ellipsoid_plot'):
        """
        Plot the fitted ellipsoid along with the given data points using Plotly and save to an HTML file
        in the specified folder.
        """
        # Ensure the target folder exists
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Generate a unique filename
        identifier = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(folder, f"ellipsoid_plot_{identifier}.html")

        # Identify outliers
        filtered_data_set = set(map(tuple, self.filtered_data))
        outliers = np.array([point for point in self.data if tuple(point) not in filtered_data_set])

        # Create scatter plot for filtered data points
        filtered_scatter = go.Scatter3d(
            x=self.filtered_data[:, 0],
            y=self.filtered_data[:, 1],
            z=self.filtered_data[:, 2],
            mode='markers',
            marker=dict(size=2, color='blue'),
            name='Filtered Data Points'
        )

        # Create scatter plot for outliers
        x = [point[0] for point in outliers]
        y = [point[1] for point in outliers]
        z = [point[2] for point in outliers]

        outlier_scatter = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=2, color='red'),
            name='Outliers'
        )

        # Create the ellipsoid
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.radii[0] * np.outer(np.cos(u), np.sin(v))
        y = self.radii[1] * np.outer(np.sin(u), np.sin(v))
        z = self.radii[2] * np.outer(np.ones_like(u), np.cos(v))

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], self.axes) + self.center

        # Create mesh for ellipsoid
        ellipsoid = go.Surface(
            x=x, y=y, z=z,
            opacity=0.5,
            colorscale='reds',
            showscale=False,
            name='Ellipsoid'
        )

        # Layout for the plot
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z')
            ),
            title='Ellipsoid Fit with Data Points'
        )

        # Create figure and add scatter plots and ellipsoid
        fig = go.Figure(data=[filtered_scatter, outlier_scatter, ellipsoid], layout=layout)

        # Save the plot as an HTML file
        pio.write_html(fig, file=filename, auto_open=False)

    def get_ellipsoid_equation(self):
        """
        Get the mathematical equation of the fitted ellipsoid.
        """
        inv_radii_squared = 1.0 / (self.radii ** 2)
        equation = "Ellipsoid equation:\n"
        for i in range(3):
            term = f"(({self.axes[i, 0]} * (x - {self.center[0]}) + " \
                   f"{self.axes[i, 1]} * (y - {self.center[1]}) + " \
                   f"{self.axes[i, 2]} * (z - {self.center[2]}))^2) / " \
                   f"{1.0 / inv_radii_squared[i]}"
            if i < 2:
                term += " + "
            else:
                term += " = 1"
            equation += term + "\n"
        return equation


class GMMPlotter:
    @staticmethod
    def plot_gmm(gmm_model, data, model_type='global', num_samples=700, num_points=70):
        """
        Plot a 3D visualization of a Gaussian Mixture Model (GMM) using Plotly.

        Parameters:
        - gmm_model: A fitted Gaussian Mixture Model with attributes means_, covariances_, weights_, and covariance_type.
        - data: The original data points used for fitting the GMM.
        - num_samples: The total number of samples to generate from the GMM.
        - num_points: The number of points to use for plotting the ellipsoid surfaces.
        """

        def plot_gmm_component(mean, covariance, color, num_points):
            """
            Generate the ellipsoid representing the GMM component defined by mean and covariance.

            Parameters:
            - mean: Mean of the Gaussian component.
            - covariance: Covariance matrix of the Gaussian component.
            - color: Color of the ellipsoid.
            - num_points: Number of points to use for plotting the ellipsoid.

            Returns:
            - A Plotly trace representing the ellipsoid.
            """
            u = np.linspace(0, 2 * np.pi, num_points)
            v = np.linspace(0, np.pi, num_points)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(num_points), np.cos(v))

            xyz = np.dot(np.vstack((x.flatten(), y.flatten(), z.flatten())).T, np.linalg.cholesky(covariance).T) + mean

            x = xyz[:, 0].reshape(num_points, num_points)
            y = xyz[:, 1].reshape(num_points, num_points)
            z = xyz[:, 2].reshape(num_points, num_points)

            return go.Surface(x=x, y=y, z=z, colorscale=[[0, color], [1, color]], opacity=0.2, showscale=False)

        def get_covariance_matrix(covariances, index, covariance_type):
            """
            Retrieve the covariance matrix for the specified component.

            Parameters:
            - covariances: Covariance matrices from the GMM.
            - index: Index of the component.
            - covariance_type: Type of the covariance matrix (full, tied, diag, spherical).

            Returns:
            - The covariance matrix for the specified component.
            """
            if covariance_type == 'full':
                return covariances[index]
            elif covariance_type == 'tied':
                return covariances
            elif covariance_type == 'diag':
                return np.diag(covariances[index])
            elif covariance_type == 'spherical':
                return np.eye(len(covariances[index])) * covariances[index]

        means = gmm_model.means_
        covariances = gmm_model.covariances_
        weights = gmm_model.weights_
        covariance_type = gmm_model.covariance_type

        data_traces = []

        # Add the original data points to the plot with special color and larger size
        original_data_trace = go.Scatter3d(
            x=data[:, 0], y=data[:, 1], z=data[:, 2],
            mode='markers',
            marker=dict(size=2, opacity=0.7, color='blue'),  # Highlight original data points with red color
            name='Original Data'
        )
        data_traces.append(original_data_trace)

        for i, weight in enumerate(weights):
            cov_matrix = get_covariance_matrix(covariances, i, covariance_type)

            # Generate random samples according to the component's parameters
            samples = np.random.multivariate_normal(means[i], cov_matrix, size=max(1, int(num_samples * weight)))

            scatter = go.Scatter3d(
                x=samples[:, 0], y=samples[:, 1], z=samples[:, 2],
                mode='markers',
                marker=dict(size=2, opacity=0.4,
                            color=f'rgba({(i * 30) % 256}, {(i * 60) % 256}, {(i * 90) % 256}, 0.4)'),
                name=f'Component {i + 1}'
            )
            data_traces.append(scatter)

            # Plot ellipsoids for 1, 2, and 3 standard deviations with lower opacity
            for k in [1, 2, 3]:
                ellipsoid = plot_gmm_component(means[i], k**2 * cov_matrix,
                                               color=f'rgb({(i * 30) % 256}, {(i * 60) % 256}, {(i * 90) % 256})',
                                               num_points=num_points)
                data_traces.append(ellipsoid)

        # Customize layout
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z')
            ),
            title='3D GMM Visualization',
            showlegend=True
        )

        fig = go.Figure(data=data_traces, layout=layout)

        # Information text
        info_text = ''.join(
            [
                f'Component {i + 1}:\n'
                f'Mean: {np.round(means[i], 2)}\n'
                f'Covariance: {np.round(get_covariance_matrix(covariances, i, covariance_type), 2)}\n\n'
                for i in range(len(weights))
            ]
        ) + f'CovType: {covariance_type}\nModelType: {model_type}'

        # Add a text box for information
        fig.add_annotation(
            x=0.5,
            y=1.1,
            xref='paper',
            yref='paper',
            text=info_text,
            showarrow=False,
            font=dict(size=12),
            align='left',
            bordercolor='black',
            borderwidth=1
        )

        # Save the figure as an HTML file
        output_directory = 'gmm/html'
        os.makedirs(output_directory, exist_ok=True)
        output_file = os.path.join(output_directory,
                                   f'gmm_model_{model_type}_{covariance_type}_{len(gmm_model.weights_)}.html')
        pyo.plot(fig, filename=output_file, auto_open=True)


class Tumor:
    def __init__(self, position=None, type=None, filename=None):
        self.position = position  # relative position
        self.type = type  # one of ['tiny', 'small', 'medium', 'large']

    def __repr__(self):
        return f"Tumor(position={self.position}, type={self.type})"


class TumorAnalyzer:
    def __init__(self):
        self.all_tumors = None
        self.gmm_model = None
        self.gmm_model_global = None
        self.gmm_model_tiny = None
        self.gmm_model_non_tiny = None
        self.gmm_flag = False
        self.healthy_ct = [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]
        self.target_volume = (300, 250, 140)

    def fit_gmm_model(self, tumors, optimal_components, cov_type='diag', tol=0.00001, max_iter=500, n_splits=5,
                      use_cv=True):
        """
        Fits a Gaussian Mixture Model to the given data.
        """
        positions = np.array([tumor.position for tumor in tumors])

        if use_cv:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            best_score = float('-inf')
            best_model = None

            for train_index, val_index in kf.split(positions):
                train_positions, val_positions = positions[train_index], positions[val_index]

                gmm_model = GaussianMixture(
                    n_components=optimal_components,
                    covariance_type=cov_type,
                    init_params='k-means++',
                    tol=tol,
                    max_iter=max_iter
                )

                gmm_model.fit(train_positions)
                val_score = gmm_model.score(val_positions)

                if val_score > best_score:
                    best_score = val_score
                    best_model = gmm_model

            # Save the best model
            self.gmm_model = best_model
            print("Best model selected with validation score: {}".format(best_score))
        else:
            self.gmm_model = GaussianMixture(
                n_components=optimal_components,
                covariance_type=cov_type,
                init_params='k-means++',
                tol=tol,
                max_iter=max_iter
            )
            self.gmm_model.fit(positions)
            val_score = self.gmm_model.score(positions)
            print("Model fitted without cross-validation {}".format(val_score))

    @staticmethod
    def get_subdirectory(data_dir):
        # Check which pair of subdirectories exists
        if os.path.isdir(os.path.join(data_dir, "imagesTr")) and os.path.isdir(os.path.join(data_dir, "labelsTr")):
            img_subdir = "imagesTr"
            label_subdir = "labelsTr"
        elif os.path.isdir(os.path.join(data_dir, "img")) and os.path.isdir(os.path.join(data_dir, "label")):
            img_subdir = "img"
            label_subdir = "label"
        else:
            img_subdir = ""
            label_subdir = ""
        return img_subdir, label_subdir

    @staticmethod
    def process_file(ct_file, data_dir):
        img_subdir, label_subdir = TumorAnalyzer.get_subdirectory(data_dir)
        img_path = os.path.join(data_dir, img_subdir, ct_file)
        label_path = os.path.join(data_dir, label_subdir, ct_file)

        if not (os.path.isfile(img_path) and os.path.isfile(label_path)):
            return [], []

        tumors = TumorAnalyzer.analyze_tumors(label_path, (300, 250, 140), 2)
        return tumors

    def load_data(self, data_dir, parallel=False):
        """
        Loads CT scan images and corresponding tumor labels from the specified data folder.
        """
        _, label_subdir = TumorAnalyzer.get_subdirectory(data_dir)
        ct_files = sorted(os.listdir(os.path.join(data_dir, label_subdir)))
        expected_count = len(ct_files) // 2 - len(self.healthy_ct)
        ct_files = [ct_file for ct_file in ct_files
                    if not ct_file.startswith("._")
                    and int(ct_file.split('_')[1].split('.')[0]) not in self.healthy_ct]

        if len(ct_files) != expected_count:
            warnings.warn(f"Expected {expected_count} files after filtering, but found {len(ct_files)}.",
                          Warning)

        all_tumors = []
        if parallel:
            max_processes = min(cpu_count(), 6)
            with Pool(max_processes) as pool:
                results = []
                for ct_file in ct_files:
                    results.append(pool.apply_async(TumorAnalyzer.process_file, (ct_file, data_dir)))

                for result in tqdm(results, total=len(results), desc="Processing dataset"):
                    tumors = result.get()
                    all_tumors.extend(tumors)

        else:
            for ct_file in tqdm(ct_files, total=len(ct_files), desc="Processing dataset"):
                tumors = TumorAnalyzer.process_file(ct_file, data_dir)
                all_tumors.extend(tumors)

        self.all_tumors = all_tumors

        tumor_count = len(all_tumors)
        type_count = {'tiny': 0, 'small': 0, 'medium': 0, 'large': 0}

        for tumor in all_tumors:
            type_count[tumor.type] += 1
        type_proportions = {tumor_type: count / tumor_count for tumor_type, count in type_count.items()}

        print("Total number of tumors:", tumor_count)
        print("Tumor counts by type: " + ", ".join(
            [f"{tumor_type}: {count}" for tumor_type, count in type_count.items()]))
        print("Tumor type proportions:",
              ", ".join([f"{tumor_type}: {proportion:.2%}" for tumor_type, proportion in type_proportions.items()]))

    def gmm_starter(self, data_dir, optimal_components, cov_type='diag', split=False,
                    use_cv=True, parallel=False):
        """
        Loads data, prepares training and validation sets, and fits GMM model with early stopping.
        """

        def generate_random_str(length=6):
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

        tol = 0.00001
        max_iter = 500

        if not self.gmm_flag:
            os.makedirs(f'gmm/{cov_type}', exist_ok=True)
            print_mode = "global" if not split else "split"
            print(f'use {print_mode} mode: {optimal_components}')

            tumors_path = os.path.join(data_dir, 'tumors.npy')
            if os.path.exists(tumors_path):
                tumors_data = np.load(tumors_path, allow_pickle=True)
                if len(tumors_data) > 850:
                    print(f"tumors.npy found with {len(tumors_data)} tumors. Skipping data loading.")
                    self.all_tumors = tumors_data.tolist()
                else:
                    print(f"tumors.npy found but only {len(tumors_data)} tumors. Loading data.")
                    self.load_data(data_dir, parallel=parallel)
            else:
                print("tumors.npy not found. Loading data.")
                self.load_data(data_dir, parallel=parallel)

            if split:
                all_tiny_tumors = [tumor for tumor in self.all_tumors if tumor.type == 'tiny']
                all_non_tiny_tumors = [tumor for tumor in self.all_tumors if tumor.type != 'tiny']
                nc_tiny, nc_non_tiny = optimal_components
                for tumor_type, tumors, nc in [("tiny", all_tiny_tumors, nc_tiny),
                                               ("non_tiny", all_non_tiny_tumors, nc_non_tiny)]:
                    self.fit_gmm_model(tumors, nc, cov_type, tol, max_iter, use_cv=use_cv)
                    gmm_model_name = f'gmm_model_{tumor_type}_{nc}_{generate_random_str()}.pkl'
                    with open(os.path.join('gmm', cov_type, gmm_model_name), 'wb') as f:
                        pickle.dump(self.gmm_model, f)
                    if tumor_type == "tiny":
                        self.gmm_model_tiny = self.gmm_model
                    else:
                        self.gmm_model_non_tiny = self.gmm_model
                    print(f"{tumor_type.capitalize()} GMM saved successfully: gmm/{cov_type}/{gmm_model_name}")

            else:
                nc = optimal_components[0]
                self.fit_gmm_model(self.all_tumors, nc, cov_type, tol, max_iter, use_cv=use_cv)
                gmm_model_name = f'gmm_model_global_{nc}_{generate_random_str()}.pkl'
                with open(os.path.join('gmm', cov_type, gmm_model_name), 'wb') as f:
                    pickle.dump(self.gmm_model, f)
                    self.gmm_model_global = self.gmm_model
                print(f"Global GMM saved successfully: gmm/{cov_type}/{gmm_model_name}")

            self.gmm_flag = True

    @staticmethod
    def analyze_tumors_shape(data_dir='datafolds/04_LiTS/label/', output_save_dir='datafolds/04_LiTS/',
                             file_reg='liver_*.nii.gz'):
        label_paths = glob.glob(os.path.join(data_dir, file_reg))
        label_paths.sort()

        valid_ct_name = []

        result_file = os.path.join(output_save_dir, 'tumor_shape_result.txt')
        with open(result_file, 'w') as f:
            for label_path in label_paths:
                print('label_path', label_path)
                file_name = os.path.basename(label_path)

                label = nib.load(label_path)
                raw_label = label.get_fdata()

                tumor_mask = np.zeros_like(raw_label).astype(np.int16)
                tumor_mask[raw_label == 2] = 1

                if len(np.unique(tumor_mask)) > 1:
                    label_numeric, gt_N = ndimage.label(tumor_mask)
                    for segid in range(1, gt_N + 1):
                        extracted_label_numeric = np.uint8(label_numeric == segid)
                        clot_size = np.sum(extracted_label_numeric)
                        if clot_size < 8:
                            continue
                        coords = np.array(np.where(extracted_label_numeric)).T
                        centroid = np.mean(coords, axis=0)
                        distances = cdist([centroid], coords)
                        x_radius = np.max(distances[:, 0])
                        y_radius = np.max(distances[:, 1])
                        z_radius = np.max(distances[:, 2])

                        print('Tumor Shape - X radius:', x_radius, 'Y radius:', y_radius, 'Z radius:', z_radius)
                        f.write(f"Tumor Shape - X radius: {x_radius}, Y radius: {y_radius}, Z radius: {z_radius}\n")

                    if not file_name in valid_ct_name:
                        valid_ct_name.append(file_name)

        with open(result_file, 'a') as f:
            f.write(f"Valid_ct: {len(valid_ct_name)}\n")

    @staticmethod
    def analyze_tumor_type_helper(clot_size, spacing_mm):
        def voxel2R(A):
            return (np.array(A) / 4 * 3 / np.pi) ** (1 / 3)

        def pixel2voxel(A, res=[0.75, 0.75, 0.5]):
            return np.array(A) * (res[0] * res[1] * res[2])

        clot_size_mm = pixel2voxel(clot_size, spacing_mm)
        clot_size_mmR = voxel2R(clot_size_mm)

        if clot_size_mmR <= 10:
            tumor_type = 'tiny'
        elif 10 < clot_size_mmR <= 25:
            tumor_type = 'small'
        elif 25 < clot_size_mmR <= 50:
            tumor_type = 'medium'
        else:
            tumor_type = 'large'
        return tumor_type, clot_size_mm, clot_size_mmR

    @staticmethod
    def analyze_tumors_type(data_dir='datafolds/04_LiTS/label/', output_save_dir='datafolds/04_LiTS/',
                            file_reg='liver_*.nii.gz'):
        tumor_counts = {'tiny': 0, 'small': 0, 'medium': 0, 'large': 0}
        total_clot_size = []
        total_clot_size_mmR = []
        valid_ct_name = []
        label_paths = glob.glob(os.path.join(data_dir, file_reg))
        label_paths.sort()

        result_file = os.path.join(output_save_dir, 'tumor_type_result.txt')
        with open(result_file, 'w') as f:
            for label_path in label_paths:
                print('label_path', label_path)
                file_name = os.path.basename(label_path)

                label = nib.load(label_path)
                pixdim = label.header['pixdim']
                spacing_mm = tuple(pixdim[1:4])
                raw_label = label.get_fdata()

                tumor_mask = np.zeros_like(raw_label).astype(np.int16)
                tumor_mask[raw_label == 2] = 1

                if len(np.unique(tumor_mask)) > 1:
                    label_numeric, gt_N = ndimage.label(tumor_mask)
                    for segid in range(1, gt_N + 1):
                        extracted_label_numeric = np.uint8(label_numeric == segid)
                        clot_size = np.sum(extracted_label_numeric)
                        if clot_size < 8:
                            continue
                        tumor_type, clot_size_mm, clot_size_mmR = TumorAnalyzer.analyze_tumor_type_helper(clot_size,
                                                                                                          spacing_mm)
                        print('tumor_clot_size_mmR', clot_size_mmR, 'tumor_type', tumor_type)

                        if tumor_type in tumor_counts:
                            tumor_counts[tumor_type] += 1
                        else:
                            tumor_counts['large'] += 1

                        total_clot_size.append(clot_size)
                        total_clot_size_mmR.append(clot_size_mmR)
                        if not file_name in valid_ct_name:
                            valid_ct_name.append(file_name)

                        f.write(f"File Name: {file_name}, "
                                f"Tumor Size (pixel): {clot_size}, "
                                f"Tumor Size (voxel): {clot_size}, "
                                f"Tumor Size (mmR): {clot_size_mmR},"
                                f"Tumor Type: {tumor_type}\n")

        with open(result_file, 'a') as f:
            f.write(f"Valid_ct: {len(valid_ct_name)}\n")

            total = sum(tumor_counts.values())
            for tumor_type, count in tumor_counts.items():
                f.write(f"{tumor_type.capitalize()}: {count} ({count / total:.2%}), ")

        return tumor_counts['tiny'], tumor_counts['small'], tumor_counts['medium'], tumor_counts[
            'large'], total_clot_size, total_clot_size_mmR, valid_ct_name

    @staticmethod
    def crop_mask(mask_scan):
        """
        Crops the volume to get the liver mask.
        """
        # for speed_generate_tumor, we only send the liver part into the generate program
        x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
        y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
        z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

        # shrink the boundary
        x_start, x_end = max(0, x_start + 1), min(mask_scan.shape[0], x_end - 1)
        y_start, y_end = max(0, y_start + 1), min(mask_scan.shape[1], y_end - 1)
        z_start, z_end = max(0, z_start + 1), min(mask_scan.shape[2], z_end - 1)

        liver_mask = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]

        return liver_mask

    @staticmethod
    def resize_mask(volume, new_shape):
        """
        Resizes the volume on given shape.
        """
        x_old, y_old, z_old = volume.shape
        x_new, y_new, z_new = new_shape

        # Create grid for interpolation
        x = np.linspace(0, x_old - 1, x_old)
        y = np.linspace(0, y_old - 1, y_old)
        z = np.linspace(0, z_old - 1, z_old)

        new_x = np.linspace(0, x_old - 1, x_new)
        new_y = np.linspace(0, y_old - 1, y_new)
        new_z = np.linspace(0, z_old - 1, z_new)

        # Create interpolation function
        interpolator = interpolate.RegularGridInterpolator((x, y, z), volume, method='nearest', bounds_error=False,
                                                           fill_value=0)

        # Interpolate volume
        new_volume = interpolator((new_x[:, None, None], new_y[None, :, None], new_z[None, None, :]))

        return np.round(new_volume).astype(int)

    @staticmethod
    def resize_mask_new(volume, new_shape):
        """
        Resizes the volume to the given shape using linear interpolation, retaining all pixel values.
        """
        from scipy.ndimage import zoom

        # Calculate the zoom factors for each dimension
        zoom_factors = [new_dim / old_dim for new_dim, old_dim in zip(new_shape, volume.shape)]

        # Perform interpolation
        new_volume = zoom(volume, zoom_factors, order=1)  # order=1 corresponds to linear interpolation

        # Round to nearest integer to retain original pixel values
        new_volume = np.round(new_volume).astype(int)

        return new_volume

    @staticmethod
    def analyze_tumors(label_path, target_volume=(300, 250, 140), tumor_label=2):
        """
        Analyzes tumor information from label data.
        """
        label = nib.load(label_path)
        pixdim = label.header['pixdim']
        spacing_mm = tuple(pixdim[1:4])
        label_data = label.get_fdata()

        organ_mask = TumorAnalyzer.crop_mask(label_data)
        organ_mask = TumorAnalyzer.resize_mask(organ_mask, target_volume)
        # organ_mask = TumorAnalyzer.resize_mask_new(organ_mask, target_volume)

        tumor_mask = np.zeros_like(organ_mask).astype(np.int16)
        tumor_mask[organ_mask == tumor_label] = 1

        tumors = []

        if len(np.unique(tumor_mask)) > 1:
            label_numeric, gt_N = ndimage.label(tumor_mask)
            for segid in range(1, gt_N + 1):
                extracted_label_numeric = np.uint8(label_numeric == segid)
                clot_size = np.sum(extracted_label_numeric)
                if clot_size < 8:
                    continue
                tumor_position = ndimage.measurements.center_of_mass(extracted_label_numeric)
                tumor_type, _, _ = TumorAnalyzer.analyze_tumor_type_helper(clot_size, spacing_mm)
                tumor = Tumor(position=tumor_position, type=tumor_type)
                tumors.append(tumor)

        return tumors


    def get_gmm_model(self, model_type='global'):
        """
        Returns the trained GMM model.
        """
        models = {
            'tiny': self.gmm_model_tiny,
            'non_tiny': self.gmm_model_non_tiny,
            'global': self.gmm_model_global
        }

        return models.get(model_type)

    def get_all_tumors(self, data_dir, save_folder, save=False, parallel=True):
        tumors_path = os.path.join(data_dir, 'tumors.npy')
        if os.path.exists(tumors_path):
            tumor_data = np.load(tumors_path, allow_pickle=True)
            if len(tumor_data) > 850:
                print(f"tumors.npy found with {len(tumor_data)} tumors. Skipping data loading.")
                return tumor_data.tolist()
            else:
                print(f"tumors.npy found but only {len(tumor_data)} tumors. Loading data.")
                self.load_data(data_dir, parallel=parallel)
        else:
            print("tumors.npy not found. Loading data.")
            self.load_data(data_dir, parallel=parallel)

        if not os.path.exists(tumors_path) and save:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, 'tumors.npy')
            np.save(save_path, np.array(self.all_tumors, dtype=object))
            print('Tumors data saved to {}'.format(save_path))

        return self.all_tumors

