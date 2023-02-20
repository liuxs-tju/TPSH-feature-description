//////////////////////////////////////////////////////////
// Trigonometric Projection Statistics Histograms (TPSH)
// Author: Xingsheng Liu @Tongji University
// Date: 2023-02-20
////////////////////////////////////////////////////////

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>  
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/features/feature.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include <numeric>
#include <algorithm>


class TPSHEstimation
{
private:
	float support_radius_;

	int mu_;
	float sigma_N_;
	float sigma_H_;
	float theta_;
	float lambda_;

	unsigned int number_of_spatial_bins_;
	unsigned int number_of_geometrical_bins_;

	bool lrf_defined_;
	unsigned int number_of_threads_;

	std::vector<Eigen::Vector3f> view_position_;
	std::vector<Eigen::Matrix3f> view_rotation_;

public:
	pcl::PointCloud<pcl::PointXYZ> cloud_;
	pcl::PointCloud<pcl::PointXYZ> keypoints_;
	pcl::KdTreeFLANN<pcl::PointXYZ> tree_;
	pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr frames_;


public:
	TPSHEstimation() :
		number_of_spatial_bins_(7),
		number_of_geometrical_bins_(7),
		number_of_threads_(1),
		support_radius_(1.0f),
		mu_(10),
		sigma_N_(0.7f),
		sigma_H_(0.3f),
		theta_(75.0f),
		lambda_(0.6f),
		lrf_defined_(false)
	{
	}

	void setSearchSurface(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
	{
		if (!cloud->empty())
		{
			cloud_ = *cloud;
			tree_.setInputCloud(cloud);
		}
	}

	void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &keypoints)
	{
		if (!keypoints->empty())
		{
			keypoints_ = *keypoints;
		}
	}

	void setInputReferenceFrames(pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr &frames)
	{
		frames_ = frames;
		lrf_defined_ = true;
	}

	void getInputReferenceFrames(pcl::PointCloud<pcl::ReferenceFrame>::ConstPtr &frames)
	{
		frames = frames_;
	}

	void setNumberOfSpatialBins(unsigned int number_of_bins)
	{
		if (number_of_bins != 0)
		{
			number_of_spatial_bins_ = number_of_bins;
		}
	}

	void setNumberOfGeometricalBins(unsigned int number_of_bins)
	{
		if (number_of_bins != 0)
		{
			number_of_geometrical_bins_ = number_of_bins;
		}
	}

	void setSupportRadius(float support_radius)
	{
		if (support_radius > 0.0f)
		{
			support_radius_ = support_radius;
		}
	}

	void setMu(int mu)
	{
		if (mu > 0)
		{
			mu_ = mu;
		}
	}

	void setSigmaN(float sigma_N)
	{
		if (sigma_N > 0.0f)
		{
			sigma_N_ = sigma_N;
		}
	}

	void setSigmaH(float sigma_H)
	{
		if (sigma_H > 0.0f)
		{
			sigma_H_ = sigma_H;
		}
	}

	void setTheta(float theta)
	{
		theta_ = theta;
	}

	void setLambda(float lambda)
	{
		if (lambda >= 0.0f && lambda <= 1.0f)
		{
			lambda_ = lambda;
		}
	}

	void setNumberOfThreads(unsigned int number_of_threads)
	{
		if (number_of_threads != 0)
		{
			number_of_threads_ = number_of_threads;
		}
	}

	/* This method initializes the parameters required for trigonometric projection. */
	void initViewParameters()
	{
		const float rad = M_PI / 180.0f;
		const float cosine = cos(theta_ * rad);
		const float sine = sin(theta_ * rad);

		std::vector<Eigen::Vector3f> view_direction(2);
		view_direction[0] << sine, 0.0f, -cosine;
		view_direction[1] << -sine, 0.0f, -cosine;

		view_position_.resize(2);
		view_position_[0] = -3.0f * support_radius_ * view_direction[0];
		view_position_[1] = -3.0f * support_radius_ * view_direction[1];

		view_rotation_.resize(2);
		Eigen::Vector3f unit(0.0f, 0.0f, 1.0f);
		for (unsigned int i_view = 0; i_view < 2; i_view++)
		{
			float angle = acos(view_direction[i_view].dot(unit));
			Eigen::Vector3f axis = view_direction[i_view].cross(unit);
			view_rotation_[i_view] = Eigen::AngleAxisf(angle, axis.normalized());
		}
	}

	/* This method extracts all neighbors for a given point in the support region. */
	void getLocalSurface(const pcl::PointXYZ point, pcl::PointCloud<pcl::PointXYZ>& local_cloud)
	{
		std::vector<int> indices;
		std::vector<float> distances;
		tree_.radiusSearch(point, support_radius_, indices, distances);
		pcl::copyPointCloud(cloud_, indices, local_cloud);
	}

	/* This method transforms all neighbors for a given point in the support region to the local reference frame. */
	void transformCloud(const pcl::PointXYZ point, const pcl::PointCloud<pcl::PointXYZ>& local_cloud, const Eigen::Matrix3f& matrix, pcl::PointCloud<pcl::PointXYZ>& transformed_cloud)
	{
		const unsigned int number_of_points = static_cast <unsigned int> (local_cloud.size());
		transformed_cloud.points.resize(number_of_points);

		for (unsigned int i = 0; i < number_of_points; i++)
		{
			Eigen::Vector3f transformed_point(
				local_cloud.points[i].x - point.x,
				local_cloud.points[i].y - point.y,
				local_cloud.points[i].z - point.z);

			transformed_point = matrix * transformed_point;

			pcl::PointXYZ tform_point;
			tform_point.x = transformed_point(0);
			tform_point.y = transformed_point(1);
			tform_point.z = transformed_point(2);
			transformed_cloud.points[i] = tform_point;
		}
	}

	/* This method calculates the eigen values and eigen vectors for a given covariance matrix. */
	void computeEigenVectors(const Eigen::Matrix3f& matrix, Eigen::Vector3f& major_axis, Eigen::Vector3f& middle_axis, Eigen::Vector3f& minor_axis) const
	{
		Eigen::EigenSolver <Eigen::Matrix3f> eigen_solver;
		eigen_solver.compute(matrix);

		Eigen::EigenSolver <Eigen::Matrix3f>::EigenvectorsType eigen_vectors;
		Eigen::EigenSolver <Eigen::Matrix3f>::EigenvalueType eigen_values;
		eigen_vectors = eigen_solver.eigenvectors();
		eigen_values = eigen_solver.eigenvalues();

		unsigned int temp = 0;
		unsigned int major_index = 0;
		unsigned int middle_index = 1;
		unsigned int minor_index = 2;

		if (eigen_values.real() (major_index) < eigen_values.real() (middle_index))
		{
			temp = major_index;
			major_index = middle_index;
			middle_index = temp;
		}

		if (eigen_values.real() (major_index) < eigen_values.real() (minor_index))
		{
			temp = major_index;
			major_index = minor_index;
			minor_index = temp;
		}

		if (eigen_values.real() (middle_index) < eigen_values.real() (minor_index))
		{
			temp = minor_index;
			minor_index = middle_index;
			middle_index = temp;
		}

		major_axis = eigen_vectors.col(major_index).real();
		middle_axis = eigen_vectors.col(middle_index).real();
		minor_axis = eigen_vectors.col(minor_index).real();
	}

	/* This method computes local reference frame for a given point. */
	void computeLRF(const pcl::PointXYZ point, const pcl::PointCloud<pcl::PointXYZ>& local_cloud, Eigen::Matrix3f &lrf_matrix)
	{
		Eigen::Vector3f feature_point(point.x, point.y, point.z);

		const unsigned int number_of_points = static_cast <unsigned int> (local_cloud.size());
		
		std::vector <float> weight_distance(number_of_points);
		std::vector <float> weight_density(number_of_points);
		std::vector <float> weight_height(number_of_points);
		std::vector<float> total_weight(number_of_points);

		std::vector<Eigen::Vector3f> neighbor_pair(number_of_points);
		std::vector <float> distance_neighbors(number_of_points);
		float min_distance = FLT_MAX;
		float max_distance = -FLT_MAX;

		std::vector<int> nn_indices(mu_);
		std::vector<float> nn_dists(mu_);

		for (int i = 0; i < static_cast<int> (number_of_points); i++)
		{
			tree_.nearestKSearch(local_cloud.points[i], mu_, nn_indices, nn_dists);

			float sum_distances = 0.0f;
			for (int j = 0; j < mu_; j++)
				sum_distances += sqrt(nn_dists[j]);
			distance_neighbors[i] = sum_distances / (float)mu_;
		}

		nn_indices.clear();
		nn_dists.clear();

		for (unsigned int i = 0; i < number_of_points; i++)
		{
			if (min_distance > distance_neighbors[i]) min_distance = distance_neighbors[i];
			if (max_distance < distance_neighbors[i]) max_distance = distance_neighbors[i];

			Eigen::Vector3f neighbor_point(local_cloud.points[i].x, local_cloud.points[i].y, local_cloud.points[i].z);
			neighbor_pair[i] = neighbor_point - feature_point;
		}

		//covariance analysis for z axis
		float distance_range = max_distance - min_distance;
		std::vector<Eigen::Matrix3f> weighted_matrices(number_of_points);
		for (int i = 0; i < static_cast<int> (number_of_points); i++)
		{
			weight_distance[i] = 1.0f - neighbor_pair[i].norm() / support_radius_;
			weight_density[i] = exp(-pow(max_distance - distance_neighbors[i], 2.0f) / (2.0f * pow(sigma_N_ * distance_range, 2.0f)));
			
			total_weight[i] = weight_distance[i] * weight_density[i];
			weighted_matrices[i] = total_weight[i] * neighbor_pair[i] * (neighbor_pair[i].transpose());
		}

		Eigen::Matrix3f covariance_matrix;
		covariance_matrix.setZero();
		float sum_weight = 0.0f;
		for (unsigned int i = 0; i < number_of_points; i++)
		{
			sum_weight += total_weight[i];
			covariance_matrix += weighted_matrices[i];
		}
		covariance_matrix /= sum_weight;

		Eigen::Vector3f v1, v2, v3;
		computeEigenVectors(covariance_matrix, v1, v2, v3);

		//sign disambiguation for z axis
		Eigen::Vector3f sum_neighbor(0.0f, 0.0f, 0.0f);
		for (unsigned int i = 0; i < number_of_points; i++)
			sum_neighbor += total_weight[i] * (-neighbor_pair[i]);

		float factor_zaxis = v3.dot(sum_neighbor);
		if (factor_zaxis < 0.0f) v3 = -v3;


		//covariance analysis for x axis
		std::vector<Eigen::Vector3f> neighbor_proj(number_of_points);
		std::vector <float> height(number_of_points);
		float min_height = FLT_MAX;
		float max_height = -FLT_MAX;
		for (unsigned int i = 0; i < number_of_points; i++)
		{
			float dot_product = v3.dot(neighbor_pair[i]);
			neighbor_proj[i] = neighbor_pair[i] - dot_product * v3;
			
			height[i] = abs(dot_product);
			if (min_height > height[i]) min_height = height[i];
			if (max_height < height[i]) max_height = height[i];
		}
		float height_range = max_height - min_height;

		for (int i = 0; i < static_cast<int> (number_of_points); i++)
		{
			weight_height[i] = exp(-pow(max_height - height[i], 2.0f) / (2.0f * pow(sigma_H_ * height_range, 2.0f)));

			total_weight[i] = weight_distance[i] * weight_density[i] * weight_height[i];
			weighted_matrices[i] = total_weight[i] * neighbor_proj[i] * (neighbor_proj[i].transpose());
		}

		covariance_matrix.setZero();
		sum_weight = 0.0f;
		for (unsigned int i = 0; i < number_of_points; i++)
		{
			sum_weight += total_weight[i];
			covariance_matrix += weighted_matrices[i];
		}
		covariance_matrix /= sum_weight;

		Eigen::Vector3f vx1, vx2, vx3;
		computeEigenVectors(covariance_matrix, vx1, vx2, vx3);

		//sign disambiguation for x axis
		sum_neighbor << 0.0f, 0.0f, 0.0f;
		for (unsigned int i = 0; i < number_of_points; i++)
			sum_neighbor += total_weight[i] * neighbor_proj[i];

		float factor_xaxis = vx1.dot(sum_neighbor);
		if (factor_xaxis < 0.0f) vx1 = -vx1;
		v1 = vx1;

		//y axis
		v2 = v3.cross(v1);

		//output
		lrf_matrix.row(0) = v1;
		lrf_matrix.row(1) = v2;
		lrf_matrix.row(2) = v3;
	}

	/* This method encodes both spatial information and geometrical information into individual statistics histograms. */
	void getFeatureHistograms(const std::vector<Eigen::Vector3f>& view_position, const std::vector<Eigen::Matrix3f>& view_rotation, 
		const pcl::PointCloud<pcl::PointXYZ>& transformed_cloud, std::vector<float> &histograms)
	{
		histograms.clear();

		const unsigned int number_of_points = static_cast <unsigned int> (transformed_cloud.size());
		const float lambda_spatial = lambda_;
		const float lambda_geometric = 1.0f - lambda_;

		//preparation
		std::vector<Eigen::Vector3f> object_points(number_of_points);
		for (unsigned int i_point = 0; i_point < number_of_points; i_point++)
		{
			object_points[i_point] << transformed_cloud.points[i_point].x, transformed_cloud.points[i_point].y, transformed_cloud.points[i_point].z;
		}

		//perspective projection
		std::vector<std::vector<Eigen::Vector3f>> image_points(2);
		std::vector<std::vector<Eigen::Vector3f>> project_vectors(2);
		std::vector<std::vector<Eigen::Vector3f>> rotated_vectors(2);
		std::vector<Eigen::Vector2f> min_coord(2), max_coord(2);

		for (unsigned int i_view = 0; i_view < 2; i_view++)
		{
			image_points[i_view].resize(number_of_points);
			project_vectors[i_view].resize(number_of_points);
			rotated_vectors[i_view].resize(number_of_points);
			min_coord[i_view] << FLT_MAX, FLT_MAX;
			max_coord[i_view] << -FLT_MAX, -FLT_MAX;

			for (unsigned int i_point = 0; i_point < number_of_points; i_point++)
			{
				project_vectors[i_view][i_point] = object_points[i_point] - view_position[i_view];
				rotated_vectors[i_view][i_point] = view_rotation[i_view] * project_vectors[i_view][i_point];

				image_points[i_view][i_point] = rotated_vectors[i_view][i_point] * support_radius_ / rotated_vectors[i_view][i_point](2);
				if (min_coord[i_view](0) > image_points[i_view][i_point](0)) min_coord[i_view](0) = image_points[i_view][i_point](0);
				if (min_coord[i_view](1) > image_points[i_view][i_point](1)) min_coord[i_view](1) = image_points[i_view][i_point](1);
				if (max_coord[i_view](0) < image_points[i_view][i_point](0)) max_coord[i_view](0) = image_points[i_view][i_point](0);
				if (max_coord[i_view](1) < image_points[i_view][i_point](1)) max_coord[i_view](1) = image_points[i_view][i_point](1);
			}
		}

		//trigonometric measurement
		std::vector<Eigen::Vector3f> geometry_attributes(number_of_points);
		Eigen::Vector3f min_value(FLT_MAX, FLT_MAX, FLT_MAX);
		Eigen::Vector3f max_value(-FLT_MAX, -FLT_MAX, -FLT_MAX);

		for (unsigned int i_point = 0; i_point < number_of_points; i_point++)
		{
			geometry_attributes[i_point](0) = project_vectors[0][i_point].norm();
			geometry_attributes[i_point](1) = project_vectors[1][i_point].norm();
			geometry_attributes[i_point](2) = acos(project_vectors[0][i_point].dot(project_vectors[1][i_point]) / (geometry_attributes[i_point](0) * geometry_attributes[i_point](1))); //2022-06-28

			if (min_value(0) > geometry_attributes[i_point](0)) min_value(0) = geometry_attributes[i_point](0);
			if (min_value(1) > geometry_attributes[i_point](1)) min_value(1) = geometry_attributes[i_point](1);
			if (min_value(2) > geometry_attributes[i_point](2)) min_value(2) = geometry_attributes[i_point](2);
			if (max_value(0) < geometry_attributes[i_point](0)) max_value(0) = geometry_attributes[i_point](0);
			if (max_value(1) < geometry_attributes[i_point](1)) max_value(1) = geometry_attributes[i_point](1);
			if (max_value(2) < geometry_attributes[i_point](2)) max_value(2) = geometry_attributes[i_point](2);
		}

		//spatial information
		std::vector<Eigen::MatrixXf> distribution_matrix(2);
		std::vector<std::vector<float>> distribution_hist(2);

		for (unsigned int i_view = 0; i_view < 2; i_view++)
		{
			distribution_matrix[i_view].resize(number_of_spatial_bins_, number_of_spatial_bins_);
			distribution_matrix[i_view].setZero();
			distribution_hist[i_view].resize(number_of_spatial_bins_ * number_of_spatial_bins_);

			const float u_bin_length = (max_coord[i_view](0) - min_coord[i_view](0)) / number_of_spatial_bins_;
			const float v_bin_length = (max_coord[i_view](1) - min_coord[i_view](1)) / number_of_spatial_bins_;

			for (unsigned int i_point = 0; i_point < number_of_points; i_point++)
			{
				const float u_length = image_points[i_view][i_point](0) - min_coord[i_view](0);
				const float v_length = image_points[i_view][i_point](1) - min_coord[i_view](1);

				const float u_ratio = u_length / u_bin_length;
				unsigned int col = static_cast <unsigned int> (u_ratio);
				if (col == number_of_spatial_bins_) col--;

				const float v_ratio = v_length / v_bin_length;
				unsigned int row = static_cast <unsigned int> (v_ratio);
				if (row == number_of_spatial_bins_) row--;

				distribution_matrix[i_view](row, col) += 1.0f;
			}

			distribution_matrix[i_view] /= static_cast<float> (number_of_points);

			for (unsigned int i_row = 0; i_row < number_of_spatial_bins_; i_row++)
			{
				for (unsigned int i_col = 0; i_col < number_of_spatial_bins_; i_col++)
				{
					unsigned int index = i_row * number_of_spatial_bins_ + i_col;
					distribution_hist[i_view][index] = lambda_spatial * distribution_matrix[i_view](i_row, i_col);
				}
			}

			histograms.insert(histograms.end(), distribution_hist[i_view].begin(), distribution_hist[i_view].end());
		}

		//geometrical information
		const unsigned int coord[3][3] = {
          {0, 1, 2},
          {0, 2, 1},
          {1, 2, 0} };

		std::vector<Eigen::MatrixXf> geometry_matrix(3);
		std::vector<std::vector<float>> geometry_hist(3);

		for (unsigned int i_geom = 0; i_geom < 3; i_geom++)
		{
			geometry_matrix[i_geom].resize(number_of_geometrical_bins_, number_of_geometrical_bins_);
			geometry_matrix[i_geom].setZero();
			geometry_hist[i_geom].resize(number_of_geometrical_bins_ * number_of_geometrical_bins_);

			const float u_div_length = (max_value(coord[i_geom][0]) - min_value(coord[i_geom][0])) / number_of_geometrical_bins_;
			const float v_div_length = (max_value(coord[i_geom][1]) - min_value(coord[i_geom][1])) / number_of_geometrical_bins_;

			for (unsigned int i_point = 0; i_point < number_of_points; i_point++)
			{
				const float u_length = geometry_attributes[i_point](coord[i_geom][0]) - min_value(coord[i_geom][0]);
				const float v_length = geometry_attributes[i_point](coord[i_geom][1]) - min_value(coord[i_geom][1]);

				const float u_ratio = u_length / u_div_length;
				unsigned int col = static_cast <unsigned int> (u_ratio);
				if (col == number_of_geometrical_bins_) col--;

				const float v_ratio = v_length / v_div_length;
				unsigned int row = static_cast <unsigned int> (v_ratio);
				if (row == number_of_geometrical_bins_) row--;

				geometry_matrix[i_geom](row, col) += 1.0f;
			}

			geometry_matrix[i_geom] /= static_cast<float> (number_of_points);

			for (unsigned int i_row = 0; i_row < number_of_geometrical_bins_; i_row++)
			{
				for (unsigned int i_col = 0; i_col < number_of_geometrical_bins_; i_col++)
				{
					unsigned int index = i_row * number_of_geometrical_bins_ + i_col;
					geometry_hist[i_geom][index] = lambda_geometric * geometry_matrix[i_geom](i_row, i_col);
				}
			}

			histograms.insert(histograms.end(), geometry_hist[i_geom].begin(), geometry_hist[i_geom].end());
		}
	}

	/* This method computes TPSH descriptors at all keypoints. */
	void computeDescriptors(pcl::PointCloud<pcl::TPSH245>& descriptors)
	{
		initViewParameters();

		int number_of_keypoints = static_cast <int> (keypoints_.size());
		descriptors.points.resize(number_of_keypoints);
		descriptors.height = keypoints_.height;
		descriptors.width = keypoints_.width;

		if (lrf_defined_)
		{
#pragma omp parallel for default(shared) num_threads(number_of_threads_)
			for (int i_point = 0; i_point < number_of_keypoints; i_point++)
			{
				pcl::PointCloud<pcl::PointXYZ> local_cloud;
				getLocalSurface(keypoints_.points[i_point], local_cloud);

				Eigen::Matrix3f lrf_matrix;
				pcl::ReferenceFrame current_frame = frames_->points[i_point];
				lrf_matrix << current_frame.x_axis[0], current_frame.x_axis[1], current_frame.x_axis[2],
					current_frame.y_axis[0], current_frame.y_axis[1], current_frame.y_axis[2],
					current_frame.z_axis[0], current_frame.z_axis[1], current_frame.z_axis[2];

				pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
				transformCloud(keypoints_.points[i_point], local_cloud, lrf_matrix, transformed_cloud);

				std::vector<float> histograms;
				getFeatureHistograms(view_position_, view_rotation_, transformed_cloud, histograms);

				for (unsigned int i_dim = 0; i_dim < static_cast<unsigned int> (histograms.size()); i_dim++)
				{
					if (std::isnan(histograms[i_dim]))
						descriptors.points[i_point].histogram[i_dim] = 0.0f;
					else
						descriptors.points[i_point].histogram[i_dim] = histograms[i_dim];
				}
			}
		}
		else
		{
			pcl::PointCloud<pcl::ReferenceFrame>::Ptr frames(new pcl::PointCloud<pcl::ReferenceFrame>);
			frames->resize(number_of_keypoints);

#pragma omp parallel for default(shared) num_threads(number_of_threads_)
			for (int i_point = 0; i_point < number_of_keypoints; i_point++)
			{
				pcl::PointCloud<pcl::PointXYZ> local_cloud;
				getLocalSurface(keypoints_.points[i_point], local_cloud);

				Eigen::Matrix3f lrf_matrix;
				computeLRF(keypoints_.points[i_point], local_cloud, lrf_matrix);

				pcl::ReferenceFrame current_frame;
				for (int d = 0; d < 3; d++)
				{
					current_frame.x_axis[d] = lrf_matrix.row(0)[d];
					current_frame.y_axis[d] = lrf_matrix.row(1)[d];
					current_frame.z_axis[d] = lrf_matrix.row(2)[d];
				}
				frames->points[i_point] = current_frame;

				pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
				transformCloud(keypoints_.points[i_point], local_cloud, lrf_matrix, transformed_cloud);

				std::vector<float> histograms;
				getFeatureHistograms(view_position_, view_rotation_, transformed_cloud, histograms);

				for (unsigned int i_dim = 0; i_dim < static_cast<unsigned int> (histograms.size()); i_dim++)
				{
					if (std::isnan(histograms[i_dim]))
						descriptors.points[i_point].histogram[i_dim] = 0.0f;
					else
						descriptors.points[i_point].histogram[i_dim] = histograms[i_dim];
				}
			}

			frames_ = frames;
		}
	}
};
