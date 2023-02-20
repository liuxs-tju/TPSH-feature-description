//////////////////////////////////////////////////////////////////////////////////
// Trigonometric Projection Statistics Histograms (TPSH)
// Author: Xingsheng Liu @Tongji University
// Note: Full code may be obtained from lxs@tongji.edu.cn upon resonable request.
//////////////////////////////////////////////////////////////////////////////////

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

	/* This method computes local reference frame for a given point. */
	void computeLRF(const pcl::PointXYZ point, const pcl::PointCloud<pcl::PointXYZ>& local_cloud, Eigen::Matrix3f &lrf_matrix)
	{

	}

	/* This method encodes both spatial information and geometrical information into individual statistics histograms. */
	void getFeatureHistograms(const std::vector<Eigen::Vector3f>& view_position, const std::vector<Eigen::Matrix3f>& view_rotation,
		const pcl::PointCloud<pcl::PointXYZ>& transformed_cloud, std::vector<float> &histograms)
	{

	}

	/* This method computes TPSH descriptors at all keypoints. */
	void computeDescriptors(pcl::PointCloud<pcl::TPSH245>& descriptors)
	{

	}
};
