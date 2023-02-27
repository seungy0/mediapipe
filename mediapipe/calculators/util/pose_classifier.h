//
// Created by seunggyu on 2023/02/27.
//

#ifndef MEDIAPIPE_POSE_CLASSIFIER_H
#define MEDIAPIPE_POSE_CLASSIFIER_H


#include "pose_embedder.h"
namespace mediapipe {
    class PoseSample {
    public:
        PoseSample(std::string name, NormalizedLandmarkList landmarks, std::string class_name,
                   std::vector <std::vector<float>> embedding);

        std::string name;
        NormalizedLandmarkList landmarks;
        std::string class_name;
        std::vector <std::vector<float>> embedding;
    };

    class PoseClassifier {
    public:
        PoseClassifier();

        int call(const NormalizedLandmarkList &landmarks);

    private:
        PoseEmbedder _pose_embedder;
        std::vector<PoseSample *> pose_sample;
        std::string pose_samples_folder;
        std::string file_extension;
        int _top_n_by_max_distance = 30;
        int _top_n_by_mean_distance = 10;
    };
}

#endif //MEDIAPIPE_POSE_CLASSIFIER_H
