//
// Created by seunggyu on 2023/02/21.
//

#ifndef MEDIAPIPE_POSE_EMBEDDER_H
#define MEDIAPIPE_POSE_EMBEDDER_H

#include <string>
#include <vector>

#include "mediapipe/framework/formats/landmark.pb.h"
namespace mediapipe {
    class PoseEmbedder {
    public:
        PoseEmbedder(float torso_size_multiplier = 2.5);

        std::vector <std::vector<float>> call(const NormalizedLandmarkList &landmarks);

    private:
        const double _torso_size_multiplier;
        const std::vector <std::string> _landmark_names;

        NormalizedLandmarkList normalize_pose_landmarks(const NormalizedLandmarkList &landmarks);

        NormalizedLandmark get_pose_center(const NormalizedLandmarkList &landmarks);

        std::vector <std::vector<float>>
        get_pose_distance_embedding(const NormalizedLandmarkList &landmarks);

        float get_pose_size(const NormalizedLandmarkList &landmarks, float torso_size_multiplier);

        NormalizedLandmark
        get_average_by_names(const NormalizedLandmarkList &landmarks, const std::string &name_from,
                             const std::string &name_to);

        std::vector<float>
        get_distance_by_names(const NormalizedLandmarkList &landmarks, const std::string &name_from,
                              const std::string &name_to);

        std::vector<float>
        get_distance(const NormalizedLandmark &lmk_from, const NormalizedLandmark &lmk_to);
    };
}
#endif //MEDIAPIPE_POSE_EMBEDDER_H
