// Written by Seunggyu Lee (https://github.com/seungy0)
// 2023-02
#pragma once
#include <memory>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

    namespace {

        constexpr char kPoseTag[] = "POSE";
        constexpr char kNormalizedLandmarksTag[] = "LANDMARKS";

        int ConvertLandmarksToPose(const NormalizedLandmarkList& landmarks) {
            int pose;
            // TODO: Implement pose recognition algorithm.
            return 1;
        }
        class PoseEmbedder {
        public:
            PoseEmbedder(float torso_size_multiplier = 2.5);
            std::vector<std::vector<float>> call(const NormalizedLandmarkList& landmarks);
        private:
            const double _torso_size_multiplier;
            const std::vector<std::string> _landmark_names;
            NormalizedLandmarkList normalize_pose_landmarks(const NormalizedLandmarkList& landmarks);
            NormalizedLandmark get_pose_center(const NormalizedLandmarkList& landmarks);
            std::vector<std::vector<float>> get_pose_distance_embedding(const NormalizedLandmarkList& landmarks);
            float get_pose_size(const NormalizedLandmarkList& landmarks, float torso_size_multiplier);
            NormalizedLandmark get_average_by_names(const NormalizedLandmarkList& landmarks, const std::string& name_from, const std::string& name_to);
            std::vector<float> get_distance_by_names(const NormalizedLandmarkList& landmarks, const std::string& name_from, const std::string& name_to);
            std::vector<float> get_distance(const NormalizedLandmark& lmk_from, const NormalizedLandmark& lmk_to);
        };
        PoseEmbedder::PoseEmbedder(float torso_size_multiplier):_torso_size_multiplier(torso_size_multiplier),
            _landmark_names({
                    "nose",
                    "left_eye_inner",
                    "left_eye",
                    "left_eye_outer",
                    "right_eye_inner",
                    "right_eye",
                    "right_eye_outer",
                    "left_ear",
                    "right_ear",
                    "mouth_left",
                    "mouth_right",
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                    "left_pinky_1",
                    "right_pinky_1",
                    "left_index_1",
                    "right_index_1",
                    "left_thumb_2",
                    "right_thumb_2",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                    "left_heel",
                    "right_heel",
                    "left_foot_index",
                    "right_foot_index"
            }) {}
        std::vector<std::vector<float>> PoseEmbedder::call(const NormalizedLandmarkList& landmarks) {
            if (landmarks.landmark_size() != 33) {
                return std::vector<std::vector<float>>();
            }
            return get_pose_distance_embedding(normalize_pose_landmarks(landmarks));
        }
        NormalizedLandmarkList PoseEmbedder::normalize_pose_landmarks(const NormalizedLandmarkList& landmarks){
            NormalizedLandmarkList normalized_landmarks;
            NormalizedLandmark pose_center = get_pose_center(landmarks);
            double pose_size = get_pose_size(landmarks, _torso_size_multiplier);
            for (const auto& landmark : landmarks.landmark()) {
                NormalizedLandmark* normalized_landmark = normalized_landmarks.add_landmark();
                normalized_landmark->set_x((landmark.x() - pose_center.x()) / pose_size);
                normalized_landmark->set_y((landmark.y() - pose_center.y()) / pose_size);
                normalized_landmark->set_z((landmark.z() - pose_center.z()) / pose_size);
            }
            return normalized_landmarks;
        }
        NormalizedLandmark PoseEmbedder::get_pose_center(const NormalizedLandmarkList& landmarks) {
            NormalizedLandmark pose_center;
            pose_center.set_x((landmarks.landmark(23).x() + landmarks.landmark(24).x())/2.0);
            pose_center.set_y((landmarks.landmark(23).y() + landmarks.landmark(24).y())/2.0);
            pose_center.set_z((landmarks.landmark(23).z() + landmarks.landmark(24).z())/2.0);
            return pose_center;
        }
        float PoseEmbedder::get_pose_size(const NormalizedLandmarkList& landmarks, float torso_size_multiplier) {
            NormalizedLandmark left_hip = landmarks.landmark(23);
            NormalizedLandmark right_hip = landmarks.landmark(24);
            NormalizedLandmark hips;
            hips.set_x((left_hip.x() + right_hip.x())/2.0);
            hips.set_y((left_hip.y() + right_hip.y())/2.0);
            NormalizedLandmark left_shoulder = landmarks.landmark(11);
            NormalizedLandmark right_shoulder = landmarks.landmark(12);
            NormalizedLandmark shoulders;
            shoulders.set_x((left_shoulder.x() + right_shoulder.x())/2.0);
            shoulders.set_y((left_shoulder.y() + right_shoulder.y())/2.0);
            float torso_size = sqrt(pow(shoulders.x() - hips.x(), 2) + pow(shoulders.y() - hips.y(), 2));
            NormalizedLandmark pose_center = get_pose_center(landmarks);
            float max_dist = 0;
            for (const auto& landmark : landmarks.landmark()) {
                float dist = sqrt(pow(landmark.x() - pose_center.x(), 2) + pow(landmark.y() - pose_center.y(), 2) + pow(landmark.z() - pose_center.z(), 2));
                if (dist > max_dist) {
                    max_dist = dist;
                }
            }
            return std::max(torso_size * torso_size_multiplier, max_dist);
        }
        std::vector<std::vector<float>> PoseEmbedder::get_pose_distance_embedding(const NormalizedLandmarkList& landmarks) {
            std::vector<std::vector<float>> embedding;
            embedding.push_back(get_distance(get_average_by_names(landmarks,"left_hip","right_hip"), get_average_by_names(landmarks,"left_shoulder","right_shoulder")));

            embedding.push_back(get_distance_by_names(landmarks,"left_shoulder","left_elbow"));
            embedding.push_back(get_distance_by_names(landmarks,"right_shoulder","right_elbow"));

            // add embedding for arms
            embedding.push_back(get_distance_by_names(landmarks,"left_elbow","left_wrist"));
            embedding.push_back(get_distance_by_names(landmarks,"right_elbow","right_wrist"));

            // add embedding for legs
            embedding.push_back(get_distance_by_names(landmarks,"left_hip","left_knee"));
            embedding.push_back(get_distance_by_names(landmarks,"right_hip","right_knee"));
            embedding.push_back(get_distance_by_names(landmarks,"left_knee","left_ankle"));
            embedding.push_back(get_distance_by_names(landmarks,"right_knee","right_ankle"));

            // add embedding for two joints
            embedding.push_back(get_distance_by_names(landmarks,"left_shoulder","left_wrist"));
            embedding.push_back(get_distance_by_names(landmarks,"right_shoulder","right_wrist"));
            embedding.push_back(get_distance_by_names(landmarks,"left_hip","left_ankle"));
            embedding.push_back(get_distance_by_names(landmarks,"right_hip","right_ankle"));

            // add embedding from hip to wrist
            embedding.push_back(get_distance_by_names(landmarks,"left_hip","left_wrist"));
            embedding.push_back(get_distance_by_names(landmarks,"right_hip","right_wrist"));

            // add embedding from shoulder to ankle
            embedding.push_back(get_distance_by_names(landmarks,"left_shoulder","left_ankle"));
            embedding.push_back(get_distance_by_names(landmarks,"right_shoulder","right_ankle"));

            // add embedding for cross body
            embedding.push_back(get_distance_by_names(landmarks,"left_elbow","right_elbow"));
            embedding.push_back(get_distance_by_names(landmarks,"left_knee","right_knee"));
            embedding.push_back(get_distance_by_names(landmarks,"left_wrist","right_wrist"));
            embedding.push_back(get_distance_by_names(landmarks,"left_ankle","right_ankle"));

            return embedding;
        }

        NormalizedLandmark PoseEmbedder::get_average_by_names(const NormalizedLandmarkList& landmarks, const std::string& name_from, const std::string& name_to) {
            const NormalizedLandmark& lmk_from = landmarks.landmark(std::find(_landmark_names.begin(), _landmark_names.end(), name_from) - _landmark_names.begin());
            const NormalizedLandmark& lmk_to = landmarks.landmark(std::find(_landmark_names.begin(), _landmark_names.end(), name_to) - _landmark_names.begin());
            NormalizedLandmark lmk_avg;
            lmk_avg.set_x((lmk_from.x() + lmk_to.x())/2.0);
            lmk_avg.set_y((lmk_from.y() + lmk_to.y())/2.0);
            lmk_avg.set_z((lmk_from.z() + lmk_to.z())/2.0);
            return lmk_avg;
        }

        std::vector<float> PoseEmbedder::get_distance_by_names(const NormalizedLandmarkList& landmarks, const std::string& name_from, const std::string& name_to) {
            const NormalizedLandmark& lmk_from = landmarks.landmark(std::find(_landmark_names.begin(), _landmark_names.end(), name_from) - _landmark_names.begin());
            const NormalizedLandmark& lmk_to = landmarks.landmark(std::find(_landmark_names.begin(), _landmark_names.end(), name_to) - _landmark_names.begin());
            return get_distance(lmk_from, lmk_to);
        }

        std::vector<float> PoseEmbedder::get_distance(const NormalizedLandmark& lmk_from, const NormalizedLandmark& lmk_to) {
            std::vector<float> lmk_dist(3);
            lmk_dist[0] = lmk_to.x() - lmk_from.x();
            lmk_dist[1] = lmk_to.y() - lmk_from.y();
            lmk_dist[2] = lmk_to.z() - lmk_from.z();
            return lmk_dist;
        }

    }  // namespace

// Converts NormalizedLandmark to Pose. A relative bounding box will
// be created containing all landmarks exactly. A calculator option is provided
// to specify a subset of landmarks for creating the detection.
//
// Input:
//  NOMR_LANDMARKS: A NormalizedLandmarkList proto.
//
// Output:
//   POSE: A Pose (int).
//
// Example config:
// node {
//   calculator: "PoseRecognitionCalculator"
//   input_stream: "LANDMARKS:landmarks"
//   output_stream: "POSE:pose"
// }
    class PoseRecognitionCalculator : public CalculatorBase {
    public:
        static absl::Status GetContract(CalculatorContract* cc);
        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;

    };
    REGISTER_CALCULATOR(PoseRecognitionCalculator);

    absl::Status PoseRecognitionCalculator::GetContract(
            CalculatorContract* cc) {
        RET_CHECK(cc->Inputs().HasTag(kNormalizedLandmarksTag));
        RET_CHECK(cc->Outputs().HasTag(kPoseTag));
        // TODO: Also support converting Landmark to Detection.
        cc->Inputs().Tag(kNormalizedLandmarksTag).Set<NormalizedLandmarkList>();
        cc->Outputs().Tag(kPoseTag).Set<int>();

        return absl::OkStatus();
    }

    absl::Status PoseRecognitionCalculator::Open(CalculatorContext* cc) {
        cc->SetOffset(TimestampDiff(0));

        return absl::OkStatus();
    }

    absl::Status PoseRecognitionCalculator::Process(CalculatorContext* cc) {
        const auto& landmarks =
                cc->Inputs().Tag(kNormalizedLandmarksTag).Get<NormalizedLandmarkList>();
        RET_CHECK_GT(landmarks.landmark_size(), 0)
                << "Input landmark vector is empty.";
        std::unique_ptr<int> result;
        result = absl::make_unique<int>(ConvertLandmarksToPose(landmarks));
        cc->Outputs()
                .Tag(kPoseTag)
                .Add(result.release(), cc->InputTimestamp());

        return absl::OkStatus();
    }

}  // namespace mediapipe
