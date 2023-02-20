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

    }  // namespace

// Converts NormalizedLandmark to Detection proto. A relative bounding box will
// be created containing all landmarks exactly. A calculator option is provided
// to specify a subset of landmarks for creating the detection.
//
// Input:
//  NOMR_LANDMARKS: A NormalizedLandmarkList proto.
//
// Output:
//   POSE: A Posse (int).
//
// Example config:
// node {
//   calculator: "PoseRecognitionCalculator"
//   input_stream: "NORM_LANDMARKS:landmarks"
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
